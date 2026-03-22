#!/usr/bin/env python3
"""
Cross-lingual embedding alignment for Text JEPA — v3.
Pushes alignment quality with multiple methods and selects the best.

Techniques:
  1. Embedding whitening (center + PCA + normalize)
  2. Automatic cognate mining (edit distance)
  3. Iterative Procrustes refinement (CSLS mutual-NN dictionary induction)
  4. Wasserstein-Procrustes (Sinkhorn optimal transport + rotation)
  5. Unsupervised initialization (VecMap sorted similarity profiles)
  6. WGAN adversarial (Wasserstein loss + weight clipping)

Usage: python3 align_embeddings.py
"""

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from prepare_text_jepa import (
    BOS_ID,
    PAD_ID,
    BPETokenizer,
    load_sentences,
    extract_words,
    MAX_CTX_LEN,
)
from train_text_jepa import (
    N_EMBD,
    DEPTH,
    N_HEAD,
    TextEncoder,
    load_encoder,
    l2_normalize,
    norm,
    DATA_DIR,
)
from evaluate_alignment import GOLD_DICT

# ---------------------------------------------------------------------------
# Seed dictionary (English-Finnish) for supervised Procrustes
# ---------------------------------------------------------------------------

SEED_DICT = [
    ("the", "se"), ("and", "ja"), ("of", "on"), ("in", "sisällä"),
    ("water", "vesi"), ("earth", "maa"), ("city", "kaupunki"),
    ("king", "kuningas"), ("year", "vuosi"), ("war", "sota"),
    ("people", "ihmiset"), ("country", "maa"), ("time", "aika"),
    ("world", "maailma"), ("north", "pohjoinen"), ("south", "etelä"),
    ("east", "itä"), ("west", "länsi"), ("sea", "meri"), ("land", "maa"),
    ("state", "valtio"), ("power", "voima"), ("army", "armeija"),
    ("new", "uusi"), ("old", "vanha"), ("great", "suuri"),
    ("small", "pieni"), ("long", "pitkä"), ("first", "ensimmäinen"),
    ("last", "viimeinen"), ("part", "osa"), ("day", "päivä"),
    ("night", "yö"), ("man", "mies"), ("woman", "nainen"),
    ("death", "kuolema"), ("life", "elämä"), ("name", "nimi"),
    ("use", "käyttö"), ("end", "loppu"), ("form", "muoto"),
    ("area", "alue"), ("number", "numero"), ("group", "ryhmä"),
    ("music", "musiikki"), ("history", "historia"), ("language", "kieli"),
    ("system", "järjestelmä"), ("church", "kirkko"), ("river", "joki"),
    ("island", "saari"), ("mountain", "vuori"), ("forest", "metsä"),
    ("school", "koulu"), ("road", "tie"), ("house", "talo"),
    ("family", "perhe"), ("work", "työ"), ("large", "suuri"),
    ("high", "korkea"), ("also", "myös"), ("between", "välillä"),
    ("two", "kaksi"), ("three", "kolme"), ("one", "yksi"),
    ("many", "monta"), ("other", "muu"), ("more", "enemmän"),
    ("after", "jälkeen"), ("before", "ennen"), ("under", "alla"),
    ("over", "yli"), ("about", "noin"), ("same", "sama"),
    ("different", "erilainen"), ("own", "oma"), ("most", "eniten"),
    ("both", "molemmat"), ("each", "jokainen"), ("some", "joitakin"),
    ("known", "tunnettu"), ("called", "nimeltään"), ("made", "tehty"),
    ("used", "käytetty"), ("second", "toinen"), ("order", "järjestys"),
    ("early", "aikainen"), ("left", "vasen"), ("right", "oikea"),
    ("period", "kausi"), ("place", "paikka"), ("however", "kuitenkin"),
    ("military", "sotilaallinen"), ("south", "etelä"), ("million", "miljoona"),
    ("century", "vuosisata"), ("region", "alue"), ("began", "alkoi"),
    ("general", "yleinen"), ("later", "myöhemmin"), ("modern", "moderni"),
    ("political", "poliittinen"), ("important", "tärkeä"),
]


# ---------------------------------------------------------------------------
# Preprocessing: whitening
# ---------------------------------------------------------------------------


def whiten_embeddings(W):
    """Center, PCA-whiten, L2-normalize.

    SVD of centered W gives U @ diag(S) @ Vt.
    U is the whitened representation (decorrelated, equal variance per dim).
    L2-normalizing puts all embeddings on the unit hypersphere.
    """
    W = W - W.mean(axis=0)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    # U has orthonormal columns; rows are whitened embeddings
    W_white = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)
    return W_white


# ---------------------------------------------------------------------------
# Cognate mining
# ---------------------------------------------------------------------------


def levenshtein(s1, s2):
    """Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def mine_cognates(en_words, fi_words, en_idx, fi_idx,
                  max_ratio=0.35, min_len=5, max_pairs=60):
    """Find likely cognates via normalized edit distance."""
    candidates = []
    for en_w in en_words:
        if len(en_w) < min_len or en_w not in en_idx:
            continue
        for fi_w in fi_words:
            if len(fi_w) < min_len or fi_w not in fi_idx:
                continue
            # Length filter: skip very different lengths
            if max(len(en_w), len(fi_w)) > 2 * min(len(en_w), len(fi_w)):
                continue
            d = levenshtein(en_w, fi_w)
            ratio = d / max(len(en_w), len(fi_w))
            if ratio <= max_ratio:
                candidates.append((en_w, fi_w, ratio))

    candidates.sort(key=lambda x: x[2])
    used_en, used_fi = set(), set()
    pairs = []
    for en_w, fi_w, _ in candidates:
        if en_w not in used_en and fi_w not in used_fi:
            pairs.append((en_w, fi_w))
            used_en.add(en_w)
            used_fi.add(fi_w)
        if len(pairs) >= max_pairs:
            break
    return pairs


# ---------------------------------------------------------------------------
# Contextual word embedding extraction
# ---------------------------------------------------------------------------


def extract_contextual_embeddings(encoder, tokenizer, sentences, target_words,
                                  batch_size=64):
    """Extract word embeddings from sentence context (hidden states).

    For each sentence: BPE-encode with word boundaries, run encoder,
    average hidden states at each word's BPE positions. Accumulate
    across all sentences and return mean embeddings.
    """
    target_set = set(target_words)
    word_sums = {}
    word_counts = {}
    max_len = MAX_CTX_LEN

    for start in range(0, len(sentences), batch_size):
        end = min(start + batch_size, len(sentences))
        batch_sents = sentences[start:end]

        batch_ids, batch_masks, batch_word_spans = [], [], []
        for sent in batch_sents:
            ids, word_spans = tokenizer.encode_sentence_with_word_boundaries(sent)
            ids = [BOS_ID] + ids
            word_spans = [(w, s + 1, e + 1) for w, s, e in word_spans]
            n = len(ids)
            if n > max_len:
                ids = ids[:max_len]
                n = max_len
            mask = [1.0] * n + [0.0] * (max_len - n)
            ids = ids + [PAD_ID] * (max_len - n)
            batch_ids.append(ids)
            batch_masks.append(mask)
            batch_word_spans.append(word_spans)

        tokens = mx.array(batch_ids, dtype=mx.int32)
        masks = mx.array(batch_masks, dtype=mx.float32)
        _, hidden = encoder.encode_full(tokens, masks)
        mx.eval(hidden)
        hidden_np = np.array(hidden)

        for i, word_spans in enumerate(batch_word_spans):
            for word, ws, we in word_spans:
                if word not in target_set or we > max_len:
                    continue
                emb = hidden_np[i, ws:we].mean(axis=0)
                if word not in word_sums:
                    word_sums[word] = np.zeros_like(emb)
                    word_counts[word] = 0
                word_sums[word] += emb
                word_counts[word] += 1

    words_found = [w for w in target_words if word_counts.get(w, 0) >= 3]
    W = np.stack([word_sums[w] / word_counts[w] for w in words_found])
    return W, words_found


# ---------------------------------------------------------------------------
# Procrustes alignment
# ---------------------------------------------------------------------------


def procrustes_align(W_src, W_tgt, src_indices, tgt_indices):
    """Orthogonal Procrustes: W* = argmin ||X @ W - Y||_F, s.t. W^T W = I."""
    X = W_src[src_indices]
    Y = W_tgt[tgt_indices]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    M = Y.T @ X
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def nearest_orthogonal(W):
    """Project W to nearest orthogonal matrix via SVD."""
    U, _, Vt = np.linalg.svd(W)
    return U @ Vt


# ---------------------------------------------------------------------------
# CSLS scoring
# ---------------------------------------------------------------------------


def csls_score(W_src, W_tgt, k=10):
    """Cross-domain Similarity Local Scaling — reduces hubness."""
    src_n = W_src / (np.linalg.norm(W_src, axis=1, keepdims=True) + 1e-8)
    tgt_n = W_tgt / (np.linalg.norm(W_tgt, axis=1, keepdims=True) + 1e-8)
    sims = src_n @ tgt_n.T
    k_src = min(k, sims.shape[1])
    k_tgt = min(k, sims.shape[0])
    r_src = np.mean(np.sort(sims, axis=1)[:, -k_src:], axis=1)
    r_tgt = np.mean(np.sort(sims, axis=0)[-k_tgt:, :], axis=0)
    return 2 * sims - r_src[:, None] - r_tgt[None, :]


def mean_csls_score(W_src, W_tgt, W_map):
    """Mean CSLS score (higher = better alignment)."""
    mapped = W_src @ W_map
    csls = csls_score(mapped, W_tgt)
    # Mean of max CSLS per source word
    return float(np.mean(np.max(csls, axis=1)))


# ---------------------------------------------------------------------------
# Iterative Procrustes refinement (MUSE-style)
# ---------------------------------------------------------------------------


def iterative_procrustes(W_src, W_tgt, W_init, n_iter=15):
    """Refine alignment via CSLS mutual-NN dictionary induction.

    Each iteration: map src → CSLS → find mutual nearest neighbors →
    use as synthetic dictionary → re-run Procrustes.
    Tracks best alignment across iterations (avoids degradation).
    """
    W = W_init.copy()
    best_W = W.copy()
    best_score = mean_csls_score(W_src, W_tgt, W)

    for it in range(n_iter):
        mapped = W_src @ W
        csls = csls_score(mapped, W_tgt)

        # Mutual nearest neighbors
        fwd = np.argmax(csls, axis=1)  # for each src, best tgt
        bwd = np.argmax(csls, axis=0)  # for each tgt, best src
        pairs = [(i, fwd[i]) for i in range(len(fwd)) if bwd[fwd[i]] == i]

        if len(pairs) < 5:
            break

        src_idx = np.array([p[0] for p in pairs])
        tgt_idx = np.array([p[1] for p in pairs])
        W_new = procrustes_align(W_src, W_tgt, src_idx, tgt_idx)

        score_new = mean_csls_score(W_src, W_tgt, W_new)
        if score_new > best_score:
            best_W = W_new.copy()
            best_score = score_new

        if np.allclose(W, W_new, atol=1e-6):
            break
        W = W_new

    return best_W


# ---------------------------------------------------------------------------
# Wasserstein-Procrustes (Grave et al. 2019)
# ---------------------------------------------------------------------------


def sinkhorn(cost, n_iter=50, epsilon=0.05):
    """Sinkhorn-Knopp for entropy-regularized optimal transport.

    Returns doubly-stochastic transport plan P.
    """
    N, M = cost.shape
    # Log-domain Sinkhorn for numerical stability
    log_K = -cost / epsilon
    log_u = np.zeros(N)
    log_v = np.zeros(M)

    for _ in range(n_iter):
        # log_u = -log(N) - logsumexp(log_K + log_v, axis=1)
        lk_v = log_K + log_v[None, :]
        max_lk_v = np.max(lk_v, axis=1, keepdims=True)
        log_u = -np.log(N) - (max_lk_v.squeeze() + np.log(
            np.sum(np.exp(lk_v - max_lk_v), axis=1)))

        # log_v = -log(M) - logsumexp(log_K + log_u, axis=0)
        lk_u = log_K + log_u[:, None]
        max_lk_u = np.max(lk_u, axis=0, keepdims=True)
        log_v = -np.log(M) - (max_lk_u.squeeze() + np.log(
            np.sum(np.exp(lk_u - max_lk_u), axis=0)))

    log_P = log_u[:, None] + log_K + log_v[None, :]
    return np.exp(log_P)


def wasserstein_procrustes(W_src, W_tgt, n_iter=20, epsilon=0.05):
    """Wasserstein-Procrustes: alternate Sinkhorn OT ↔ Procrustes rotation.

    Fully unsupervised — finds correspondences via optimal transport,
    then finds the best rotation to match them.
    """
    D = W_src.shape[1]
    src_n = W_src / (np.linalg.norm(W_src, axis=1, keepdims=True) + 1e-8)
    tgt_n = W_tgt / (np.linalg.norm(W_tgt, axis=1, keepdims=True) + 1e-8)

    R = np.eye(D, dtype=np.float32)

    for i in range(n_iter):
        # Cost matrix: 1 - cosine similarity
        mapped = src_n @ R
        cost = 1.0 - mapped @ tgt_n.T

        # Sinkhorn optimal transport
        P = sinkhorn(cost, n_iter=50, epsilon=epsilon)

        # Procrustes: find R minimizing ||src @ R - P @ tgt||
        M = tgt_n.T @ P.T @ src_n
        U, _, Vt = np.linalg.svd(M)
        R_new = U @ Vt

        if np.allclose(R, R_new, atol=1e-6):
            break
        R = R_new

    return R


# ---------------------------------------------------------------------------
# Unsupervised initialization (VecMap-style similarity profiles)
# ---------------------------------------------------------------------------


def unsupervised_init(W_src, W_tgt, n_seeds=200):
    """VecMap-style unsupervised initialization.

    Key insight: sorted intra-language similarity profiles are
    language-invariant. Words with similar "popularity" profiles
    across their own language's vocabulary are likely translations.
    """
    src_n = W_src / (np.linalg.norm(W_src, axis=1, keepdims=True) + 1e-8)
    tgt_n = W_tgt / (np.linalg.norm(W_tgt, axis=1, keepdims=True) + 1e-8)

    # Intra-language similarity matrices
    sim_src = src_n @ src_n.T
    sim_tgt = tgt_n @ tgt_n.T

    # Sort each row (descending similarity profile)
    sorted_src = np.sort(sim_src, axis=1)[:, ::-1]
    sorted_tgt = np.sort(sim_tgt, axis=1)[:, ::-1]

    # Truncate to same length
    K = min(sorted_src.shape[1], sorted_tgt.shape[1])
    sorted_src = sorted_src[:, :K]
    sorted_tgt = sorted_tgt[:, :K]

    # Normalize profiles
    sorted_src = sorted_src / (np.linalg.norm(sorted_src, axis=1, keepdims=True) + 1e-8)
    sorted_tgt = sorted_tgt / (np.linalg.norm(sorted_tgt, axis=1, keepdims=True) + 1e-8)

    # Cross-lingual profile similarity
    profile_sims = sorted_src @ sorted_tgt.T

    # Mutual nearest neighbors in profile space
    fwd = np.argmax(profile_sims, axis=1)
    bwd = np.argmax(profile_sims, axis=0)
    pairs = [(i, fwd[i]) for i in range(len(fwd)) if bwd[fwd[i]] == i]
    pairs.sort(key=lambda p: -profile_sims[p[0], p[1]])
    pairs = pairs[:n_seeds]

    if len(pairs) < 5:
        return np.eye(W_src.shape[1], dtype=np.float32)

    src_idx = np.array([p[0] for p in pairs])
    tgt_idx = np.array([p[1] for p in pairs])
    return procrustes_align(W_src, W_tgt, src_idx, tgt_idx)


# ---------------------------------------------------------------------------
# WGAN adversarial alignment
# ---------------------------------------------------------------------------


class Discriminator(nn.Module):
    """Critic for WGAN: outputs unbounded score (no sigmoid)."""

    def __init__(self, n_embd, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, 1, bias=True)

    def __call__(self, x):
        h = nn.leaky_relu(self.fc1(x), negative_slope=0.2)
        h = nn.leaky_relu(self.fc2(h), negative_slope=0.2)
        return self.fc3(h).squeeze(-1)


class MappingMatrix(nn.Module):
    """Learnable orthogonal mapping W (D, D)."""

    def __init__(self, d):
        super().__init__()
        self.W = mx.eye(d)

    def __call__(self, x):
        return x @ self.W


def _sgd_step(module, grads, lr):
    """Single SGD step on module parameters."""
    flat_p = dict(tree_flatten(module.parameters()))
    flat_g = dict(tree_flatten(grads))
    for path, param in flat_p.items():
        if path in flat_g:
            _set_nested(module, path, param - lr * flat_g[path])
    mx.eval(module.parameters())


def _set_nested(obj, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        if isinstance(obj, list):
            obj = obj[int(part)]
        elif isinstance(obj, dict):
            obj = obj[part]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if isinstance(obj, dict):
        obj[last] = value
    else:
        setattr(obj, last, value)


def adversarial_align(W_src, W_tgt, n_epochs=80, batch_size=64,
                      lr_d=0.001, lr_w=0.001, n_critic=3, clip_val=0.05):
    """WGAN adversarial alignment with weight clipping.

    Wasserstein loss: D tries to maximize E[D(tgt)] - E[D(mapped_src)].
    W tries to minimize -E[D(mapped_src)] (fool discriminator).
    Weight clipping enforces Lipschitz constraint on D.
    """
    D = W_src.shape[1]
    N_src, N_tgt = W_src.shape[0], W_tgt.shape[0]

    src_n = W_src / (np.linalg.norm(W_src, axis=1, keepdims=True) + 1e-8)
    tgt_n = W_tgt / (np.linalg.norm(W_tgt, axis=1, keepdims=True) + 1e-8)
    src_all = mx.array(src_n, dtype=mx.float32)
    tgt_all = mx.array(tgt_n, dtype=mx.float32)

    mapping = MappingMatrix(D)
    disc = Discriminator(D)
    mx.eval(mapping.parameters(), disc.parameters())

    # Wasserstein critic loss: minimize E[D(fake)] - E[D(real)]
    def disc_loss_fn(d, mapped_src, tgt):
        return mx.mean(d(mapped_src)) - mx.mean(d(tgt))

    # Mapping loss: minimize -E[D(fake)] (fool critic)
    def w_loss_fn(m, src, d):
        return -mx.mean(d(m(src)))

    disc_vg = nn.value_and_grad(disc, disc_loss_fn)
    w_vg = nn.value_and_grad(mapping, w_loss_fn)

    best_W = np.eye(D, dtype=np.float32)
    best_score = -1.0

    for epoch in range(n_epochs):
        src_perm = np.random.permutation(N_src)
        tgt_perm = np.random.permutation(N_tgt)
        n_batches = min(N_src, N_tgt) // batch_size
        d_losses, w_losses = [], []

        for b in range(n_batches):
            si = src_perm[b * batch_size:(b + 1) * batch_size]
            ti = tgt_perm[b * batch_size:(b + 1) * batch_size]
            src_batch = src_all[mx.array(si)]
            tgt_batch = tgt_all[mx.array(ti)]

            # --- Critic steps (multiple per generator step) ---
            for _ in range(n_critic):
                mapped = mx.stop_gradient(mapping(src_batch))
                d_loss, d_grads = disc_vg(disc, mapped, tgt_batch)
                mx.eval(d_loss, d_grads)
                d_losses.append(float(d_loss.item()))
                _sgd_step(disc, d_grads, lr_d)

                # Weight clipping (Lipschitz constraint)
                for path, param in tree_flatten(disc.parameters()):
                    _set_nested(disc, path, mx.clip(param, -clip_val, clip_val))
                mx.eval(disc.parameters())

            # --- Generator (mapping) step ---
            w_loss, w_grads = w_vg(mapping, src_batch, disc)
            mx.eval(w_loss, w_grads)
            w_losses.append(float(w_loss.item()))
            _sgd_step(mapping, w_grads, lr_w)

        # Orthogonal projection after each epoch
        W_np = np.array(mapping.W)
        W_np = nearest_orthogonal(W_np)
        mapping.W = mx.array(W_np, dtype=mx.float32)
        mx.eval(mapping.parameters())

        # Track best
        mapped_all = src_n @ W_np
        sims = mapped_all @ tgt_n.T
        score = float(np.mean(np.max(sims, axis=1)))
        if score > best_score:
            best_score = score
            best_W = W_np.copy()

        if (epoch + 1) % 10 == 0:
            md = np.mean(d_losses[-n_batches * n_critic:]) if d_losses else 0
            mw = np.mean(w_losses[-n_batches:]) if w_losses else 0
            print(f"    Epoch {epoch+1}/{n_epochs}: d={md:.4f} w={mw:.4f} score={score:.4f}")

    return best_W


def adversarial_align_seeded(W_src, W_tgt, W_init, n_epochs=40, batch_size=64,
                             lr_d=0.001, lr_w=0.001, n_critic=3, clip_val=0.05):
    """WGAN starting from a Procrustes-seeded mapping instead of identity."""
    D = W_src.shape[1]
    N_src, N_tgt = W_src.shape[0], W_tgt.shape[0]

    src_n = W_src / (np.linalg.norm(W_src, axis=1, keepdims=True) + 1e-8)
    tgt_n = W_tgt / (np.linalg.norm(W_tgt, axis=1, keepdims=True) + 1e-8)
    src_all = mx.array(src_n, dtype=mx.float32)
    tgt_all = mx.array(tgt_n, dtype=mx.float32)

    mapping = MappingMatrix(D)
    mapping.W = mx.array(W_init.astype(np.float32))
    disc = Discriminator(D)
    mx.eval(mapping.parameters(), disc.parameters())

    def disc_loss_fn(d, mapped_src, tgt):
        return mx.mean(d(mapped_src)) - mx.mean(d(tgt))

    def w_loss_fn(m, src, d):
        return -mx.mean(d(m(src)))

    disc_vg = nn.value_and_grad(disc, disc_loss_fn)
    w_vg = nn.value_and_grad(mapping, w_loss_fn)

    best_W = W_init.copy()
    best_score = -1.0

    for epoch in range(n_epochs):
        src_perm = np.random.permutation(N_src)
        tgt_perm = np.random.permutation(N_tgt)
        n_batches = min(N_src, N_tgt) // batch_size

        for b in range(n_batches):
            si = src_perm[b * batch_size:(b + 1) * batch_size]
            ti = tgt_perm[b * batch_size:(b + 1) * batch_size]
            src_batch = src_all[mx.array(si)]
            tgt_batch = tgt_all[mx.array(ti)]

            for _ in range(n_critic):
                mapped = mx.stop_gradient(mapping(src_batch))
                d_loss, d_grads = disc_vg(disc, mapped, tgt_batch)
                mx.eval(d_loss, d_grads)
                _sgd_step(disc, d_grads, lr_d)
                for path, param in tree_flatten(disc.parameters()):
                    _set_nested(disc, path, mx.clip(param, -clip_val, clip_val))
                mx.eval(disc.parameters())

            w_loss, w_grads = w_vg(mapping, src_batch, disc)
            mx.eval(w_loss, w_grads)
            _sgd_step(mapping, w_grads, lr_w)

        W_np = np.array(mapping.W)
        W_np = nearest_orthogonal(W_np)
        mapping.W = mx.array(W_np, dtype=mx.float32)
        mx.eval(mapping.parameters())

        mapped_all = src_n @ W_np
        sims = mapped_all @ tgt_n.T
        score = float(np.mean(np.max(sims, axis=1)))
        if score > best_score:
            best_score = score
            best_W = W_np.copy()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: score={score:.4f}")

    return best_W


# ---------------------------------------------------------------------------
# Quick evaluation helper
# ---------------------------------------------------------------------------


def quick_eval(W_src, W_tgt, W_map, words_src, words_tgt, gold_dict):
    """Quick P@1/5/10 evaluation using CSLS."""
    src_idx = {w: i for i, w in enumerate(words_src)}
    tgt_idx = {w: i for i, w in enumerate(words_tgt)}

    mapped = W_src @ W_map
    csls = csls_score(mapped, W_tgt)

    hits = {1: 0, 5: 0, 10: 0}
    n = 0
    for en_w, fi_ws in gold_dict.items():
        if en_w not in src_idx:
            continue
        valid = [tgt_idx[w] for w in fi_ws if w in tgt_idx]
        if not valid:
            continue
        n += 1
        nn = np.argsort(csls[src_idx[en_w]])[::-1]
        for k in [1, 5, 10]:
            if any(idx in valid for idx in nn[:k]):
                hits[k] += 1

    if n == 0:
        return {k: 0.0 for k in [1, 5, 10]}, 0
    return {k: hits[k] / n for k in [1, 5, 10]}, n


def print_nn_analysis(W_src, W_tgt, W_map, words_src, words_tgt, gold_dict,
                      n_words=20):
    """Print detailed nearest-neighbor analysis."""
    src_idx = {w: i for i, w in enumerate(words_src)}
    tgt_idx = {w: i for i, w in enumerate(words_tgt)}

    mapped = W_src @ W_map
    csls = csls_score(mapped, W_tgt)

    test_words = sorted(gold_dict.keys())[:n_words]
    for en_w in test_words:
        if en_w not in src_idx:
            continue
        nn_indices = np.argsort(csls[src_idx[en_w]])[::-1][:5]
        neighbors = [words_tgt[i] for i in nn_indices]
        expected = gold_dict[en_w]
        hit = any(n in expected for n in neighbors)
        marker = "V" if hit else "X"
        print(f"  [{marker}] {en_w:15s} -> {', '.join(neighbors)}"
              f"  (exp: {', '.join(expected[:2])})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    np.random.seed(42)

    print("=" * 60)
    print("Cross-lingual alignment v3: EN <-> FI (all methods)")
    print("=" * 60)

    tokenizer = BPETokenizer.load()

    # Load and extract contextual embeddings
    print("\nLoading corpora and extracting embeddings...")
    en_sents = load_sentences("en")
    fi_sents = load_sentences("fi")
    en_words_all = extract_words(en_sents, min_freq=3)
    fi_words_all = extract_words(fi_sents, min_freq=3)

    en_enc = load_encoder("en", vocab_size=tokenizer.vocab_size)
    fi_enc = load_encoder("fi", vocab_size=tokenizer.vocab_size)

    W_en_raw, en_words = extract_contextual_embeddings(
        en_enc, tokenizer, en_sents, en_words_all)
    W_fi_raw, fi_words = extract_contextual_embeddings(
        fi_enc, tokenizer, fi_sents, fi_words_all)

    print(f"  EN: {W_en_raw.shape}, FI: {W_fi_raw.shape}")

    # Whiten embeddings (center + PCA + normalize)
    print("\nWhitening embeddings...")
    W_en = whiten_embeddings(W_en_raw)
    W_fi = whiten_embeddings(W_fi_raw)

    # Effective dimensionality check
    for name, W in [("EN-raw", W_en_raw), ("EN-white", W_en), ("FI-raw", W_fi_raw), ("FI-white", W_fi)]:
        Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
        _, S, _ = np.linalg.svd(Wn, full_matrices=False)
        eff_dim = (np.sum(S ** 2) ** 2) / np.sum(S ** 4)
        intra = Wn @ Wn.T
        np.fill_diagonal(intra, 0)
        print(f"  {name:10s}: eff_dim={eff_dim:.1f}, intra_sim={np.mean(intra):.4f}")

    # Save whitened embeddings (evaluate_alignment.py reads these)
    np.savez_compressed(
        os.path.join(DATA_DIR, "word_embeddings.npz"),
        W_en=W_en, W_fi=W_fi,
        words_en=np.array(en_words, dtype=object),
        words_fi=np.array(fi_words, dtype=object),
    )

    en_idx = {w: i for i, w in enumerate(en_words)}
    fi_idx = {w: i for i, w in enumerate(fi_words)}

    # Mine cognates
    print("\nMining cognates via edit distance...")
    cognates = mine_cognates(en_words, fi_words, en_idx, fi_idx)
    print(f"  Found {len(cognates)} cognate pairs:")
    for en_w, fi_w in cognates[:15]:
        print(f"    {en_w} <-> {fi_w}")
    if len(cognates) > 15:
        print(f"    ... and {len(cognates) - 15} more")

    # Combine seed pairs: cognates + manual dictionary
    seed_src, seed_tgt = [], []
    used_en, used_fi = set(), set()

    for en_w, fi_w in cognates:
        if en_w in en_idx and fi_w in fi_idx:
            seed_src.append(en_idx[en_w])
            seed_tgt.append(fi_idx[fi_w])
            used_en.add(en_w)
            used_fi.add(fi_w)

    for en_w, fi_w in SEED_DICT:
        if en_w in en_idx and fi_w in fi_idx and en_w not in used_en and fi_w not in used_fi:
            seed_src.append(en_idx[en_w])
            seed_tgt.append(fi_idx[fi_w])
            used_en.add(en_w)
            used_fi.add(fi_w)

    seed_src = np.array(seed_src)
    seed_tgt = np.array(seed_tgt)
    print(f"\n  Total seed pairs (cognates + dict): {len(seed_src)}")

    # =====================================================================
    # Run all alignment methods
    # =====================================================================

    results = {}

    # 1. Procrustes (supervised)
    print("\n--- 1. Procrustes (supervised) ---")
    W_proc = procrustes_align(W_en, W_fi, seed_src, seed_tgt)
    p, n = quick_eval(W_en, W_fi, W_proc, en_words, fi_words, GOLD_DICT)
    print(f"  P@1={p[1]*100:.1f}% P@5={p[5]*100:.1f}% P@10={p[10]*100:.1f}% ({n} pairs)")
    results["Procrustes"] = (W_proc, p)

    # 2. Procrustes + iterative refinement
    print("\n--- 2. Procrustes + iterative refinement ---")
    W_proc_r = iterative_procrustes(W_en, W_fi, W_proc)
    p, n = quick_eval(W_en, W_fi, W_proc_r, en_words, fi_words, GOLD_DICT)
    print(f"  P@1={p[1]*100:.1f}% P@5={p[5]*100:.1f}% P@10={p[10]*100:.1f}% ({n} pairs)")
    results["Procrustes+refine"] = (W_proc_r, p)

    # 3. Wasserstein-Procrustes (unsupervised)
    print("\n--- 3. Wasserstein-Procrustes (unsupervised) ---")
    W_wass = wasserstein_procrustes(W_en, W_fi)
    p, n = quick_eval(W_en, W_fi, W_wass, en_words, fi_words, GOLD_DICT)
    print(f"  P@1={p[1]*100:.1f}% P@5={p[5]*100:.1f}% P@10={p[10]*100:.1f}% ({n} pairs)")
    results["Wasserstein-Procrustes"] = (W_wass, p)

    # 4. Wasserstein-Procrustes + iterative refinement
    print("\n--- 4. WP + iterative refinement ---")
    W_wass_r = iterative_procrustes(W_en, W_fi, W_wass)
    p, n = quick_eval(W_en, W_fi, W_wass_r, en_words, fi_words, GOLD_DICT)
    print(f"  P@1={p[1]*100:.1f}% P@5={p[5]*100:.1f}% P@10={p[10]*100:.1f}% ({n} pairs)")
    results["WP+refine"] = (W_wass_r, p)

    # 5. Unsupervised init (similarity profiles) + refinement
    print("\n--- 5. Unsupervised init (VecMap profiles) + refinement ---")
    W_unsup = unsupervised_init(W_en, W_fi)
    p0, _ = quick_eval(W_en, W_fi, W_unsup, en_words, fi_words, GOLD_DICT)
    print(f"  Init: P@1={p0[1]*100:.1f}% P@5={p0[5]*100:.1f}% P@10={p0[10]*100:.1f}%")
    W_unsup_r = iterative_procrustes(W_en, W_fi, W_unsup)
    p, n = quick_eval(W_en, W_fi, W_unsup_r, en_words, fi_words, GOLD_DICT)
    print(f"  Refined: P@1={p[1]*100:.1f}% P@5={p[5]*100:.1f}% P@10={p[10]*100:.1f}% ({n} pairs)")
    results["VecMap+refine"] = (W_unsup_r, p)

    # 6. WGAN adversarial
    print("\n--- 6. WGAN adversarial ---")
    W_adv = adversarial_align(W_en, W_fi)
    p, n = quick_eval(W_en, W_fi, W_adv, en_words, fi_words, GOLD_DICT)
    print(f"  P@1={p[1]*100:.1f}% P@5={p[5]*100:.1f}% P@10={p[10]*100:.1f}% ({n} pairs)")
    results["Adversarial"] = (W_adv, p)

    # 7. Adversarial + iterative refinement
    print("\n--- 7. Adversarial + iterative refinement ---")
    W_adv_r = iterative_procrustes(W_en, W_fi, W_adv)
    p, n = quick_eval(W_en, W_fi, W_adv_r, en_words, fi_words, GOLD_DICT)
    print(f"  P@1={p[1]*100:.1f}% P@5={p[5]*100:.1f}% P@10={p[10]*100:.1f}% ({n} pairs)")
    results["Adversarial+refine"] = (W_adv_r, p)

    # 8. Procrustes-seeded adversarial + refinement
    print("\n--- 8. Procrustes-seeded WGAN + refinement ---")
    W_adv2 = adversarial_align_seeded(W_en, W_fi, W_proc)
    W_adv2_r = iterative_procrustes(W_en, W_fi, W_adv2)
    p, n = quick_eval(W_en, W_fi, W_adv2_r, en_words, fi_words, GOLD_DICT)
    print(f"  P@1={p[1]*100:.1f}% P@5={p[5]*100:.1f}% P@10={p[10]*100:.1f}% ({n} pairs)")
    results["Proc-seeded-WGAN+refine"] = (W_adv2_r, p)

    # =====================================================================
    # Summary
    # =====================================================================

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Method':<30} {'P@1':>6} {'P@5':>6} {'P@10':>6}")
    print("  " + "-" * 54)

    best_name, best_p10 = None, -1
    for name, (W, p) in results.items():
        print(f"  {name:<30} {p[1]*100:5.1f}% {p[5]*100:5.1f}% {p[10]*100:5.1f}%")
        score = p[10] + p[5] * 0.5 + p[1] * 0.25  # weighted score
        if score > best_p10:
            best_name = name
            best_p10 = score

    print(f"\n  Best method: {best_name}")

    # Save best as W_procrustes.npy (evaluate_alignment.py compatibility)
    best_W = results[best_name][0]
    np.save(os.path.join(DATA_DIR, "W_procrustes.npy"), best_W)

    # Save adversarial result for comparison
    adv_key = "Adversarial+refine" if "Adversarial+refine" in results else "Adversarial"
    np.save(os.path.join(DATA_DIR, "W_adversarial.npy"), results[adv_key][0])

    # Detailed NN analysis for best method
    print(f"\n--- Nearest-neighbor analysis ({best_name}) ---")
    print_nn_analysis(W_en, W_fi, best_W, en_words, fi_words, GOLD_DICT)

    print(f"\nSaved to {DATA_DIR}")


if __name__ == "__main__":
    main()
