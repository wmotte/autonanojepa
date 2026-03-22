#!/usr/bin/env python3
"""
Unsupervised cross-lingual embedding alignment.

Designed for aligning embedding spaces where no bilingual dictionary,
cognates, or shared script exists (e.g., Voynich manuscript).

Pipeline:
  1. Embedding whitening / partial whitening sweep
  2. Unsupervised initialization (VecMap-style similarity profiles)
  3. Wasserstein-Procrustes (Sinkhorn OT + rotation)
  4. WGAN adversarial alignment
  5. Iterative refinement (CSLS mutual-NN, no supervision needed)
  6. Unsupervised self-learning (bootstrap synthetic dictionary)
  7. Selection via intrinsic metrics (no gold dictionary required)

Intrinsic evaluation:
  - Mean cosine similarity to nearest neighbor
  - Mutual nearest neighbor ratio (alignment symmetry)
  - Hubness skewness (lower = better)
  - Cycle consistency (A→B→A recovery)

Usage: python3 align_unsupervised.py
"""

import os
import numpy as np

from align_embeddings import (
    whiten_embeddings,
    partial_whiten,
    procrustes_align,
    csls_score,
    iterative_procrustes,
    iterative_procrustes_topk,
    wasserstein_procrustes,
    unsupervised_init,
    adversarial_align,
    adversarial_align_seeded,
    extract_contextual_embeddings,
    quick_eval,
    print_nn_analysis,
)
from prepare_text_jepa import BPETokenizer, load_sentences, extract_words
from train_text_jepa import DATA_DIR, load_encoder


# ---------------------------------------------------------------------------
# Intrinsic evaluation metrics (no gold dictionary needed)
# ---------------------------------------------------------------------------


def mean_cosine_nn(W_src, W_tgt, W_map):
    """Mean cosine similarity between each mapped source and its nearest target."""
    mapped = W_src @ W_map
    mapped_n = mapped / (np.linalg.norm(mapped, axis=1, keepdims=True) + 1e-8)
    tgt_n = W_tgt / (np.linalg.norm(W_tgt, axis=1, keepdims=True) + 1e-8)
    sims = mapped_n @ tgt_n.T
    return float(np.mean(np.max(sims, axis=1)))


def mutual_nn_ratio(W_src, W_tgt, W_map):
    """Fraction of source words whose nearest target is a mutual nearest neighbor."""
    mapped = W_src @ W_map
    mapped_n = mapped / (np.linalg.norm(mapped, axis=1, keepdims=True) + 1e-8)
    tgt_n = W_tgt / (np.linalg.norm(W_tgt, axis=1, keepdims=True) + 1e-8)
    sims = mapped_n @ tgt_n.T
    fwd = np.argmax(sims, axis=1)
    bwd = np.argmax(sims, axis=0)
    mutual = sum(1 for i in range(len(fwd)) if bwd[fwd[i]] == i)
    return mutual / len(fwd)


def hubness_skewness(W_src, W_tgt, W_map, k=10):
    """Skewness of k-occurrence distribution (lower = less hubness = better)."""
    mapped = W_src @ W_map
    mapped_n = mapped / (np.linalg.norm(mapped, axis=1, keepdims=True) + 1e-8)
    tgt_n = W_tgt / (np.linalg.norm(W_tgt, axis=1, keepdims=True) + 1e-8)
    sims = mapped_n @ tgt_n.T

    k_eff = min(k, sims.shape[1])
    topk_idx = np.argpartition(-sims, k_eff, axis=1)[:, :k_eff]

    counts = np.zeros(W_tgt.shape[0])
    for row in topk_idx:
        for j in row:
            counts[j] += 1

    mean_c = np.mean(counts)
    std_c = np.std(counts) + 1e-8
    return float(np.mean(((counts - mean_c) / std_c) ** 3))


def cycle_consistency(W_fwd, W_bwd, W_src, n_sample=500):
    """Mean cosine between source and round-tripped (src->tgt->src) embeddings.

    For orthogonal mappings W_bwd = W_fwd.T, this would be perfect.
    When computed with an independently-estimated reverse mapping, it measures
    whether the alignment is consistent in both directions.
    """
    n = min(n_sample, W_src.shape[0])
    idx = np.random.choice(W_src.shape[0], n, replace=False)
    X = W_src[idx]
    X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    X_rt = (X_n @ W_fwd) @ W_bwd
    X_rt = X_rt / (np.linalg.norm(X_rt, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(X_n * X_rt, axis=1)))


def intrinsic_eval(W_src, W_tgt, W_map):
    """Compute all intrinsic metrics. Returns dict."""
    cos = mean_cosine_nn(W_src, W_tgt, W_map)
    mnn = mutual_nn_ratio(W_src, W_tgt, W_map)
    hub = hubness_skewness(W_src, W_tgt, W_map)
    # Composite: cos and mnn are good when high, hubness good when low
    composite = cos * 0.3 + mnn * 0.5 - max(0, hub) / 20.0 * 0.2
    return {"cos": cos, "mnn": mnn, "hub": hub, "score": composite}


def fmt_metrics(m):
    """Format metrics dict as string."""
    return (f"cos={m['cos']:.4f} mnn={m['mnn']:.4f} "
            f"hub={m['hub']:.1f} score={m['score']:.4f}")


# ---------------------------------------------------------------------------
# Unsupervised self-learning (no seed dictionary)
# ---------------------------------------------------------------------------


def unsupervised_self_learning(W_src, W_tgt, W_init, n_rounds=8, verbose=True):
    """Bootstrap synthetic dictionary from mutual NN and iteratively refine.

    Starts from an unsupervised init (VecMap/Wasserstein/adversarial),
    discovers confident mutual nearest neighbors as translation pairs,
    re-aligns with Procrustes on this synthetic dictionary, and repeats.
    """
    W = W_init.copy()
    best_W = W.copy()
    best_score = intrinsic_eval(W_src, W_tgt, W)["score"]

    for rnd in range(n_rounds):
        # Find mutual NN (synthetic dictionary)
        mapped = W_src @ W
        cs = csls_score(mapped, W_tgt)
        fwd = np.argmax(cs, axis=1)
        bwd = np.argmax(cs, axis=0)
        fwd_scores = cs[np.arange(len(fwd)), fwd]

        # Mutual NN with positive CSLS score
        pairs = [(i, fwd[i], fwd_scores[i]) for i in range(len(fwd))
                 if bwd[fwd[i]] == i and fwd_scores[i] > 0]
        pairs.sort(key=lambda x: -x[2])

        if len(pairs) < 10:
            if verbose:
                print(f"    Round {rnd+1}: {len(pairs)} mutual NN — stopping")
            break

        # Use top-half most confident pairs
        n_use = max(50, len(pairs) // 2)
        n_use = min(n_use, len(pairs))
        src_idx = np.array([p[0] for p in pairs[:n_use]])
        tgt_idx = np.array([p[1] for p in pairs[:n_use]])

        # Procrustes on synthetic dictionary + refine
        W_new = procrustes_align(W_src, W_tgt, src_idx, tgt_idx)
        W_new = iterative_procrustes_topk(
            W_src, W_tgt, W_new, n_iter=10, top_k=min(200, n_use))

        m = intrinsic_eval(W_src, W_tgt, W_new)
        if verbose:
            print(f"    Round {rnd+1}: dict={n_use} {fmt_metrics(m)}")

        if m["score"] > best_score:
            best_W = W_new.copy()
            best_score = m["score"]

        if np.allclose(W, W_new, atol=1e-6):
            break
        W = W_new

    return best_W


# ---------------------------------------------------------------------------
# Full unsupervised alignment pipeline for one pair of embedding matrices
# ---------------------------------------------------------------------------


def unsupervised_align(W_src, W_tgt, verbose=True):
    """Run all unsupervised methods, refine each, pick best by intrinsic score.

    Methods:
      1. VecMap-style unsupervised init (similarity profiles)
      2. Wasserstein-Procrustes (Sinkhorn OT, multiple epsilon)
      3. Adversarial WGAN
      4. WGAN seeded from best init
      5. Unsupervised self-learning from best so far

    Each method is refined with iterative Procrustes (mutual-NN / top-K).

    Returns (best_W, results_dict).
    """
    results = {}

    # --- 1. VecMap-style unsupervised init ---
    if verbose:
        print("\n--- 1. VecMap unsupervised init ---")
    W_vm = unsupervised_init(W_src, W_tgt, n_seeds=200)
    m = intrinsic_eval(W_src, W_tgt, W_vm)
    results["VecMap"] = (W_vm, m)
    if verbose:
        print(f"  {fmt_metrics(m)}")

    # + mutual-NN iterative refinement
    W_vm_r = iterative_procrustes(W_src, W_tgt, W_vm, n_iter=20)
    m = intrinsic_eval(W_src, W_tgt, W_vm_r)
    results["VecMap+refine"] = (W_vm_r, m)
    if verbose:
        print(f"  + mutual-NN refine: {fmt_metrics(m)}")

    # + top-K refinement
    W_vm_tk = iterative_procrustes_topk(W_src, W_tgt, W_vm, n_iter=20, top_k=200)
    m = intrinsic_eval(W_src, W_tgt, W_vm_tk)
    results["VecMap+topk"] = (W_vm_tk, m)
    if verbose:
        print(f"  + top-K refine:     {fmt_metrics(m)}")

    # --- 2. Wasserstein-Procrustes (multiple epsilon) ---
    if verbose:
        print("\n--- 2. Wasserstein-Procrustes ---")

    # Subsample for Sinkhorn O(n^2)
    n_ot = min(2000, W_src.shape[0], W_tgt.shape[0])
    src_sub = W_src[np.random.choice(W_src.shape[0], n_ot, replace=False)]
    tgt_sub = W_tgt[np.random.choice(W_tgt.shape[0], n_ot, replace=False)]

    for eps in [0.03, 0.05, 0.1]:
        if verbose:
            print(f"  eps={eps}...")
        W_ws = wasserstein_procrustes(src_sub, tgt_sub, n_iter=30, epsilon=eps)
        m = intrinsic_eval(W_src, W_tgt, W_ws)
        results[f"Wass(e={eps})"] = (W_ws, m)
        if verbose:
            print(f"    {fmt_metrics(m)}")

        # Refine on full vocabulary
        W_ws_r = iterative_procrustes_topk(
            W_src, W_tgt, W_ws, n_iter=20, top_k=200)
        m = intrinsic_eval(W_src, W_tgt, W_ws_r)
        results[f"Wass(e={eps})+refine"] = (W_ws_r, m)
        if verbose:
            print(f"    + refine: {fmt_metrics(m)}")

    # --- 3. Adversarial (WGAN) ---
    if verbose:
        print("\n--- 3. Adversarial (WGAN) ---")
    W_adv = adversarial_align(W_src, W_tgt, n_epochs=80, batch_size=64)
    m = intrinsic_eval(W_src, W_tgt, W_adv)
    results["WGAN"] = (W_adv, m)
    if verbose:
        print(f"  {fmt_metrics(m)}")

    W_adv_r = iterative_procrustes_topk(
        W_src, W_tgt, W_adv, n_iter=20, top_k=200)
    m = intrinsic_eval(W_src, W_tgt, W_adv_r)
    results["WGAN+refine"] = (W_adv_r, m)
    if verbose:
        print(f"  + refine: {fmt_metrics(m)}")

    # --- 4. WGAN seeded from best unsupervised init ---
    base_methods = {k: v for k, v in results.items()
                    if "refine" not in k and "topk" not in k}
    best_init_name = max(base_methods, key=lambda k: base_methods[k][1]["score"])
    best_init_W = base_methods[best_init_name][0]

    if verbose:
        print(f"\n--- 4. WGAN seeded from {best_init_name} ---")
    W_adv_s = adversarial_align_seeded(
        W_src, W_tgt, best_init_W, n_epochs=40)
    m = intrinsic_eval(W_src, W_tgt, W_adv_s)
    results["WGAN-seeded"] = (W_adv_s, m)
    if verbose:
        print(f"  {fmt_metrics(m)}")

    W_adv_sr = iterative_procrustes_topk(
        W_src, W_tgt, W_adv_s, n_iter=20, top_k=200)
    m = intrinsic_eval(W_src, W_tgt, W_adv_sr)
    results["WGAN-seeded+refine"] = (W_adv_sr, m)
    if verbose:
        print(f"  + refine: {fmt_metrics(m)}")

    # --- 5. Unsupervised self-learning ---
    best_so_far = max(results, key=lambda k: results[k][1]["score"])
    if verbose:
        print(f"\n--- 5. Self-learning from {best_so_far} ---")
    W_sl = unsupervised_self_learning(
        W_src, W_tgt, results[best_so_far][0], n_rounds=8, verbose=verbose)
    m = intrinsic_eval(W_src, W_tgt, W_sl)
    results["Self-learning"] = (W_sl, m)
    if verbose:
        print(f"  Final: {fmt_metrics(m)}")

    # --- Pick best ---
    best_name = max(results, key=lambda k: results[k][1]["score"])
    if verbose:
        print(f"\n  >> Best method: {best_name}")
    return results[best_name][0], results


# ---------------------------------------------------------------------------
# Align a pair with preprocessing sweep
# ---------------------------------------------------------------------------


def align_pair(W_src_raw, W_tgt_raw, pair_name="src-tgt", verbose=True):
    """Full alignment: whitening sweep + unsupervised methods.

    Runs fast methods (VecMap, Wasserstein) on multiple preprocessing
    variants, then runs slow methods (WGAN) only on the best preprocessing.

    Returns (best_W, best_W_src, best_W_tgt, all_results).
    """
    # ---- Phase 1: fast methods on all preprocessings ----
    preprocessings = {
        "white": (whiten_embeddings(W_src_raw), whiten_embeddings(W_tgt_raw)),
    }
    for alpha in [0.0, 0.2, 0.5]:
        preprocessings[f"pw-{alpha}"] = (
            partial_whiten(W_src_raw, alpha),
            partial_whiten(W_tgt_raw, alpha),
        )

    fast_results = {}  # pp_name -> (W_map, metrics, W_src, W_tgt)

    for pp_name, (W_src, W_tgt) in preprocessings.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"[{pair_name}] Preprocessing: {pp_name} (fast sweep)")
            print(f"{'='*60}")

        # VecMap init + refinements
        W_vm = unsupervised_init(W_src, W_tgt, n_seeds=200)
        W_vm_r = iterative_procrustes_topk(W_src, W_tgt, W_vm, n_iter=20, top_k=200)
        m = intrinsic_eval(W_src, W_tgt, W_vm_r)
        fast_results[f"{pp_name}/VecMap+topk"] = (W_vm_r, m, W_src, W_tgt)
        if verbose:
            print(f"  VecMap+topk: {fmt_metrics(m)}")

        # Wasserstein (single best epsilon)
        n_ot = min(2000, W_src.shape[0], W_tgt.shape[0])
        src_sub = W_src[np.random.choice(W_src.shape[0], n_ot, replace=False)]
        tgt_sub = W_tgt[np.random.choice(W_tgt.shape[0], n_ot, replace=False)]
        W_ws = wasserstein_procrustes(src_sub, tgt_sub, n_iter=30, epsilon=0.05)
        W_ws_r = iterative_procrustes_topk(W_src, W_tgt, W_ws, n_iter=20, top_k=200)
        m = intrinsic_eval(W_src, W_tgt, W_ws_r)
        fast_results[f"{pp_name}/Wass+topk"] = (W_ws_r, m, W_src, W_tgt)
        if verbose:
            print(f"  Wass+topk:   {fmt_metrics(m)}")

    # ---- Phase 2: pick best preprocessing, run full pipeline ----
    best_fast = max(fast_results, key=lambda k: fast_results[k][1]["score"])
    best_pp = best_fast.split("/")[0]
    W_src_best, W_tgt_best = preprocessings[best_pp]

    if verbose:
        print(f"\n{'='*60}")
        print(f"[{pair_name}] Full pipeline on best preprocessing: {best_pp}")
        print(f"{'='*60}")

    best_W, full_results = unsupervised_align(
        W_src_best, W_tgt_best, verbose=verbose)

    # ---- Combine all results ----
    all_results = {}
    for name, (W, m, _, _) in fast_results.items():
        all_results[name] = (W, m)
    for name, (W, m) in full_results.items():
        all_results[f"{best_pp}/{name}"] = (W, m)

    # ---- Cycle consistency on top-3 ----
    ranked = sorted(all_results.items(), key=lambda x: -x[1][1]["score"])
    if verbose:
        print(f"\n  Computing reverse mapping for cycle consistency...")
    W_rev = unsupervised_init(W_tgt_best, W_src_best, n_seeds=200)
    W_rev = iterative_procrustes_topk(
        W_tgt_best, W_src_best, W_rev, n_iter=15, top_k=200)

    for name, (W, m) in ranked[:5]:
        # Only compute CC for methods using best preprocessing
        if name.startswith(best_pp + "/") or "/" not in name:
            cc = cycle_consistency(W, W_rev, W_src_best)
            m["cycle"] = cc
            m["score_cc"] = m["score"] * 0.7 + cc * 0.3

    # ---- Final ranking ----
    def final_score(item):
        m = item[1][1]
        return m.get("score_cc", m["score"])

    ranked = sorted(all_results.items(), key=lambda x: -final_score(x))
    best_name = ranked[0][0]
    best_W = ranked[0][1][0]

    # Determine which preprocessing the winner used
    if "/" in best_name:
        winner_pp = best_name.split("/")[0]
    else:
        winner_pp = best_pp
    W_src_final = preprocessings[winner_pp][0]
    W_tgt_final = preprocessings[winner_pp][1]

    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY ({pair_name})")
        print(f"{'='*60}")
        print(f"  {'Method':<40} {'cos':>6} {'mnn':>6} "
              f"{'hub':>5} {'score':>7} {'CC':>6}")
        print(f"  {'-'*72}")
        for name, (W, m) in ranked[:20]:
            cc_str = f"{m['cycle']:.3f}" if "cycle" in m else "  -  "
            sc = f"{final_score((name, (W, m))):.4f}"
            marker = " *" if name == best_name else ""
            print(f"  {name:<40} {m['cos']:.4f} {m['mnn']:.4f} "
                  f"{m['hub']:5.1f} {sc:>7} {cc_str}{marker}")
        if len(ranked) > 20:
            print(f"  ... and {len(ranked)-20} more")
        print(f"\n  Best: {best_name}")

    return best_W, W_src_final, W_tgt_final, all_results


# ---------------------------------------------------------------------------
# Voynich nearest-neighbor display
# ---------------------------------------------------------------------------


def print_nn_unsupervised(W_src, W_tgt, W_map, words_src, words_tgt,
                          src_name="src", tgt_name="tgt", n_show=30):
    """Print nearest neighbors after mapping (no gold dict needed)."""
    mapped = W_src @ W_map
    cs = csls_score(mapped, W_tgt)

    # Show a spread of source words (by frequency rank)
    step = max(1, len(words_src) // n_show)
    indices = list(range(0, len(words_src), step))[:n_show]

    print(f"\n  {src_name:>15} -> {tgt_name} nearest neighbors (CSLS)")
    print(f"  {'-'*60}")
    for i in indices:
        nn_idx = np.argsort(cs[i])[::-1][:5]
        neighbors = [words_tgt[j] for j in nn_idx]
        scores = [cs[i, j] for j in nn_idx]
        nn_str = ", ".join(f"{w}({s:.2f})" for w, s in zip(neighbors, scores))
        print(f"  {words_src[i]:>15} -> {nn_str}")


# ---------------------------------------------------------------------------
# Embedding extraction for a new language
# ---------------------------------------------------------------------------


def load_or_extract_embeddings(lang, cache_key=None):
    """Load cached embeddings or extract from trained model.

    Returns (W_raw, words) or (None, None) if model not found.
    """
    if cache_key is None:
        cache_key = lang

    cache_path = os.path.join(DATA_DIR, f"word_embeddings_{cache_key}.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        W = data[f"W_{cache_key}"]
        words = list(data[f"words_{cache_key}"])
        return W, words

    # Try to extract from model
    model_path = os.path.join(DATA_DIR, f"model_{lang}.npz")
    sents_path = os.path.join(DATA_DIR, f"sentences_{lang}.txt")

    if not os.path.exists(model_path):
        return None, None
    if not os.path.exists(sents_path):
        return None, None

    print(f"  Extracting embeddings for '{lang}'...")
    tokenizer = BPETokenizer.load()
    sentences = load_sentences(lang)
    words_all = extract_words(sentences, min_freq=3)
    encoder = load_encoder(lang, vocab_size=tokenizer.vocab_size)

    W, words = extract_contextual_embeddings(
        encoder, tokenizer, sentences, words_all)

    np.savez_compressed(
        cache_path,
        **{f"W_{cache_key}": W, f"words_{cache_key}": np.array(words, dtype=object)},
    )
    print(f"  Cached {W.shape} embeddings to {cache_path}")
    return W, words


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    np.random.seed(42)

    print("=" * 60)
    print("Unsupervised cross-lingual alignment")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load EN-FI embeddings
    # ------------------------------------------------------------------
    raw_cache = os.path.join(DATA_DIR, "word_embeddings_raw.npz")
    if os.path.exists(raw_cache):
        print("\nLoading EN-FI embeddings from cache...")
        data = np.load(raw_cache, allow_pickle=True)
        W_en_raw = data["W_en"]
        W_fi_raw = data["W_fi"]
        en_words = list(data["words_en"])
        fi_words = list(data["words_fi"])
    else:
        print("\nExtracting EN-FI embeddings...")
        tokenizer = BPETokenizer.load()
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
        np.savez_compressed(
            raw_cache,
            W_en=W_en_raw, W_fi=W_fi_raw,
            words_en=np.array(en_words, dtype=object),
            words_fi=np.array(fi_words, dtype=object),
        )

    print(f"  EN: {W_en_raw.shape} ({len(en_words)} words)")
    print(f"  FI: {W_fi_raw.shape} ({len(fi_words)} words)")

    # ------------------------------------------------------------------
    # Validate: unsupervised EN-FI alignment
    # ------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# Validation: unsupervised EN <-> FI")
    print("#" * 60)

    W_enfi, W_en, W_fi, enfi_results = align_pair(
        W_en_raw, W_fi_raw, pair_name="EN-FI")

    # Gold-dict validation (if available)
    try:
        from evaluate_alignment import GOLD_DICT
        print(f"\nGold-dict validation (best unsupervised):")
        p, n = quick_eval(W_en, W_fi, W_enfi, en_words, fi_words, GOLD_DICT)
        print(f"  P@1={p[1]*100:.1f}% P@5={p[5]*100:.1f}% "
              f"P@10={p[10]*100:.1f}% ({n} pairs)")
        print_nn_analysis(W_en, W_fi, W_enfi, en_words, fi_words, GOLD_DICT,
                          n_words=10)
    except ImportError:
        pass

    # Save EN-FI unsupervised alignment
    np.save(os.path.join(DATA_DIR, "W_unsup_en_fi.npy"), W_enfi)
    print(f"\n  Saved W_unsup_en_fi.npy")

    # ------------------------------------------------------------------
    # Voynich alignment (if embeddings available)
    # ------------------------------------------------------------------
    vo_cache = os.path.join(DATA_DIR, "word_embeddings_vo.npz")
    if os.path.exists(vo_cache):
        print("\n" + "#" * 60)
        print("# Voynich alignment")
        print("#" * 60)

        vo_data = np.load(vo_cache, allow_pickle=True)
        W_vo_raw = vo_data["W_vo"]
        vo_words = list(vo_data["words_vo"])
        print(f"  VO: {W_vo_raw.shape} ({len(vo_words)} words)")

        # EN <-> Voynich
        print("\n" + "-" * 40)
        print("EN <-> Voynich")
        print("-" * 40)
        W_en_vo, W_en_v, W_vo_v, _ = align_pair(
            W_en_raw, W_vo_raw, pair_name="EN-VO")
        np.save(os.path.join(DATA_DIR, "W_unsup_en_vo.npy"), W_en_vo)

        # Save preprocessed embeddings used by the winning method
        np.savez_compressed(
            os.path.join(DATA_DIR, "word_embeddings_en_vo.npz"),
            W_en=W_en_v, W_vo=W_vo_v,
            words_en=np.array(en_words, dtype=object),
            words_vo=np.array(vo_words, dtype=object),
        )

        # Show VO -> EN nearest neighbors (reverse mapping = transpose)
        print_nn_unsupervised(
            W_vo_v, W_en_v, W_en_vo.T, vo_words, en_words,
            src_name="Voynich", tgt_name="English")

        # FI <-> Voynich
        print("\n" + "-" * 40)
        print("FI <-> Voynich")
        print("-" * 40)
        W_fi_vo, W_fi_v, W_vo_v2, _ = align_pair(
            W_fi_raw, W_vo_raw, pair_name="FI-VO")
        np.save(os.path.join(DATA_DIR, "W_unsup_fi_vo.npy"), W_fi_vo)

        np.savez_compressed(
            os.path.join(DATA_DIR, "word_embeddings_fi_vo.npz"),
            W_fi=W_fi_v, W_vo=W_vo_v2,
            words_fi=np.array(fi_words, dtype=object),
            words_vo=np.array(vo_words, dtype=object),
        )

        print_nn_unsupervised(
            W_vo_v2, W_fi_v, W_fi_vo.T, vo_words, fi_words,
            src_name="Voynich", tgt_name="Finnish")

        # Triangulation: check EN-VO vs FI-VO consistency
        print("\n" + "-" * 40)
        print("Triangulation: EN-VO-FI consistency")
        print("-" * 40)
        # Map EN -> VO -> FI and compare with direct EN -> FI
        W_en_via_vo = W_en_vo @ W_fi_vo.T  # EN->VO then VO->FI (= FI->VO transposed)
        m_tri = intrinsic_eval(W_en_v, W_fi, W_en_via_vo)
        print(f"  EN->VO->FI (indirect): {fmt_metrics(m_tri)}")
        m_direct = intrinsic_eval(W_en, W_fi, W_enfi)
        print(f"  EN->FI (direct):       {fmt_metrics(m_direct)}")

    else:
        print(f"\nNo Voynich embeddings found at {vo_cache}")
        print("To prepare Voynich embeddings:")
        print("  1. Place transliterated text in data/text/sentences_vo.txt")
        print("     (one line per paragraph/folio, EVA transliteration)")
        print("  2. Train: python3 train_text_jepa.py  (add 'vo' language)")
        print("  3. Embeddings will be extracted automatically on next run")
        print()
        print("Expected format for word_embeddings_vo.npz:")
        print("  W_vo:      np.array shape (n_words, 128)")
        print("  words_vo:  np.array of word strings")

    print("\nDone.")


if __name__ == "__main__":
    main()
