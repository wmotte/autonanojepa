#!/usr/bin/env python3
"""
Evaluate cross-lingual embedding alignment quality.
Produces metrics, nearest-neighbor analysis, and t-SNE visualization.

Usage: uv run evaluate_alignment.py
"""

import os
import numpy as np

from prepare_text_jepa import load_sentences, extract_words, DATA_DIR

# ---------------------------------------------------------------------------
# Gold translation pairs for evaluation
# ---------------------------------------------------------------------------

GOLD_DICT = {
    "water": ["vesi", "veden", "veteen"],
    "war": ["sota", "sodan", "sotaa"],
    "city": ["kaupunki", "kaupungin"],
    "year": ["vuosi", "vuoden", "vuotta"],
    "sea": ["meri", "meren"],
    "king": ["kuningas", "kuninkaan"],
    "music": ["musiikki", "musiikin"],
    "history": ["historia", "historian"],
    "north": ["pohjoinen", "pohjoisen", "pohjois"],
    "army": ["armeija", "armeijan"],
    "forest": ["metsä", "metsän", "metsää"],
    "island": ["saari", "saaren"],
    "power": ["voima", "voiman", "valta", "vallan"],
    "death": ["kuolema", "kuoleman"],
    "language": ["kieli", "kielen"],
    "school": ["koulu", "koulun"],
    "family": ["perhe", "perheen"],
    "church": ["kirkko", "kirkon"],
    "river": ["joki", "joen"],
    "mountain": ["vuori", "vuoren"],
    "people": ["ihmiset", "ihmisten"],
    "time": ["aika", "ajan"],
    "world": ["maailma", "maailman"],
    "land": ["maa", "maan"],
    "state": ["valtio", "valtion"],
    "new": ["uusi", "uuden"],
    "old": ["vanha", "vanhan"],
    "great": ["suuri", "suuren"],
    "long": ["pitkä", "pitkän"],
    "part": ["osa", "osan"],
    "day": ["päivä", "päivän"],
    "work": ["työ", "työn"],
    "place": ["paikka", "paikan"],
    "area": ["alue", "alueen"],
    "south": ["etelä", "etelän"],
    "east": ["itä", "idän"],
    "west": ["länsi", "lännen"],
}


def csls_score(W_src, W_tgt, k=10):
    """CSLS similarity matrix."""
    src_n = W_src / (np.linalg.norm(W_src, axis=1, keepdims=True) + 1e-8)
    tgt_n = W_tgt / (np.linalg.norm(W_tgt, axis=1, keepdims=True) + 1e-8)
    sims = src_n @ tgt_n.T
    k_src = min(k, sims.shape[1])
    k_tgt = min(k, sims.shape[0])
    r_src = np.mean(np.sort(sims, axis=1)[:, -k_src:], axis=1)
    r_tgt = np.mean(np.sort(sims, axis=0)[-k_tgt:, :], axis=0)
    return 2 * sims - r_src[:, None] - r_tgt[None, :]


def evaluate_translation(W_src_mapped, W_tgt, words_src, words_tgt, gold_dict, k_values=[1, 5, 10]):
    """Evaluate word translation accuracy using gold dictionary."""
    src_to_idx = {w: i for i, w in enumerate(words_src)}
    tgt_to_idx = {w: i for i, w in enumerate(words_tgt)}

    csls = csls_score(W_src_mapped, W_tgt)

    results = {k: 0 for k in k_values}
    n_eval = 0

    for src_word, tgt_translations in gold_dict.items():
        if src_word not in src_to_idx:
            continue
        # Check if any translation exists in target vocab
        valid_tgt_ids = [tgt_to_idx[t] for t in tgt_translations if t in tgt_to_idx]
        if not valid_tgt_ids:
            continue

        n_eval += 1
        src_idx = src_to_idx[src_word]
        nn_indices = np.argsort(csls[src_idx])[::-1]

        for k in k_values:
            if any(idx in valid_tgt_ids for idx in nn_indices[:k]):
                results[k] += 1

    if n_eval == 0:
        return {k: 0.0 for k in k_values}, 0

    return {k: results[k] / n_eval for k in k_values}, n_eval


def embedding_diagnostics(W_en, W_fi, words_en, words_fi):
    """Print diagnostic statistics about the embedding spaces."""
    print("\n--- Embedding Space Diagnostics ---")

    # L2 norms
    en_norms = np.linalg.norm(W_en, axis=1)
    fi_norms = np.linalg.norm(W_fi, axis=1)
    print(f"  EN norms: mean={np.mean(en_norms):.3f} std={np.std(en_norms):.3f} min={np.min(en_norms):.3f} max={np.max(en_norms):.3f}")
    print(f"  FI norms: mean={np.mean(fi_norms):.3f} std={np.std(fi_norms):.3f} min={np.min(fi_norms):.3f} max={np.max(fi_norms):.3f}")

    # Intra-language similarity
    en_n = W_en / (np.linalg.norm(W_en, axis=1, keepdims=True) + 1e-8)
    fi_n = W_fi / (np.linalg.norm(W_fi, axis=1, keepdims=True) + 1e-8)

    en_sims = en_n @ en_n.T
    fi_sims = fi_n @ fi_n.T
    np.fill_diagonal(en_sims, 0)
    np.fill_diagonal(fi_sims, 0)

    print(f"  EN intra-sim: mean={np.mean(en_sims):.4f} max={np.max(en_sims):.4f}")
    print(f"  FI intra-sim: mean={np.mean(fi_sims):.4f} max={np.max(fi_sims):.4f}")

    # Cross-language similarity (no alignment)
    cross_sims = en_n @ fi_n.T
    print(f"  Cross-sim (no align): mean={np.mean(cross_sims):.4f} max={np.max(cross_sims):.4f}")

    # Effective dimensionality (ratio of squared Frobenius norm to squared spectral norm)
    U_en, S_en, _ = np.linalg.svd(en_n, full_matrices=False)
    U_fi, S_fi, _ = np.linalg.svd(fi_n, full_matrices=False)
    eff_dim_en = (np.sum(S_en ** 2) ** 2) / np.sum(S_en ** 4)
    eff_dim_fi = (np.sum(S_fi ** 2) ** 2) / np.sum(S_fi ** 4)
    print(f"  Effective dimensionality: EN={eff_dim_en:.1f} FI={eff_dim_fi:.1f}")

    # Top singular values
    print(f"  EN top-5 singular values: {S_en[:5].round(2)}")
    print(f"  FI top-5 singular values: {S_fi[:5].round(2)}")

    # Hubness: how many words are NN of >10 other words
    en_nn = np.argmax(en_sims, axis=1)
    fi_nn = np.argmax(fi_sims, axis=1)
    from collections import Counter
    en_hub = Counter(en_nn)
    fi_hub = Counter(fi_nn)
    en_hubs = sum(1 for v in en_hub.values() if v > 10)
    fi_hubs = sum(1 for v in fi_hub.values() if v > 10)
    print(f"  Hub words (NN of >10 others): EN={en_hubs} FI={fi_hubs}")

    # Sample nearest neighbors within each language
    print("\n  EN intra-language nearest neighbors:")
    for word in ["water", "the", "music", "year", "war"]:
        if word in {w: i for i, w in enumerate(words_en)}:
            idx = {w: i for i, w in enumerate(words_en)}[word]
            nn_idx = np.argsort(en_sims[idx])[::-1][:5]
            nns = [words_en[i] for i in nn_idx]
            print(f"    {word:12s} -> {', '.join(nns)}")

    print("\n  FI intra-language nearest neighbors:")
    for word in ["meri", "sota", "vuosi", "metsä", "kaupunki"]:
        if word in {w: i for i, w in enumerate(words_fi)}:
            idx = {w: i for i, w in enumerate(words_fi)}[word]
            nn_idx = np.argsort(fi_sims[idx])[::-1][:5]
            nns = [words_fi[i] for i in nn_idx]
            print(f"    {word:12s} -> {', '.join(nns)}")


def make_tsne_plot(W_en_mapped, W_fi, words_en, words_fi, gold_dict, output_path, n_words=150):
    """Create t-SNE visualization of aligned embeddings."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Skipping t-SNE plot (matplotlib/sklearn not available)")
        return

    # Select top-N most frequent words per language
    en_subset = words_en[:n_words]
    fi_subset = words_fi[:n_words]

    en_emb = W_en_mapped[:n_words]
    fi_emb = W_fi[:n_words]

    # Combine
    combined = np.concatenate([en_emb, fi_emb], axis=0)
    labels = ["en"] * len(en_subset) + ["fi"] * len(fi_subset)

    # L2 normalize
    combined = combined / (np.linalg.norm(combined, axis=1, keepdims=True) + 1e-8)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined) // 4))
    coords = tsne.fit_transform(combined)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Plot English words
    en_coords = coords[:len(en_subset)]
    fi_coords = coords[len(en_subset):]

    ax.scatter(en_coords[:, 0], en_coords[:, 1], c="steelblue", alpha=0.6, s=20, label="English")
    ax.scatter(fi_coords[:, 0], fi_coords[:, 1], c="coral", alpha=0.6, s=20, label="Finnish")

    # Annotate some words
    en_to_idx = {w: i for i, w in enumerate(en_subset)}
    fi_to_idx = {w: i for i, w in enumerate(fi_subset)}

    annotate_en = [w for w in ["water", "war", "city", "year", "sea", "music", "history",
                                "north", "army", "forest", "power", "death", "language"]
                   if w in en_to_idx]
    for w in annotate_en:
        i = en_to_idx[w]
        ax.annotate(w, (en_coords[i, 0], en_coords[i, 1]), fontsize=7, color="navy")

    annotate_fi = [w for w in ["meri", "sota", "kaupunki", "vuosi", "metsä", "musiikki",
                                "historia", "pohjoinen", "armeija", "voima", "kuolema", "kieli"]
                   if w in fi_to_idx]
    for w in annotate_fi:
        i = fi_to_idx[w]
        ax.annotate(w, (fi_coords[i, 0], fi_coords[i, 1]), fontsize=7, color="darkred")

    # Draw lines between known translation pairs
    for en_w, tgt_words in gold_dict.items():
        if en_w not in en_to_idx:
            continue
        for fi_w in tgt_words:
            if fi_w in fi_to_idx:
                i_en = en_to_idx[en_w]
                i_fi = fi_to_idx[fi_w]
                ax.plot(
                    [en_coords[i_en, 0], fi_coords[i_fi, 0]],
                    [en_coords[i_en, 1], fi_coords[i_fi, 1]],
                    "k-", alpha=0.15, linewidth=0.5,
                )
                break

    ax.legend()
    ax.set_title("t-SNE of aligned EN-FI JEPA embeddings")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  t-SNE plot saved to {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Evaluation: Cross-lingual JEPA alignment EN <-> FI")
    print("=" * 60)

    # Load embeddings
    data = np.load(os.path.join(DATA_DIR, "word_embeddings.npz"), allow_pickle=True)
    W_en = data["W_en"]
    W_fi = data["W_fi"]
    words_en = list(data["words_en"])
    words_fi = list(data["words_fi"])

    print(f"\nEmbeddings: EN={W_en.shape}, FI={W_fi.shape}")

    # Diagnostics
    embedding_diagnostics(W_en, W_fi, words_en, words_fi)

    # Load alignment matrices
    W_proc_path = os.path.join(DATA_DIR, "W_procrustes.npy")
    W_adv_path = os.path.join(DATA_DIR, "W_adversarial.npy")

    for name, path in [("Procrustes", W_proc_path), ("Adversarial", W_adv_path)]:
        if not os.path.exists(path):
            print(f"\n{name}: alignment matrix not found, skipping")
            continue

        W = np.load(path)
        mapped_en = W_en @ W

        print(f"\n{'=' * 60}")
        print(f"Results: {name} alignment")
        print(f"{'=' * 60}")

        # Translation accuracy
        prec, n_eval = evaluate_translation(mapped_en, W_fi, words_en, words_fi, GOLD_DICT)
        print(f"\n  Word translation accuracy (evaluated on {n_eval} pairs):")
        for k, p in prec.items():
            print(f"    P@{k}: {p * 100:.1f}%")

        # Cross-language similarity after alignment
        en_n = mapped_en / (np.linalg.norm(mapped_en, axis=1, keepdims=True) + 1e-8)
        fi_n = W_fi / (np.linalg.norm(W_fi, axis=1, keepdims=True) + 1e-8)
        cross_sims = en_n @ fi_n.T
        print(f"\n  Cross-sim (aligned): mean={np.mean(cross_sims):.4f} max={np.max(cross_sims):.4f}")

        # Detailed NN analysis
        print(f"\n  Detailed nearest-neighbor analysis:")
        test_words = sorted(GOLD_DICT.keys())[:20]
        for en_w in test_words:
            if en_w not in {w: i for i, w in enumerate(words_en)}:
                continue
            en_idx = {w: i for i, w in enumerate(words_en)}[en_w]
            csls = csls_score(mapped_en[en_idx:en_idx+1], W_fi)
            nn_indices = np.argsort(csls[0])[::-1][:5]
            neighbors = [words_fi[i] for i in nn_indices]
            expected = GOLD_DICT[en_w]
            hit = any(n in expected for n in neighbors)
            marker = "V" if hit else "X"
            print(f"    [{marker}] {en_w:15s} -> {', '.join(neighbors)} (expected: {', '.join(expected)})")

        # t-SNE
        plot_path = os.path.join(DATA_DIR, f"tsne_{name.lower()}.png")
        make_tsne_plot(mapped_en, W_fi, words_en, words_fi, GOLD_DICT, plot_path)

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  Corpus: ~170k chars/language")
    print(f"  EN words: {len(words_en)}, FI words: {len(words_fi)}")
    print(f"  Training: 300s per language, ~9000 steps")
    print(f"  Results saved to: {DATA_DIR}/")


if __name__ == "__main__":
    main()
