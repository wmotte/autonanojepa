#!/usr/bin/env python3
"""
Data preparation for Text JEPA cross-lingual alignment experiment.
BPE tokenization, Wikipedia download, masked-span JEPA dataloaders.

Usage:
  python3 prepare_text_jepa.py           # download + preprocess + train BPE
  python3 prepare_text_jepa.py --lang en  # English only (BPE must exist)
"""

import json
import os
import random
import re
import sys
import unicodedata
import urllib.request
import urllib.parse
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_CHARS = 170_000
MAX_CTX_LEN = 64   # BPE compresses ~4x vs char-level
MAX_TGT_LEN = 24
N_VAL_PAIRS = 2000
BPE_N_MERGES = 500

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "text")

# Special token IDs (fixed)
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
MASK_ID = 3

# Valid characters for preprocessing
_VALID_CHARS = set("abcdefghijklmnopqrstuvwxyzäöåšžàâéèêëïîôùûüçæ .,;:?!'()-0123456789")

# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------


class BPETokenizer:
    """Minimal byte-pair encoding tokenizer. No external dependencies."""

    def __init__(self):
        self.merges = []          # [(str_a, str_b), ...]
        self.token_to_id = {}     # token_str -> id
        self.id_to_token = {}     # id -> token_str
        self.vocab_size = 0

    def train(self, texts, n_merges=BPE_N_MERGES):
        """Learn BPE merges from list of text strings."""
        # Build initial corpus: each word as tuple of chars + end-of-word marker
        word_freqs = defaultdict(int)
        for text in texts:
            for word in text.split():
                word = word.strip(".,;:?!'()-")
                if word:
                    word_freqs[" ".join(list(word)) + " </w>"] += 1

        # Iteratively merge most frequent pair
        self.merges = []
        for i in range(n_merges):
            # Count all adjacent pairs
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pair_freqs[(symbols[j], symbols[j + 1])] += freq

            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            self.merges.append(best_pair)

            # Merge best pair in all words
            new_word_freqs = {}
            bigram = " ".join(best_pair)
            replacement = "".join(best_pair)
            for word, freq in word_freqs.items():
                new_word = word.replace(bigram, replacement)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs

            if (i + 1) % 100 == 0:
                print(f"  BPE merge {i+1}/{n_merges}: {best_pair[0]} + {best_pair[1]} -> {''.join(best_pair)}")

        # Build vocabulary from specials + all unique tokens
        self.token_to_id = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<mask>": 3}
        # Add single characters first
        all_chars = set()
        for text in texts:
            all_chars.update(ch for ch in text if ch in _VALID_CHARS)
        for ch in sorted(all_chars):
            if ch not in self.token_to_id:
                self.token_to_id[ch] = len(self.token_to_id)
        # Add </w> marker
        if "</w>" not in self.token_to_id:
            self.token_to_id["</w>"] = len(self.token_to_id)
        # Add merged tokens
        for a, b in self.merges:
            merged = a + b
            if merged not in self.token_to_id:
                self.token_to_id[merged] = len(self.token_to_id)

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)

    def _apply_merges(self, symbols):
        """Apply learned merges to a list of symbols."""
        for a, b in self.merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == a and symbols[i + 1] == b:
                    symbols = symbols[:i] + [a + b] + symbols[i + 2:]
                else:
                    i += 1
        return symbols

    def encode_word(self, word):
        """Encode a single word to BPE token IDs."""
        symbols = list(word) + ["</w>"]
        symbols = self._apply_merges(symbols)
        return [self.token_to_id.get(s, PAD_ID) for s in symbols]

    def encode(self, text):
        """Encode text to BPE token IDs (without BOS/EOS)."""
        ids = []
        for word in text.split():
            word = word.strip()
            if word:
                ids.extend(self.encode_word(word))
        return ids

    def decode(self, ids):
        """Decode BPE token IDs to text."""
        tokens = []
        for i in ids:
            if i in (PAD_ID, BOS_ID, EOS_ID, MASK_ID):
                continue
            tok = self.id_to_token.get(i, "")
            tokens.append(tok)
        text = "".join(tokens).replace("</w>", " ").strip()
        return text

    def save(self, path):
        """Save tokenizer to JSON."""
        data = {
            "merges": self.merges,
            "token_to_id": self.token_to_id,
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path=None):
        """Load tokenizer from JSON."""
        if path is None:
            path = os.path.join(DATA_DIR, "bpe_tokenizer.json")
        tok = cls()
        with open(path) as f:
            data = json.load(f)
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        tok.vocab_size = len(tok.token_to_id)
        return tok

    def encode_sentence_with_word_boundaries(self, sentence):
        """Encode sentence, returning (ids, word_spans).

        word_spans: list of (word_str, start_pos, end_pos) where positions
        are indices into the ids list.
        """
        ids = []
        word_spans = []
        for word in sentence.split():
            word_clean = word.strip(".,;:?!'()-")
            if not word_clean:
                continue
            start = len(ids)
            word_ids = self.encode_word(word_clean)
            ids.extend(word_ids)
            end = len(ids)
            word_spans.append((word_clean, start, end))
        return ids, word_spans


# ---------------------------------------------------------------------------
# Wikipedia download (unchanged from v1)
# ---------------------------------------------------------------------------

_EN_ARTICLES = [
    "Atlantic_Ocean", "Photosynthesis", "Roman_Empire", "Jazz",
    "Quantum_mechanics", "Amazon_rainforest", "Chess", "Volcanic_eruption",
    "William_Shakespeare", "Industrial_Revolution", "DNA", "Solar_System",
    "Buddhism", "Olympic_Games", "Artificial_intelligence", "Renaissance",
    "Elephant", "Plate_tectonics", "Democracy", "Impressionism",
    "Black_hole", "Coffee", "Migration", "Electric_guitar",
    "Antarctica", "Genome", "Algebra", "Steam_engine",
    "French_Revolution", "Coral_reef", "Piano", "Computer_science",
    "Philosophy", "Magnetism", "Dinosaur", "Wheat",
    "Printing_press", "Submarine", "Human_brain", "Silk_Road",
]

_FI_ARTICLES = [
    "Itämeri", "Sauna", "Kalevala", "Jääkausi",
    "Sisu", "Helsinki", "Metsä", "Runo",
    "Suomen_historia", "Pohjoismainen_malli", "Talvisota", "Nokia_(yritys)",
    "Joulupukki", "Muumi", "Karjala", "Saimaa",
    "Suomen_luonto", "Kantele", "Tango_(tanssi)", "Poronhoito",
    "Lappi_(maakunta)", "Suomen_kieli", "Sampo", "Åland",
    "Turku", "Tampere", "Ilmastonmuutos", "Kalastus",
    "Sähkö", "Matematiikka", "Kemia", "Fysiikka",
    "Biologia", "Maantiede", "Urheilu", "Kirjallisuus",
    "Musiikki", "Arkkitehtuuri", "Lääketiede", "Tähtitiede",
]


def fetch_wikipedia_text(title, lang="en"):
    """Fetch full plain text extract of a Wikipedia article."""
    import time as _time
    base = f"https://{lang}.wikipedia.org/w/api.php"
    params = urllib.parse.urlencode({
        "action": "query", "titles": title, "prop": "extracts",
        "explaintext": "true", "exlimit": "1", "format": "json",
    })
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                f"{base}?{params}",
                headers={"User-Agent": "TextJEPA-Research/1.0 (research project)"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for page in pages.values():
                    return page.get("extract", "")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                _time.sleep(2 ** (attempt + 1))
            else:
                break
        except Exception:
            break
    return ""


def download_corpus(lang, articles, target_chars):
    """Download Wikipedia articles until we have enough text."""
    import time as _time
    print(f"Downloading {lang} Wikipedia articles...")
    all_text, total_chars = [], 0
    for title in articles:
        if total_chars >= target_chars * 1.5:
            break
        text = fetch_wikipedia_text(title, lang)
        if text:
            all_text.append(text)
            total_chars += len(text)
            print(f"  {title}: {len(text)} chars (total: {total_chars})")
        _time.sleep(0.5)
    if total_chars < target_chars:
        print(f"  Need more text ({total_chars}/{target_chars}), fetching random...")
        for _ in range(100):
            if total_chars >= target_chars * 1.5:
                break
            try:
                base = f"https://{lang}.wikipedia.org/w/api.php"
                params = urllib.parse.urlencode({
                    "action": "query", "list": "random", "rnnamespace": "0",
                    "rnlimit": "1", "format": "json",
                })
                req = urllib.request.Request(f"{base}?{params}",
                    headers={"User-Agent": "TextJEPA-Research/1.0 (research project)"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    rand_title = data["query"]["random"][0]["title"]
                _time.sleep(0.5)
                text = fetch_wikipedia_text(rand_title, lang)
                if text and len(text) > 200:
                    all_text.append(text)
                    total_chars += len(text)
                    print(f"  {rand_title}: {len(text)} chars (total: {total_chars})")
                _time.sleep(0.5)
            except Exception:
                _time.sleep(1)
    return "\n".join(all_text)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_text(raw_text):
    """Clean and normalize text, return list of sentences."""
    text = unicodedata.normalize("NFC", raw_text).lower()
    text = re.sub(r"\[[\d,\s]+\]", "", text)
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        line = line.strip()
        if len(line) < 5:
            continue
        if len(line) < 40 and "." not in line:
            continue
        clean_lines.append(line)
    text = " ".join(clean_lines)
    text = re.sub(r"\s+", " ", text)
    filtered = [ch if ch in _VALID_CHARS else " " if ch in "\n\t\r" else ""
                for ch in text]
    text = re.sub(r" +", " ", "".join(filtered)).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if 20 <= len(s.strip()) <= 300]


# ---------------------------------------------------------------------------
# Masked span JEPA pair generation (BPE version)
# ---------------------------------------------------------------------------


def make_masked_pair_bpe(sentence, tokenizer, rng, max_ctx_len=MAX_CTX_LEN, max_tgt_len=MAX_TGT_LEN):
    """Create (context_with_mask, target_span) pair using BPE tokens."""
    ids = tokenizer.encode(sentence)
    n = len(ids)
    if n < 4:
        # Too short, pad minimally
        ids = ids + [PAD_ID] * (4 - n)
        n = 4

    # Span: 15-40% of token length, at least 2 tokens
    min_span = max(2, int(n * 0.15))
    max_span = min(int(n * 0.40), max_tgt_len - 2, n - 1)
    if min_span > max_span:
        min_span = max(2, max_span)
    span_len = rng.randint(min_span, max(min_span + 1, max_span + 1))

    max_start = n - span_len
    start = rng.randint(0, max(0, max_start))
    end = start + span_len

    # Target: the span
    target_ids = [BOS_ID] + ids[start:end]
    target_ids = target_ids[:max_tgt_len]

    # Context: before + MASK + after
    context_ids = [BOS_ID] + ids[:start] + [MASK_ID] + ids[end:]
    context_ids = context_ids[:max_ctx_len]

    # Pad
    ctx_n = len(context_ids)
    ctx_mask = [1] * ctx_n + [0] * (max_ctx_len - ctx_n)
    context_ids = context_ids + [PAD_ID] * (max_ctx_len - ctx_n)

    tgt_n = len(target_ids)
    tgt_mask = [1] * tgt_n + [0] * (max_tgt_len - tgt_n)
    target_ids = target_ids + [PAD_ID] * (max_tgt_len - tgt_n)

    return (
        np.array(context_ids, dtype=np.int32),
        np.array(ctx_mask, dtype=np.int32),
        np.array(target_ids, dtype=np.int32),
        np.array(tgt_mask, dtype=np.int32),
    )


def make_text_dataloader(sentences, tokenizer, batch_size, max_ctx_len=MAX_CTX_LEN, max_tgt_len=MAX_TGT_LEN):
    """Infinite generator. 50% full sentences, 50% sub-spans."""
    rng = random.Random(42)
    while True:
        ctx_batch, ctx_mask_batch, tgt_batch, tgt_mask_batch = [], [], [], []

        for _ in range(batch_size):
            sent = rng.choice(sentences)

            # 50% chance: use a sub-clause instead of full sentence
            if rng.random() < 0.5:
                # Split on comma/semicolon, pick a random clause
                clauses = re.split(r"[,;]", sent)
                clauses = [c.strip() for c in clauses if len(c.strip()) > 15]
                if clauses:
                    sent = rng.choice(clauses)

            ctx_ids, ctx_mask, tgt_ids, tgt_mask = make_masked_pair_bpe(
                sent, tokenizer, rng, max_ctx_len, max_tgt_len
            )
            ctx_batch.append(ctx_ids)
            ctx_mask_batch.append(ctx_mask)
            tgt_batch.append(tgt_ids)
            tgt_mask_batch.append(tgt_mask)

        yield (
            np.stack(ctx_batch), np.stack(ctx_mask_batch),
            np.stack(tgt_batch), np.stack(tgt_mask_batch),
        )


# ---------------------------------------------------------------------------
# Validation cache
# ---------------------------------------------------------------------------


def generate_val_pairs(sentences, tokenizer, n_pairs=N_VAL_PAIRS):
    """Generate deterministic validation pairs."""
    rng = random.Random(12345)
    ctx_all, ctx_mask_all, tgt_all, tgt_mask_all = [], [], [], []
    for _ in range(n_pairs):
        sent = rng.choice(sentences)
        ctx_ids, ctx_mask, tgt_ids, tgt_mask = make_masked_pair_bpe(
            sent, tokenizer, rng, MAX_CTX_LEN, MAX_TGT_LEN
        )
        ctx_all.append(ctx_ids)
        ctx_mask_all.append(ctx_mask)
        tgt_all.append(tgt_ids)
        tgt_mask_all.append(tgt_mask)
    return np.stack(ctx_all), np.stack(ctx_mask_all), np.stack(tgt_all), np.stack(tgt_mask_all)


def save_val_pairs(lang, ctx, ctx_mask, tgt, tgt_mask):
    path = os.path.join(DATA_DIR, f"val_pairs_{lang}.npz")
    np.savez_compressed(path, ctx=ctx, ctx_mask=ctx_mask, tgt=tgt, tgt_mask=tgt_mask)
    print(f"Saved {len(ctx)} validation pairs to {path}")


def load_val_pairs(lang):
    path = os.path.join(DATA_DIR, f"val_pairs_{lang}.npz")
    data = np.load(path)
    return data["ctx"], data["ctx_mask"], data["tgt"], data["tgt_mask"]


# ---------------------------------------------------------------------------
# Word extraction
# ---------------------------------------------------------------------------


def extract_words(sentences, min_freq=3):
    """Extract unique words with frequency >= min_freq."""
    word_freq = {}
    for sent in sentences:
        for word in sent.split():
            word = word.strip(".,;:?!'()-")
            if len(word) >= 2:
                word_freq[word] = word_freq.get(word, 0) + 1
    return sorted([w for w, f in word_freq.items() if f >= min_freq])


def save_sentences(lang, sentences):
    path = os.path.join(DATA_DIR, f"sentences_{lang}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(sent + "\n")
    print(f"Saved {len(sentences)} sentences to {path}")


def load_sentences(lang):
    path = os.path.join(DATA_DIR, f"sentences_{lang}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def prepare_language(lang, articles):
    """Download and preprocess one language."""
    os.makedirs(DATA_DIR, exist_ok=True)

    sent_path = os.path.join(DATA_DIR, f"sentences_{lang}.txt")
    if os.path.exists(sent_path):
        print(f"\n{lang}: already prepared, loading from disk")
        sentences = load_sentences(lang)
    else:
        raw_text = download_corpus(lang, articles, TARGET_CHARS)
        sentences = preprocess_text(raw_text)
        total = 0
        trimmed = []
        for sent in sentences:
            if total + len(sent) > TARGET_CHARS:
                break
            trimmed.append(sent)
            total += len(sent)
        sentences = trimmed
        save_sentences(lang, sentences)

    total_chars = sum(len(s) for s in sentences)
    words = extract_words(sentences)
    print(f"\n{lang} corpus stats:")
    print(f"  Sentences:    {len(sentences)}")
    print(f"  Total chars:  {total_chars}")
    print(f"  Unique words: {len(extract_words(sentences, min_freq=1))}")
    print(f"  Words (f>=3): {len(words)}")
    print(f"  Avg sent len: {total_chars / len(sentences):.0f} chars")
    return sentences


if __name__ == "__main__":
    lang_arg = None
    if "--lang" in sys.argv:
        idx = sys.argv.index("--lang")
        if idx + 1 < len(sys.argv):
            lang_arg = sys.argv[idx + 1]

    # Prepare corpora
    en_sents, fi_sents = [], []
    if lang_arg is None or lang_arg == "en":
        en_sents = prepare_language("en", _EN_ARTICLES)
    if lang_arg is None or lang_arg == "fi":
        fi_sents = prepare_language("fi", _FI_ARTICLES)

    # Train shared BPE tokenizer
    bpe_path = os.path.join(DATA_DIR, "bpe_tokenizer.json")
    if not os.path.exists(bpe_path):
        print("\nTraining shared BPE tokenizer...")
        if not en_sents:
            en_sents = load_sentences("en")
        if not fi_sents:
            fi_sents = load_sentences("fi")
        tokenizer = BPETokenizer()
        tokenizer.train(en_sents + fi_sents, n_merges=BPE_N_MERGES)
        tokenizer.save(bpe_path)
        print(f"BPE vocab size: {tokenizer.vocab_size}")
        print(f"Saved to {bpe_path}")
    else:
        tokenizer = BPETokenizer.load(bpe_path)
        print(f"\nBPE tokenizer loaded: {tokenizer.vocab_size} tokens")

    # Show BPE stats
    for lang in ["en", "fi"]:
        if lang_arg is not None and lang_arg != lang:
            continue
        sents = load_sentences(lang)
        total_bpe = sum(len(tokenizer.encode(s)) for s in sents)
        total_chars = sum(len(s) for s in sents)
        print(f"\n{lang} BPE stats:")
        print(f"  Total BPE tokens: {total_bpe}")
        print(f"  Compression ratio: {total_chars / total_bpe:.1f}x")
        print(f"  Avg tokens/sent: {total_bpe / len(sents):.0f}")
        # Sample
        sample = sents[0][:80]
        enc = tokenizer.encode(sample)
        dec = tokenizer.decode(enc)
        print(f"  Sample: '{sample}'")
        print(f"  Encoded: {len(enc)} tokens")
        print(f"  Decoded: '{dec[:60]}'")

    # Regenerate val pairs with BPE
    for lang in ["en", "fi"]:
        if lang_arg is not None and lang_arg != lang:
            continue
        sents = load_sentences(lang)
        val_path = os.path.join(DATA_DIR, f"val_pairs_{lang}.npz")
        # Always regenerate for BPE
        ctx, ctx_mask, tgt, tgt_mask = generate_val_pairs(sents, tokenizer)
        save_val_pairs(lang, ctx, ctx_mask, tgt, tgt_mask)

    print("\nDone!")
