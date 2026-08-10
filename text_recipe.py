"""
SHARED TEXT RECIPE (T7.1/T7.3/T7.4/T7.5, phases 43/45/46/47).

Phases 19, 20, 21 and 22 each carry their own copy of the same four steps:
tokenize -> frequency-filtered vocabulary -> PPMI+SVD embeddings -> a
`hold`-repeated frame stream. Category-7 needs that recipe from four
scripts, so it lives here once instead of being copy-pasted a fifth time.
Nothing here is new mechanism -- the numbers below are transcribed from the
phase scripts they came from, and `RECIPE_19` / `RECIPE_20` reproduce those
scripts' settings exactly.

Corpus provenance. `load_gutenberg8()` reads the phase-20 eight-book corpus
from /tmp/gutenberg_corpus and, unlike phase 20's bare glob, **pins the
ebook ids and verifies every file's Gutenberg `Title:` header** on every
load -- phase 27's technique, which exists because a wrong or stale ebook
id otherwise injects mismatched text silently. A mismatch aborts.

OOV policy is an explicit argument, never a default. T7.5's sandbox
precursor found that mapping out-of-vocabulary tokens to a single UNK
symbol versus deleting them gives materially different second-order
statistics (UNK was 25.3% of that stream and dominated the top chunks), so
`oov="drop"` and `oov="unk"` are both offered and the choice is printed
with the stream.
"""

import glob
import os
import re
from collections import Counter

import numpy as np

CORPUS_DIR_8 = "/tmp/gutenberg_corpus"

# phase 20's eight books, pinned by Gutenberg ebook id with the title
# substring each file's header must contain (case-insensitive).
BOOKS_8 = {
    11: ("alice", "alice's adventures in wonderland"),
    12: ("looking_glass", "through the looking-glass"),
    55: ("wizard_oz", "wonderful wizard of oz"),
    16: ("peter_pan", "peter pan"),
    2591: ("grimm", "grimms' fairy tales"),
    74: ("tom_sawyer", "adventures of tom sawyer"),
    76: ("huck_finn", "adventures of huckleberry finn"),
    1661: ("sherlock_adventures", "adventures of sherlock holmes"),
}

_GB_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                       re.IGNORECASE | re.DOTALL)
_GB_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*",
                     re.IGNORECASE | re.DOTALL)
_TITLE = re.compile(r"^Title:\s*(.+)$", re.MULTILINE)

FETCH_HINT_8 = ("  mkdir -p " + CORPUS_DIR_8 + "\n" + "\n".join(
    f"  curl -sS -o {CORPUS_DIR_8}/{bid}.txt "
    f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt" for bid in BOOKS_8))


def load_gutenberg8(corpus_dir=CORPUS_DIR_8, verbose=True):
    """The phase-20 corpus, id-pinned and title-verified. Returns the joined
    body text with Gutenberg headers/footers stripped."""
    paths = sorted(glob.glob(f"{corpus_dir}/*.txt"))
    if not paths:
        raise SystemExit("Corpus missing (public-domain, re-fetchable):\n\n"
                         + FETCH_HINT_8 + "\n")
    bodies, seen, errors = [], set(), []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        if not stem.isdigit() or int(stem) not in BOOKS_8:
            errors.append(f"{path}: not one of the pinned eight ebook ids")
            continue
        bid = int(stem)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        m = _TITLE.search(text)
        title = (m.group(1) if m else "").strip().lower()
        want = BOOKS_8[bid][1]
        if want not in title:
            errors.append(f"{path}: header title {title!r} does not contain {want!r}")
            continue
        seen.add(bid)
        s = _GB_START.search(text)
        e = _GB_END.search(text)
        bodies.append(text[s.end() if s else 0: e.start() if e else len(text)])
    missing = set(BOOKS_8) - seen
    if missing:
        errors.append("missing ebook ids: " + ", ".join(
            f"{b} ({BOOKS_8[b][0]})" for b in sorted(missing)))
    if errors:
        raise SystemExit("Corpus verification FAILED:\n  " + "\n  ".join(errors)
                         + "\n\nRe-fetch:\n" + FETCH_HINT_8 + "\n")
    if verbose:
        print(f"  corpus: {len(paths)} books, titles verified against pinned ids")
    return "\n".join(bodies)


def load_fables():
    """Phase 19's hand-written fable corpus (in-repo, no fetch)."""
    from real_text_corpus import CORPUS_TEXT
    return CORPUS_TEXT


class Recipe:
    """Tokenized corpus + PPMI/SVD embeddings + frame-stream builder.

    Attributes: vocab, word_to_idx, counts, train_seq (token ids, OOV handled
    per `oov`), embeddings (complex, unit-norm rows), N, NORM, unk (the UNK
    token id under oov="unk", else None).
    """

    def __init__(self, text, min_count, dim, window=4, oov="drop", verbose=True):
        if oov not in ("drop", "unk"):
            raise ValueError("oov must be 'drop' or 'unk'")
        raw = re.findall(r"[a-zA-Z']+", text.lower())
        self.raw_tokens = raw
        self.counts = Counter(raw)
        vocab = sorted([w for w, c in self.counts.items() if c >= min_count])
        self.oov = oov
        if oov == "unk":
            vocab = vocab + ["<unk>"]
        self.vocab = vocab
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.unk = self.word_to_idx["<unk>"] if oov == "unk" else None
        if oov == "unk":
            self.train_seq = [self.word_to_idx.get(w, self.unk) for w in raw]
        else:
            self.train_seq = [self.word_to_idx[w] for w in raw if w in self.word_to_idx]
        self.min_count, self.window, self.dim = min_count, window, dim
        self._build_embeddings(dim, window)
        if verbose:
            frac = (0.0 if self.unk is None else
                    float(np.mean(np.asarray(self.train_seq) == self.unk)))
            print(f"  raw tokens {len(raw)}  vocab {len(vocab)} (count>={min_count})"
                  f"  stream {len(self.train_seq)}  oov={oov}"
                  + (f"  UNK share {frac:.3f}" if self.unk is not None else ""))

    def _build_embeddings(self, dim, window):
        V = len(self.vocab)
        seq = np.asarray(self.train_seq)
        cooc = np.zeros((V, V))
        for off in range(1, window + 1):          # symmetric window, vectorized
            a, b = seq[:-off], seq[off:]
            np.add.at(cooc, (a, b), 1.0)
            np.add.at(cooc, (b, a), 1.0)
        total = cooc.sum()
        row, col = cooc.sum(1, keepdims=True), cooc.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((cooc * total) / (row @ col + 1e-12) + 1e-12)
        ppmi = np.maximum(pmi, 0.0)
        U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
        d = min(dim, U.shape[1])
        emb = U[:, :d] * np.sqrt(S[:d])
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        self.embeddings_real = emb
        self.embeddings = emb.astype(complex)
        self.N = d
        self.NORM = np.sqrt(d)

    def stream(self, hold, seq=None):
        """The frame stream: each token's embedding repeated `hold` times."""
        out = []
        for w in (self.train_seq if seq is None else seq):
            e = self.embeddings[w]
            out.extend([e] * hold)
        return out


# The two committed configurations, transcribed from their phase scripts.
RECIPE_19 = dict(min_count=2, dim=40, window=4)      # phase 19 fables
RECIPE_20 = dict(min_count=150, dim=50, window=4)    # phase 20 eight books

# phase 19/20 perceive settings (phase 19 fixed recruit=0.75 after measuring
# the coverage ceiling at the default; phase 20 inherited it)
PERCEIVE_19 = dict(g_in=5.0, dt=0.05, eta=0.02, recruit=0.75)
HOLD_19 = 8


if __name__ == "__main__":
    print("fables (phase 19 recipe):")
    r19 = Recipe(load_fables(), **RECIPE_19)
    print("gutenberg8 (phase 20 recipe):")
    r20 = Recipe(load_gutenberg8(), **RECIPE_20)
    print(f"  embeddings {r19.embeddings.shape} / {r20.embeddings.shape}")
