"""Small deterministic text utilities used for deduplication, novelty and relevance."""

from __future__ import annotations

import hashlib
import re

_TOKEN = re.compile(r"[a-z0-9']+")
STOPWORDS = frozenset(
    "a an the and or but if then of to in on at by for with from as is are was were be been "
    "being it its this that these those there here which who whom what when where why how "
    "not no nor so than too very can could should would may might must will shall do does did "
    "have has had i you he she we they me him her us them my your our their about into over "
    "under again further once also just only own same such some any each few more most other "
    "both all one".split()
)
NEGATIONS = frozenset({"not", "no", "never", "none", "cannot", "can't", "isn't", "aren't",
                       "wasn't", "weren't", "doesn't", "don't", "didn't", "won't", "nor",
                       "false", "untrue"})


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_words(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in STOPWORDS and t not in NEGATIONS and len(t) > 2}


def normalize(text: str) -> str:
    return " ".join(tokens(text))


def norm_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode()).hexdigest()


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def ngrams(toks: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def distinct_n(texts: list[str], n: int = 2) -> float:
    grams = [g for t in texts for g in ngrams(tokens(t), n)]
    return len(set(grams)) / len(grams) if grams else 0.0


def internal_repetition(text: str, n: int = 3) -> float:
    """Fraction of repeated n-grams inside one text (degenerate loops score high)."""
    grams = ngrams(tokens(text), n)
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def negated(text: str) -> bool:
    return any(t in NEGATIONS for t in tokens(text))


def fts_query(text: str, limit_terms: int = 12) -> str:
    words = [w for w in dict.fromkeys(tokens(text)) if w not in STOPWORDS and len(w) > 2]
    words = [w.replace("'", "") for w in words][:limit_terms]
    return " OR ".join(f'"{w}"' for w in words if w)
