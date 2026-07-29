"""Hybrid Mathlib retriever: dense slogan search -> BM25 lexical rerank.

Two entry points:
  retrieve(query, k)   -> premise grounding for the initial prompt (B)
  lookup(identifier, k) -> candidate names for a missing identifier (A)

Loads once and stays in memory; emb.npy is mmap'd. Raises if the index is
missing so callers can decide to run without retrieval.
"""
import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np

_TOK = re.compile(r"[A-Za-z0-9_']+")


def _tok(s: str):
    return [t.lower() for t in _TOK.findall(s or "")]


class HybridRetriever:
    def __init__(self, index_dir: str = "retrieval/index", device: str = "cpu"):
        d = Path(index_dir)
        if not (d / "emb.npy").exists():
            raise FileNotFoundError(f"retrieval index not found at {d}; run retrieval/build_index.py")
        self.emb = np.load(d / "emb.npy", mmap_mode="r")
        self.meta = [json.loads(l) for l in open(d / "meta.jsonl")]
        self.names = [m["name"] for m in self.meta]
        self._lname = [n.lower() for n in self.names]
        bm25 = json.load(open(d / "bm25.json"))
        self.idf, self.avgdl, self._model_name = bm25["idf"], bm25["avgdl"], bm25["model"]
        self._device = device
        self._model = None  # lazy: only load the embedder when first querying

    def _embed(self, query: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model.encode([query], normalize_embeddings=True).astype(np.float32)[0]

    def _bm25(self, q_tokens, doc_tokens, k1: float = 1.5, b: float = 0.75) -> float:
        if not doc_tokens:
            return 0.0
        tf = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1
        dl = len(doc_tokens)
        s = 0.0
        for t in q_tokens:
            if t in tf:
                idf = self.idf.get(t, 0.0)
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / self.avgdl))
        return s

    def retrieve(self, query: str, k: int = 8, pool: int = 100) -> List[dict]:
        """Dense top-`pool` candidates, reranked to top-`k` by BM25 (captures exact
        Lean names the dense model misses)."""
        scores = np.asarray(self.emb @ self._embed(query))
        pool = min(pool, len(scores))
        cand = np.argpartition(-scores, pool - 1)[:pool]
        qt = _tok(query)
        cand = sorted(cand, key=lambda i: self._bm25(qt, _tok(self.meta[i]["slogan"] + " " + self.names[i])),
                      reverse=True)
        return [self.meta[i] for i in cand[:k]]

    def lookup(self, identifier: str, k: int = 5) -> List[dict]:
        """Candidate declarations for a compiler-reported missing identifier (A).
        Lexical: substring match on names, then fuzzy fallback."""
        q = identifier.lower()
        hits = [i for i, n in enumerate(self._lname) if q in n or n in q]
        if len(hits) < k:
            import difflib
            hits = sorted(range(len(self.names)),
                          key=lambda i: difflib.SequenceMatcher(None, q, self._lname[i]).ratio(),
                          reverse=True)[:k]
        else:
            hits = sorted(hits, key=lambda i: len(self.names[i]))[:k]
        return [self.meta[i] for i in hits]


def load_retriever(index_dir: str = "retrieval/index", device: str = "cpu") -> Optional[HybridRetriever]:
    """Best-effort loader: returns None (retrieval disabled) if the index is absent."""
    try:
        return HybridRetriever(index_dir, device)
    except FileNotFoundError:
        return None
