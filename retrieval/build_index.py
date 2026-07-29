"""Build the hybrid Mathlib retrieval index (run once, CPU by default).

Dense: slogan embeddings (natural-language descriptions of Mathlib declarations,
e.g. TheoremGraph) — better semantic match than raw symbolic Lean.
Sparse: BM25 corpus stats (idf + avgdl) for exact lexical reranking / lookup.

  venv/bin/python retrieval/build_index.py --repo uw-math-ai/math-graph --device cpu

Artifacts in retrieval/index/: emb.npy (L2-normalized f32), meta.jsonl, bm25.json
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

_TOK = re.compile(r"[A-Za-z0-9_']+")


def tokenize(s: str):
    return [t.lower() for t in _TOK.findall(s or "")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="uw-math-ai/math-graph")
    ap.add_argument("--split", default="train")
    ap.add_argument("--name-col", default="name")
    ap.add_argument("--sig-col", default="signature")
    ap.add_argument("--slogan-col", default="slogan")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cpu")  # cpu to avoid GPU contention
    ap.add_argument("--out", default="retrieval/index")
    args = ap.parse_args()

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    ds = load_dataset(args.repo, split=args.split)
    metas, texts = [], []
    for r in ds:
        name = r.get(args.name_col)
        if not name:
            continue
        slogan, sig = r.get(args.slogan_col) or "", r.get(args.sig_col) or ""
        metas.append({"name": name, "signature": sig, "slogan": slogan})
        texts.append(slogan or name)

    model = SentenceTransformer(args.model, device=args.device)
    emb = model.encode(texts, batch_size=256, normalize_embeddings=True,
                       show_progress_bar=True).astype(np.float32)

    # BM25 corpus stats over (slogan + name); per-doc tokens are re-derived at
    # query time from meta, so we only persist idf + avgdl.
    docs = [tokenize(m["slogan"] + " " + m["name"]) for m in metas]
    df = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    n = len(docs)
    idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
    avgdl = sum(len(d) for d in docs) / max(n, 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "emb.npy", emb)
    with open(out / "meta.jsonl", "w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    with open(out / "bm25.json", "w") as f:
        json.dump({"idf": idf, "avgdl": avgdl, "model": args.model}, f)
    print(f"Indexed {n} declarations -> {out}")


if __name__ == "__main__":
    main()
