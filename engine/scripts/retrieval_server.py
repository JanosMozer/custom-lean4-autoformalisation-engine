#!/usr/bin/env python3
"""
Persistent retrieval server — loads the embedding index + BM25 once, then
serves JSON-line requests from stdin.

Protocol:
  System B request:  {"op": "retrieve", "query": "...", "k": 8}
  System B response: [{"name": "...", "signature": "...", "slogan": "..."}, ...]

  System A request:  {"op": "lookup", "idents": ["X", "Y", ...], "k": 5}
  System A response: {"X": ["Mathlib.Foo", ...], "Y": [...], ...}

  Shutdown:          {"op": "shutdown"}
"""
import json
import logging
import sys
import os

logging.basicConfig(
    stream=sys.stderr,
    format="%(asctime)s [retrieval_server] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("retrieval_server")

sys.path.insert(0, os.environ.get("LEANBENCH_ROOT", "."))

try:
    from retrieval.retriever import HybridRetriever
    retriever = HybridRetriever()
    log.info("HybridRetriever ready")
except Exception as exc:
    log.warning("HybridRetriever unavailable (%s) — running in stub mode", exc)
    retriever = None


def _stub_retrieve(query, k):
    return []

def _stub_lookup(ident, k):
    class H:
        name = f"Mathlib.Stub.{ident}"
    return [H()]


def handle(req: dict) -> object:
    op = req.get("op")
    if op == "retrieve":
        k = int(req.get("k", 8))
        query = req.get("query", "")
        hits = retriever.retrieve(query, k) if retriever else _stub_retrieve(query, k)
        return [h.__dict__ for h in hits]
    elif op == "lookup":
        idents = req.get("idents", [])
        k = int(req.get("k", 5))
        result = {}
        for ident in idents:
            hits = retriever.lookup(ident, k) if retriever else _stub_lookup(ident, k)
            result[ident] = [h.name for h in hits]
        return result
    else:
        return {"error": f"unknown op: {op}"}


def main():
    log.info("ready — waiting for requests on stdin")
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError as e:
            sys.stdout.write(json.dumps({"error": f"bad JSON: {e}"}) + "\n")
            sys.stdout.flush()
            continue

        if req.get("op") == "shutdown":
            log.info("shutdown requested")
            break

        try:
            resp = handle(req)
        except Exception as exc:
            log.exception("error handling request")
            resp = {"error": str(exc)}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

    log.info("exiting")


if __name__ == "__main__":
    main()
