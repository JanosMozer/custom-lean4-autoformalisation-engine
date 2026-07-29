"""Prompt augmentation using the hybrid retriever.

B: augment_user  - prepend retrieved Mathlib premises to the initial prompt.
A: repair_context - on a compiler error, suggest candidates for missing identifiers.

Both no-op cleanly when retriever is None (index not built), so callers stay simple.
"""
import re
from typing import List, Optional

_UNKNOWN = re.compile(r"unknown (?:identifier|constant) '([^']+)'")


def format_premises(decls: List[dict]) -> str:
    if not decls:
        return ""
    lines = ["-- Relevant Mathlib declarations:"]
    lines += [f"--   {d['name']}" + (f" : {d['signature']}" if d.get("signature") else "") for d in decls]
    return "\n".join(lines)


def augment_user(informal: str, retriever, k: int = 8) -> str:
    """B: ground the initial generation in retrieved premises."""
    if retriever is None:
        return informal
    prem = format_premises(retriever.retrieve(informal, k=k))
    return f"{prem}\n\n{informal}" if prem else informal


def extract_unknown(errors: List[str]) -> List[str]:
    return list({m for e in errors for m in _UNKNOWN.findall(e)})


def repair_context(errors: List[str], retriever, k: int = 5) -> str:
    """A: for each unknown identifier in the compiler output, offer real candidates."""
    if retriever is None:
        return ""
    out = []
    for nm in extract_unknown(errors):
        cands = retriever.lookup(nm, k=k)
        if cands:
            out.append(f"-- '{nm}' is unknown. Did you mean: " + ", ".join(c["name"] for c in cands))
    return "\n".join(out)
