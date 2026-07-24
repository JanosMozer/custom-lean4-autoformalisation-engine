"""Stage-1 syntax eval: generate Lean 4 for held-out informal statements and
score basic well-formedness. No Lean toolchain here, so this is a heuristic
parse check plus sample dumps for manual judgment.

Usage: venv/bin/python tests/eval_syntax.py [--adapter DIR] [--n 50] [--bs 8]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("WANDB_MODE", "disabled")
# Put the venv-local Lean toolchain on PATH.
_ELAN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "venv", "elan", "bin")
os.environ["PATH"] = _ELAN + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("ELAN_HOME", os.path.join(os.path.dirname(_ELAN)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import PeftModel

from syntaxtuning._config import ModelConfig, PromptConfig, load_model_and_tokenizer

DEFAULT_ADAPTER = "models/sft_checkpoints/exp-1-checkpoint-1650"


def load_holdout(n: int, data: str = None):
    """Load n eval records. With --data, read that jsonl directly (leak-free set
    like miniF2F). Otherwise take the tail of the training files as a proxy."""
    rows = []
    files = [data] if data else ["data/syntax/herald.jsonl", "data/syntax/lean_workbook.jsonl"]
    for f in files:
        with open(f) as fh:
            lines = fh.readlines()
        chunk = lines if data else lines[-(n // 2):]
        for line in chunk:
            d = json.loads(line)
            if d.get("informal_statement") and d.get("formal_statement"):
                rows.append(d)
    return rows[:n]


def is_wellformed(code: str) -> bool:
    c = code.strip()
    if not c:
        return False
    has_decl = ("theorem " in c) or ("lemma " in c) or ("example" in c)
    has_sig = ":=" in c or " : " in c
    balanced = all(c.count(a) == c.count(b) for a, b in [("(", ")"), ("[", "]"), ("{", "}")])
    return has_decl and has_sig and balanced


def lean_compiles(code: str, project: str) -> bool:
    """True compile check: run `lake env lean` on `import Mathlib` + code inside a
    Mathlib lake project. Requires --mathlib-project (building Mathlib is heavy)."""
    with tempfile.NamedTemporaryFile("w", suffix=".lean", dir=project, delete=False) as f:
        f.write("import Mathlib\n" + code + "\n")
        path = f.name
    try:
        r = subprocess.run(["lake", "env", "lean", path], cwd=project,
                           capture_output=True, text=True, timeout=120)
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--mathlib-project", default=None,
                    help="Path to a Mathlib lake project; enables true compile rate.")
    ap.add_argument("--data", default=None,
                    help="jsonl of held-out informal/formal pairs (e.g. miniF2F).")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 8)

    model, tok = load_model_and_tokenizer(ModelConfig())
    tok.padding_side = "left"  # decoder-only generation needs left padding
    # Mask model_type so peft's buggy qwen3_moe adapter conversion is skipped.
    orig = model.config.model_type
    model.config.model_type = "qwen3_moe_unfused"
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.config.model_type = orig
    model.eval()

    system = PromptConfig().system
    rows = load_holdout(args.n, args.data)
    print(f"Evaluating {len(rows)} held-out statements from adapter {args.adapter}\n")

    ok = 0
    results = []
    batches = [rows[i:i + args.bs] for i in range(0, len(rows), args.bs)]
    for bi, batch in enumerate(batches, 1):
        print(f"[gen] batch {bi}/{len(batches)}  ({(bi-1)*args.bs}/{len(rows)} done, "
              f"{ok} well-formed so far)", flush=True)
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": r["informal_statement"]}],
                tokenize=False, add_generation_prompt=True,
            ) for r in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(**enc, max_new_tokens=512, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for r, g in zip(batch, gen):
            good = is_wellformed(g)
            ok += good
            results.append((good, r["informal_statement"], g, r["formal_statement"]))

    print(f"Well-formed rate: {ok}/{len(rows)} = {100*ok/len(rows):.1f}%\n")

    if args.mathlib_project:
        comp = 0
        for i, (_, _, g, _) in enumerate(results, 1):
            comp += lean_compiles(g, args.mathlib_project)
            print(f"[compile] {i}/{len(results)}  ({comp} compiled)", flush=True)
        print(f"\nCompile rate: {comp}/{len(results)} = {100*comp/len(results):.1f}%\n")

    print("=" * 80, "\nSAMPLES (first 5):\n")
    for good, inf, g, ref in results[:5]:
        print(f"[{'OK ' if good else 'BAD'}] informal: {inf[:140]}")
        print(f"      pred: {g.strip()[:300]}")
        print(f"      ref : {ref[:200]}\n")


if __name__ == "__main__":
    main()
