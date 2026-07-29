"""ProofNet benchmark with an agentic compiler-feedback loop.

For each informal statement the model generates a Lean 4 theorem statement; if it
does not type-check, the compiler error is fed back and the model edits it, up to
`--max-iters` attempts. We report a broad metric panel so model state is objective:

  well-formed        - fraction that parse as a Lean statement
  compile@k          - fraction type-checking within k attempts (k=1..max-iters)
  pass@N             - final type-check rate after the loop
  mean-iters-solved  - avg attempts used by solved problems (feedback usefulness)
  faithfulness       - surface similarity of generated code to gold
  goal-similarity    - similarity of the ELABORATED goal to gold's elaborated goal
                       (semantic: two different-looking equivalent statements match)
  goal-exact-match   - normalized elaborated goals identical to gold
  error-breakdown    - category counts among still-failing problems

Usage:
  venv/bin/python proofnet_eval/eval.py --adapter <path> [--split test] [--n N]
                                        [--bs 8] [--max-iters 5]
"""
import argparse
import difflib
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_ELAN = str(Path(__file__).resolve().parent.parent / "venv" / "elan" / "bin")
os.environ["PATH"] = _ELAN + os.pathsep + os.environ.get("PATH", "")

import torch
from peft import PeftModel

from syntaxtuning._config import ModelConfig, load_model_and_tokenizer
from rlcf._config import _clean_code, _faithfulness, _is_wellformed
from rlcf.lean_server import make_pool
from retrieval.retriever import load_retriever
from retrieval.augment import augment_user, repair_context

SYSTEM = ("You are an expert mathematician and Lean 4 programmer. Translate the informal "
          "statement into a single formal Lean 4 theorem statement ending in ':= by sorry'. "
          "Output only Lean 4 code.")


def load_rows(split, n):
    rows = [json.loads(l) for l in open(f"data/proofnet/{split}.jsonl")]
    return rows[:n] if n else rows


def opens_only(header):  # keep `open ...`, drop `import` (Mathlib preloaded in REPL)
    return "\n".join(l for l in header.splitlines() if not l.strip().startswith("import"))


def strip_imports(code):
    return re.sub(r"^\s*import.*$", "", code, flags=re.M)


def norm_goal(g):
    return re.sub(r"\s+", " ", g).strip() if g else ""


def categorize(errors):
    e = " ".join(errors).lower()
    for key, cat in [("unknown identifier", "unknown_ident"), ("unknown constant", "unknown_ident"),
                     ("unexpected token", "syntax"), ("expected", "syntax"),
                     ("type mismatch", "type_mismatch"), ("function expected", "type_mismatch"),
                     ("ambiguous", "ambiguous"), ("failed to synthesize", "typeclass")]:
        if key in e:
            return cat
    return "other" if e.strip() else "empty"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/base_qwen")
    ap.add_argument("--adapter", default="models/rlcf_checkpoints/final",
                    help="LoRA adapter path, or 'none' for the raw base model.")
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--mathlib-project", default="leaneval")
    ap.add_argument("--repl-bin", default="repl_tool/.lake/build/bin/repl")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--use-retrieval", action="store_true", help="Enable A+B Mathlib retrieval.")
    ap.add_argument("--index-dir", default="retrieval/index")
    args = ap.parse_args()
    retriever = load_retriever(args.index_dir) if args.use_retrieval else None

    torch.set_num_threads(os.cpu_count() or 8)
    model, tok = load_model_and_tokenizer(ModelConfig(base_path=args.model))
    tok.padding_side = "left"
    model.config._is_quantized = True
    model.config.dtype = torch.bfloat16
    if args.adapter and args.adapter.lower() != "none":
        model.config.model_type = "qwen3_moe_unfused"
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()

    rows = load_rows(args.split, args.n)
    n = len(rows)
    opens = [opens_only(r["header"]) for r in rows]
    pool = make_pool(args.repl_bin, args.mathlib_project, min(args.bs, 12), 60)

    def check(i, code):
        return pool.run(opens[i] + "\n" + strip_imports(code))

    gen_stats = {"tokens": 0, "time": 0.0}

    def generate(msgs_list):
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs_list]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        new = out[:, enc.input_ids.shape[1]:]
        gen_stats["tokens"] += int((new != tok.pad_token_id).sum())
        gen_stats["time"] += time.time() - t0
        return [_clean_code(x) for x in tok.batch_decode(new, skip_special_tokens=True)]

    def gen_all(msgs_list):  # batch across problems
        res = []
        for b in range(0, len(msgs_list), args.bs):
            res.extend(generate(msgs_list[b:b + args.bs]))
        return res

    print(f"ProofNet[{args.split}]: {n} problems | adapter={args.adapter} | max_iters={args.max_iters}\n", flush=True)

    # Gold elaborated goals (semantic reference).
    with ThreadPoolExecutor(max_workers=min(args.bs, 12)) as ex:
        gold_goals = list(ex.map(lambda ig: pool.run(ig[0] + "\n" + strip_imports(ig[1]["formal_statement"])).get("goal"),
                                 [(opens[i], rows[i]) for i in range(n)]))

    # Initial generation.
    codes = gen_all([[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": augment_user(r["informal_statement"], retriever)}] for r in rows])
    solved_at = [None] * n
    gen_goals = [None] * n
    last_err = [[] for _ in range(n)]

    for it in range(1, args.max_iters + 1):
        idx = [i for i in range(n) if solved_at[i] is None]
        if not idx:
            break
        with ThreadPoolExecutor(max_workers=min(args.bs, 12)) as ex:
            results = list(ex.map(lambda i: check(i, codes[i]), idx))
        fails = []
        for i, res in zip(idx, results):
            if res["ok"] and _is_wellformed(codes[i]):
                solved_at[i], gen_goals[i] = it, res.get("goal")
            else:
                last_err[i] = res["errors"]
                fails.append(i)
        print(f"iter {it}: solved {sum(s is not None for s in solved_at)}/{n} "
              f"(+{len(idx) - len(fails)} this iter)", flush=True)
        if it < args.max_iters and fails:
            repairs = [[{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": augment_user(rows[i]["informal_statement"], retriever)},
                        {"role": "assistant", "content": codes[i]},
                        {"role": "user", "content": "It failed to compile:\n" +
                         "\n".join(last_err[i])[:800] + "\n" +
                         repair_context(last_err[i], retriever) +
                         "\nOutput a corrected Lean 4 theorem statement only."}] for i in fails]
            for i, c in zip(fails, gen_all(repairs)):
                codes[i] = c

    # ---- metrics ----
    def pct(x):
        return f"{x}/{n} = {100 * x / n:.1f}%"
    wf = sum(_is_wellformed(c) for c in codes)
    faith = sum(_faithfulness(c, r["formal_statement"]) for c, r in zip(codes, rows)) / n
    solved = [s for s in solved_at if s is not None]
    goal_sims, goal_exact = [], 0
    for i in range(n):
        if solved_at[i] and gen_goals[i] and gold_goals[i]:
            goal_sims.append(difflib.SequenceMatcher(None, norm_goal(gen_goals[i]), norm_goal(gold_goals[i])).ratio())
            goal_exact += norm_goal(gen_goals[i]) == norm_goal(gold_goals[i])
    errs = Counter(categorize(last_err[i]) for i in range(n) if solved_at[i] is None)

    print("\n" + "=" * 64)
    print(f"ProofNet {args.split}  (n={n}, adapter={args.adapter})")
    print(f"  well-formed:        {pct(wf)}")
    for k in range(1, args.max_iters + 1):
        print(f"  compile@{k}:          {pct(sum(1 for s in solved if s <= k))}")
    print(f"  pass@{args.max_iters} (final):     {pct(len(solved))}")
    print(f"  mean-iters-solved:  {sum(solved)/len(solved):.2f}" if solved else "  mean-iters-solved:  n/a")
    print(f"  faithfulness(code): {faith:.3f}")
    print(f"  goal-similarity:    {sum(goal_sims)/len(goal_sims):.3f}" if goal_sims else "  goal-similarity:    n/a")
    print(f"  goal-exact-match:   {pct(goal_exact)}  (semantic: same elaborated goal as gold)")
    print(f"  gold-compiles:      {pct(sum(1 for g in gold_goals if g is not None))}  (sanity: gold type-checks here)")
    print(f"  error-breakdown (unsolved): {dict(errs)}")
    tps = gen_stats["tokens"] / gen_stats["time"] if gen_stats["time"] else 0
    print(f"  throughput:         {tps:.1f} tok/s ({gen_stats['tokens']} tok / {gen_stats['time']:.1f}s gen)")

    print("\n" + "=" * 64 + "\nSAMPLES (informal / model / gold):\n")
    for i in range(min(5, n)):
        tag = f"solved@{solved_at[i]}" if solved_at[i] else "UNSOLVED"
        print(f"[{tag}] {rows[i]['informal_statement'][:150]}")
        print(f"  model: {codes[i].strip()[:280]}")
        print(f"  gold : {rows[i]['formal_statement'].strip()[:280]}\n")


if __name__ == "__main__":
    main()
