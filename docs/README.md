# leanbench - measurement doc

Goal: fine-tune a Qwen coder model to beat autoformalization SOTA on Lean 4
(ProofNet hold-out; break 27.0%). Single RTX 5090, 32 GB.

## Stages
1. Syntax tuning (SFT) - DONE. Align output to valid Lean 4 syntax.
2. RLCF - NEXT. RL with Lean compiler feedback (miniF2F, LeanDojo).
Eval: ProofNet (Lean 4 port), held out.

## Stage 1: components
- Base: Qwen3-Coder-30B-A3B (Qwen3Moe, 30B/~3B active, 128 experts, top-8).
- Method: QLoRA, whole model NF4 (double-quant), GPU-only, no CPU offload.
  Experts unfused to per-expert nn.Linear for bnb quantization; peft
  model_type masked to bypass buggy fused-MoE adapter conversion.
- LoRA: attn q/k/v/o r=64; experts gate/up/down r=8. 468M trainable (2.92%).
- Loss: completion-only (prompt masked). Optim: paged_adamw_8bit, lr 6e-5 cosine,
  dropout 0.1, grad ckpt on. bs 8 x seq 512, eff batch 16.
- Data: Herald + Lean-Workbook (720k pairs, subsampled 40k; mean 198 tok).
- VRAM: 16.7 GB load / ~30 GB train. Throughput ~357 tok/s (MoE launch-bound,
  ~27% GPU util - per-expert python loop is the ceiling).

## Stage 1: run
- Best checkpoint: exp-1-checkpoint-1650 (epoch ~0.7).
- Train loss ~0.38; eval loss 0.334 (still decreasing, diminishing returns).
  Curve: 500->0.415, 1000->0.366, 1650->0.334.

## Metrics (tests/eval_syntax.py; greedy, 512 new tokens)
Well-formed = heuristic parse check. Compile = statement type-checks against
Mathlib via `lake env lean` (sorry bodies allowed).

| set                        | n  | well-formed | compile (type-check) |
|----------------------------|----|-------------|----------------------|
| Herald/Workbook tail proxy | 50 | 100%        | 74%                  |
| miniF2F (leak-free)        | 100| 100%        | 74%                  |

Baseline to beat in Stage 2: 74% statement type-check.
The 26% failures are semantic (e.g. Finite vs Fintype, .card, notation) - the
target of compiler-feedback RL, not syntax SFT.

## Eval infra
- Lean toolchain (elan + Lean 4.32.1 + Lake) in venv/elan.
- Mathlib lake project in leaneval/ (prebuilt cache; import Mathlib verified).
- Commands:
  - proxy:   venv/bin/python tests/eval_syntax.py --mathlib-project leaneval --n 50
  - miniF2F: venv/bin/python tests/eval_syntax.py --data data/rlcf/minif2f.jsonl --mathlib-project leaneval --n 100

