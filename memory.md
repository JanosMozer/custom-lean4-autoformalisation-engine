# leanbench - project memory

Goal: fine-tune a Qwen coder model to beat autoformalization SOTA on Lean 4
(ProofNet hold-out, break 27.0%). Single RTX 5090, 32 GB. Multi-stage:
1. Syntax tuning (SFT) - this stage. Align latent space to Lean 4 syntax.
2. RLCF - RL with Lean compiler feedback (miniF2F, LeanDojo).
Eval: ProofNet (Lean 4 port), held out.

## Model
Qwen3-Coder-30B-A3B (Qwen3Moe): 30B total / ~3B active, 48 layers,
128 experts, top-8, hidden 2048. bf16 = ~60 GB -> does not fit in bf16.
Local path: models/base_qwen.

## Stage 1 approach: QLoRA, GPU-only
Whole model quantized to NF4 (double-quant) and resident entirely on GPU.
No CPU offload. ~16.7 GB after load, ~22 GB peak in training (bs=2).

Key mechanics (syntaxtuning/_config.py):
- MoE experts are stored upstream as fused 3D params (gate_up_proj/down_proj),
  which bitsandbytes cannot quantize. We rebuild each layer's experts as
  per-expert nn.Linear (gate_proj/up_proj/down_proj), copying pretrained
  weights across, then quantize in place with a weight-preserving 4-bit
  conversion (transformers' replace_with_bnb_linear builds meta tensors and is
  wrong for post-hoc quant), then move to GPU layer-by-layer.
- peft (transformers 5) auto-rewrites qwen3_moe expert targets into fused
  target_parameters, which misses our unfused Linears. We mask
  model.config.model_type during get_peft_model to disable that conversion.
- prepare_model_for_kbit_training enables gradient checkpointing
  (use_reentrant=False) and input grads; FA2 wired via config._attn_implementation.

LoRA: attention q/k/v/o at r=64 a=128; experts gate/up/down at r=8 a=16 (via
rank_pattern/alpha_pattern). ~468M trainable (2.92%).
Loss: completion-only (system+user prompt masked to -100; loss on Lean output).
Optim: paged_adamw_8bit, cosine, lr 2e-4, warmup 100, grad clip 1.0.

## Data
data/syntax/{herald.jsonl, lean_workbook.jsonl}, ~720k pairs
(informal_statement -> formal_statement). Mean 198 tokens, p99 470, max 857
-> max_seq_length 1024 truncates 0%.

## Throughput (measured)
MoE per-expert loop is launch-bound: fixed cost per microbatch (~10 s), all 128
experts fire regardless of tokens. Batch amortizes it (bs 1->4 = 3.5x tok/s).
- bs=4, dense seq-1024: ~357 tok/s, 30 GB peak (too tight).
- bs=2 chosen: ~26 GB peak. bs=8 OOMs.
Short unpacked sequences (~200 tok) waste the fixed overhead badly; packing to
1024 is what unlocks the 357 tok/s ceiling. This is ~20-40x slower than a dense
7-14B would run; the MoE-on-one-card is the tax.

## Decisions for Stage 1
- Do NOT train all 720k x 3 epochs (overfits phrasing, hurts generalization).
  Syntax alignment is shallow: subsample (config max_train_samples=80000),
  1 epoch, early-stop at eval_loss elbow. eval_steps=50 to watch the curve.
- eval_loss is a weak proxy; real signal is Lean compile rate on samples.
- QLoRA nails syntax/format, not semantic faithfulness (that is Stage 2).

## Files
- syntaxtuning/main.py     - entrypoint (seed, wandb, load, train, save adapter)
- syntaxtuning/_config.py  - dataclasses, data pipeline, model loader, LoRA, Trainer
- syntaxtuning/config.yaml - all hyperparameters
- syntaxtuning/main.sh     - venv/bin/python main.py <run_name>
- scratch_smoke.py         - end-to-end validation (tiny data, 3 steps)
- scratch_bench.py         - throughput/VRAM benchmark vs batch size
Stack (pinned in requirements.txt): torch 2.13, transformers 5.14.1, trl 1.8.0,
peft 0.19.1, bitsandbytes 0.49.2, flash-attn 2.8.4 (works on Blackwell sm_120).

## Open item
Packing vs completion-only tradeoff on the 30B MoE unresolved. Options: padding-
free packing WITH masking (best), full-seq packing (fast, trains on prompt),
subsample (current), or switch to dense Qwen2.5-Coder-7B/14B (10-40x faster,
standard SOTA base). Current config: completion-only + subsample, no packing.
