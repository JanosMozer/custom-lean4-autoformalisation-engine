# System Architecture (pseudocode)

## Model / base
```
base = Qwen3-Coder-30B-A3B (MoE, 128 experts top-8)
load:
    load_on_cpu(bf16)
    experts: fused 3D params -> per-expert nn.Linear        # so bnb can quantize
    quantize_all_linears -> NF4 (double-quant)              # ~16.7 GB
    move layer-by-layer -> GPU                              # GPU-only, no offload
    attn = flash_attention_2 ; config._is_quantized = True  # bf16 cast in generate
adapter = DoRA (attn r64 a128 ; experts r8 a16)            # QDoRA on NF4 base
mask config.model_type during peft load/inject             # bypass buggy MoE conversion
```

## Stage 1 - Syntax tuning (SFT)
```
data = Herald + Lean-Workbook (informal, formal)
for ex: text = chat(system,user=informal,assistant=formal)
        input_ids, labels ; labels[:prompt] = -100          # completion-only
train: HF Trainer, DoRA, paged_adamw_8bit, grad-checkpoint on
       bs8 x seq512, cosine, ~1 epoch, best-by-eval_loss
-> sft_adapter (checkpoint-1650)
```

## Stage 2 - RLCF (GRPO + compiler)
```
policy = base + sft_adapter (trainable, inherits DoRA)
ref    = policy with adapter disabled                        # KL anchor
data   = prompt-only informal (lean-workbook=breadth, then miniF2F=hard)

loop step:
    for prompt: sample G completions (temperature)
    reward(completion, gold):
        code = strip_fences(completion)
        if not well_formed(code): r = 0.0
        elif not type_checks(code): r = 0.1                  # REPL pool
        else: r = 0.3 + 0.7 * faithfulness(code, gold)       # surface sim (lexical)
    advantage = within-group(reward)                         # GRPO
    update policy ; KL penalty (beta) vs ref
```

## Lean verification (shared)
```
LeanREPLPool(size):
    each worker = `lake env repl` over pty (pexpect)         # forces flush
                  import Mathlib ONCE -> env0                 # ~ms per check after
    check(code)  -> ok:bool
    run(code)    -> {ok, errors[], goal}                     # goal = elaborated Prop
thread-safe queue -> parallel checks across the batch
```

## Retrieval - A + B (this folder)
```
build_index (once, CPU):
    decls = TheoremGraph slogans (name, signature, slogan)
    emb   = embed(slogans) L2-normalized                     # dense
    bm25  = {idf, avgdl}                                     # sparse stats
    -> emb.npy, meta.jsonl, bm25.json

HybridRetriever:
    retrieve(query, k):                                      # B
        cand = dense_topN(query, N=100)                      # semantic
        return bm25_rerank(cand, query)[:k]                  # exact Lean names
    lookup(identifier, k):                                   # A
        return name_match(identifier) or fuzzy(identifier)

augment_user(informal)   = premises(retrieve(informal)) + informal      # B
repair_context(errors)   = for each unknown-ident in errors:
                               "did you mean " + lookup(ident)           # A
```

## Evaluation - agentic compiler-feedback loop
```
for problem (informal, gold, header):
    code = generate(augment_user(informal))                  # B
    for iter in 1..max_iters:
        res = repl.run(header_opens + code)
        if res.ok and well_formed(code): solved@iter ; break
        code = generate(chat(informal, prev=code,
                             feedback=res.errors + repair_context(res.errors)))  # A
metrics:
    well-formed, compile@1..k, pass@N, mean-iters-solved,
    faithfulness(surface), goal-similarity + goal-exact(semantic: elaborated goal),
    throughput(tok/s), error-breakdown, 5 samples (informal/model/gold)
```

## Pipeline
```
SFT(DoRA) -> RLCF(GRPO, workbook -> miniF2F) -> eval on ProofNet (final hold-out)
retrieval (A+B) plugs into generation in both RLCF prompts and eval; never trains.
```
