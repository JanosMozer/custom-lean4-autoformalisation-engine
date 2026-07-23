import os, sys, time
os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from syntaxtuning._config import (LoRAConfig, ModelConfig, build_lora_config, load_model_and_tokenizer)
from peft import get_peft_model, prepare_model_for_kbit_training

model, tok = load_model_and_tokenizer(ModelConfig())
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                        gradient_checkpointing_kwargs={"use_reentrant": False})
mt = model.config.model_type; model.config.model_type = "x"
model = get_peft_model(model, build_lora_config(LoRAConfig()), autocast_adapter_dtype=False)
model.config.model_type = mt
model.train()

SEQ = 1024
def run(bs, iters=4):
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, 150000, (bs, SEQ), device="cuda")
    labels = ids.clone()
    # warmup
    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, labels=labels)
        out.loss.backward(); model.zero_grad(set_to_none=True)
    step()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize(); dt = (time.time() - t0) / iters
    tok_s = bs * SEQ / dt
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"bs={bs:2d}  {dt:6.2f} s/microbatch  {tok_s:8.0f} tok/s  peak {peak:5.2f} GB", flush=True)

for bs in [1, 4, 8, 16]:
    try:
        run(bs)
    except RuntimeError as e:
        print(f"bs={bs}: {repr(e)[:80]}"); torch.cuda.empty_cache(); break
print("BENCH DONE")
