import os, sys, subprocess
os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from syntaxtuning._config import (
    DataConfig, LoRAConfig, ModelConfig, PromptConfig, RunConfig, SFTCoreConfig,
    build_lora_config, build_trainer, load_model_and_tokenizer, load_syntax_dataset,
)
from transformers import TrainingArguments
from transformers import DataCollatorForSeq2Seq, Trainer
from peft import get_peft_model, prepare_model_for_kbit_training

os.makedirs("data/syntax_smoke", exist_ok=True)
for f in ["herald.jsonl", "lean_workbook.jsonl"]:
    src = f"data/syntax/{f}"
    if os.path.exists(src):
        subprocess.run(f"head -n 300 {src} > data/syntax_smoke/{f}", shell=True, check=True)

model_cfg = ModelConfig()
model, tok = load_model_and_tokenizer(model_cfg)
print("VRAM after load (GB):", round(torch.cuda.memory_allocated()/1e9, 2))

data_cfg = DataConfig(syntax_dir="data/syntax_smoke", num_workers=8, max_seq_length=1024)
ds = load_syntax_dataset(data_cfg, PromptConfig(), tok)
print("train/val:", len(ds["train"]), len(ds["validation"]))

lora = build_lora_config(LoRAConfig())
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                        gradient_checkpointing_kwargs={"use_reentrant": False})
_orig_mt = model.config.model_type
model.config.model_type = "qwen3_moe_unfused"
try:
    model = get_peft_model(model, lora, autocast_adapter_dtype=False)
finally:
    model.config.model_type = _orig_mt
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable {trainable:,} / {total:,} = {100*trainable/total:.2f}%")

args = TrainingArguments(
    output_dir="/tmp/smoke_ckpt", per_device_train_batch_size=1, gradient_accumulation_steps=2,
    max_steps=3, logging_steps=1, eval_strategy="no", save_strategy="no", report_to="none",
    bf16=True, optim="paged_adamw_8bit", gradient_checkpointing=False, learning_rate=2e-4,
    dataloader_num_workers=2, remove_unused_columns=False,
)
collator = DataCollatorForSeq2Seq(tok, padding="longest", label_pad_token_id=-100)
trainer = Trainer(model=model, args=args, train_dataset=ds["train"], data_collator=collator, processing_class=tok)
trainer.train()
print("PEAK VRAM (GB):", round(torch.cuda.max_memory_allocated()/1e9, 2))
# Verify at least one LoRA param received a gradient
got = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
print("LoRA params with grad:", len(got), "example:", got[0] if got else None)
print("SMOKE OK")
