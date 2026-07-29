import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.activations import ACT2FN

logger = logging.getLogger("SFTCore")


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class RunConfig:
    project: str = "mesh-autoformalizer"
    entity: str = ""
    group: str = "syntax-tuning"
    name: str = "run"
    seed: int = 42


@dataclass
class ModelConfig:
    base_path: str = "models/base_qwen"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    # QLoRA: the whole model (attention + MoE experts) is quantized to NF4 and
    # lives entirely on the GPU. There is no CPU offload.
    load_in_4bit: bool = True


@dataclass
class LoRAConfig:
    # Attention adapters (full rank).
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    r: int = 64
    lora_alpha: int = 128
    # Expert adapters. Experts hold ~96% of params, so a lower rank keeps the
    # trainable/optimizer footprint inside 32 GB while still adapting them.
    expert_target_modules: List[str] = field(default_factory=lambda: ["gate_proj", "up_proj", "down_proj"])
    expert_r: int = 8
    expert_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    use_dora: bool = True  # DoRA: weight-decomposed adaptation (magnitude + direction)


@dataclass
class DataConfig:
    syntax_dir: str = "data/syntax"
    datasets: List[str] = field(default_factory=lambda: ["herald.jsonl", "lean_workbook.jsonl"])
    max_seq_length: int = 1024
    val_split: float = 0.02
    # Subsample the combined corpus for light syntax alignment (0 / null = use all).
    # Syntax tuning is shallow; a curated subset over ~1 epoch avoids overfitting.
    max_train_samples: int = 0
    num_workers: int = 4


@dataclass
class PromptConfig:
    system: str = (
        "You are an expert mathematician and Lean 4 programmer. Your task is to "
        "translate informal mathematical statements into formal Lean 4 theorem "
        "statements. Output only valid Lean 4 code."
    )


@dataclass
class SFTCoreConfig:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)


# =============================================================================
# Config Loading
# =============================================================================


def load_config(config_path: str, run_name: Optional[str] = None) -> SFTCoreConfig:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = SFTCoreConfig(
        run=RunConfig(**raw.get("run", {})),
        model=ModelConfig(**raw.get("model", {})),
        lora=LoRAConfig(**raw.get("lora", {})),
        data=DataConfig(**raw.get("data", {})),
        prompt=PromptConfig(**raw.get("prompt", {})),
    )

    if run_name:
        cfg.run.name = run_name

    return cfg


# =============================================================================
# Dataset Loading (completion-only supervision)
# =============================================================================


def load_syntax_dataset(
    data_cfg: DataConfig,
    prompt_cfg: PromptConfig,
    tokenizer: PreTrainedTokenizer,
) -> DatasetDict:
    """Load, tokenize and mask the syntax-tuning corpus.

    Loss is computed only over the assistant (formal Lean) completion; the
    system+user prompt tokens are masked with -100.
    """
    syntax_dir = Path(data_cfg.syntax_dir)
    all_datasets: List[Dataset] = []

    for ds_file in data_cfg.datasets:
        path = syntax_dir / ds_file
        if not path.exists():
            logger.warning(f"Dataset file not found, skipping: {path}")
            continue

        logger.info(f"Loading dataset: {path}")
        ds = Dataset.from_json(str(path))

        original_len = len(ds)
        ds = ds.filter(
            lambda ex: bool(ex["informal_statement"]) and bool(ex["formal_statement"]),
            num_proc=data_cfg.num_workers,
        )
        filtered = original_len - len(ds)
        if filtered > 0:
            logger.warning(f"Filtered {filtered} malformed records from {ds_file}")

        all_datasets.append(ds)
        logger.info(f"  -> {len(ds)} valid records from {ds_file}")

    if not all_datasets:
        raise FileNotFoundError(f"No dataset files found in {syntax_dir}.")

    combined = concatenate_datasets(all_datasets)
    logger.info(f"Combined dataset size: {len(combined)} records")

    if data_cfg.max_train_samples and len(combined) > data_cfg.max_train_samples:
        combined = combined.shuffle(seed=42).select(range(data_cfg.max_train_samples))
        logger.info(f"Subsampled to {len(combined)} records for light syntax alignment")

    system = prompt_cfg.system
    max_len = data_cfg.max_seq_length

    def tokenize_batch(batch: Dict[str, List]) -> Dict[str, List]:
        input_ids_out, labels_out, attn_out = [], [], []
        for informal, formal in zip(batch["informal_statement"], batch["formal_statement"]):
            base = [{"role": "system", "content": system}, {"role": "user", "content": informal}]
            prompt_text = tokenizer.apply_chat_template(base, tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(
                base + [{"role": "assistant", "content": formal}],
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

            # Guard against tokenizer edge cases where the prompt is not a prefix.
            n_prompt = min(len(prompt_ids), len(full_ids))
            full_ids = full_ids[:max_len]
            labels = [-100] * min(n_prompt, len(full_ids)) + full_ids[min(n_prompt, len(full_ids)):]

            input_ids_out.append(full_ids)
            labels_out.append(labels)
            attn_out.append([1] * len(full_ids))
        return {"input_ids": input_ids_out, "labels": labels_out, "attention_mask": attn_out}

    tokenized = combined.map(
        tokenize_batch,
        batched=True,
        batch_size=512,
        num_proc=data_cfg.num_workers,
        remove_columns=combined.column_names,
        desc="Tokenizing (completion-only)",
    )

    # Drop examples whose completion was fully truncated away (no supervised tokens).
    tokenized = tokenized.filter(
        lambda ex: any(t != -100 for t in ex["labels"]),
        num_proc=data_cfg.num_workers,
    )

    split = tokenized.train_test_split(test_size=data_cfg.val_split, seed=42, shuffle=True)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


# =============================================================================
# Model Loading  (whole-model NF4 QLoRA, GPU-only)
# =============================================================================


class _Expert(nn.Module):
    """A single MoE expert as three plain nn.Linear layers (bnb-quantizable)."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class _Experts(nn.Module):
    """Drop-in for Qwen3MoeExperts using per-expert nn.Linear modules.

    Matches the upstream forward signature so the unmodified sparse-MoE block
    calls it transparently. Unlike the fused 3D-parameter original, each Linear
    can be replaced by bitsandbytes Linear4bit and wrapped with its own LoRA
    adapter.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.act_fn = ACT2FN[config.hidden_act]
        for j in range(self.num_experts):
            self.add_module(str(j), _Expert(config.hidden_size, config.moe_intermediate_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            # One host sync per layer (.tolist) instead of one per hit expert (.item).
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()

        for expert_idx in expert_hit:
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            expert = getattr(self, str(expert_idx))
            gate = expert.gate_proj(current_state)
            up = expert.up_proj(current_state)
            h = self.act_fn(gate) * up
            h = expert.down_proj(h)
            h = h * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, h.to(final_hidden_states.dtype))

        return final_hidden_states


def _rebuild_experts_as_linear(model: PreTrainedModel, config) -> None:
    """Replace each layer's fused MoE experts with per-expert nn.Linear modules,
    copying the pretrained weights across (fused gate_up -> gate_proj/up_proj)."""
    inter = config.moe_intermediate_size
    for layer in model.model.layers:
        fused = layer.mlp.experts
        gate_up = fused.gate_up_proj.data  # [E, 2*inter, hidden]
        down = fused.down_proj.data        # [E, hidden, inter]

        new_experts = _Experts(config).to(dtype=gate_up.dtype)
        for j in range(config.num_experts):
            expert = getattr(new_experts, str(j))
            expert.gate_proj.weight.data.copy_(gate_up[j, :inter, :])
            expert.up_proj.weight.data.copy_(gate_up[j, inter:, :])
            expert.down_proj.weight.data.copy_(down[j])

        layer.mlp.experts = new_experts
        del fused, gate_up, down
        gc.collect()


def _quantize_linears_4bit(module: nn.Module, compute_dtype: torch.dtype, skip: Tuple[str, ...], _prefix: str = "") -> None:
    """Recursively swap every nn.Linear for a bnb Linear4bit, carrying the
    real (bf16) weights across so quantization happens later on .to('cuda').

    Unlike transformers' replace_with_bnb_linear (which builds meta modules and
    defers weight materialization to a state-dict load), this copies weights in
    place, which is what post-hoc quantization of an already-loaded model needs.
    """
    import bitsandbytes as bnb

    for name, child in list(module.named_children()):
        full_name = f"{_prefix}.{name}" if _prefix else name
        if type(child) is nn.Linear and not any(s in full_name for s in skip):
            has_bias = child.bias is not None
            new = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=has_bias,
                compute_dtype=compute_dtype,
                compress_statistics=True,
                quant_type="nf4",
                device="cpu",
            )
            new.weight = bnb.nn.Params4bit(
                child.weight.data,
                requires_grad=False,
                compress_statistics=True,
                quant_type="nf4",
            )
            if has_bias:
                new.bias = nn.Parameter(child.bias.data, requires_grad=False)
            setattr(module, name, new)
        else:
            _quantize_linears_4bit(child, compute_dtype, skip, full_name)


def load_model_and_tokenizer(model_cfg: ModelConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_cfg.torch_dtype, torch.bfloat16)

    if not model_cfg.load_in_4bit:
        raise ValueError("Qwen3-Coder-30B does not fit on 32 GB in bf16; load_in_4bit must be True.")

    config = AutoConfig.from_pretrained(model_cfg.base_path, trust_remote_code=True)
    config._attn_implementation = model_cfg.attn_implementation

    logger.info(f"Loading base model on CPU from: {model_cfg.base_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.base_path,
        config=config,
        dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    logger.info("Rebuilding MoE experts as per-expert linear layers...")
    _rebuild_experts_as_linear(model, config)
    gc.collect()

    logger.info("Quantizing all linear layers to NF4 (weight-preserving, in place)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
    )
    _quantize_linears_4bit(model, compute_dtype=torch_dtype, skip=("lm_head",))
    model.is_loaded_in_4bit = True
    model.config.quantization_config = bnb_config

    logger.info("Moving model to GPU layer-by-layer (quantization happens on .cuda())...")
    model.model.embed_tokens = model.model.embed_tokens.to("cuda:0")
    model.model.norm = model.model.norm.to("cuda:0")
    model.lm_head = model.lm_head.to("cuda:0")
    n_layers = len(model.model.layers)
    for i in range(n_layers):
        model.model.layers[i] = model.model.layers[i].to("cuda:0")
        gc.collect()
        torch.cuda.empty_cache()
        if (i + 1) % 8 == 0:
            logger.info(f"  Moved {i + 1}/{n_layers} layers to GPU")

    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"Base model resident on GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.base_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    logger.info("Model and tokenizer loaded successfully (GPU-only).")
    return model, tokenizer


# =============================================================================
# LoRA Config
# =============================================================================


def build_lora_config(lora_cfg: LoRAConfig) -> LoraConfig:
    all_targets = list(lora_cfg.target_modules) + list(lora_cfg.expert_target_modules)
    # Give expert projections a lower rank via pattern matching on their names.
    rank_pattern = {m: lora_cfg.expert_r for m in lora_cfg.expert_target_modules}
    alpha_pattern = {m: lora_cfg.expert_alpha for m in lora_cfg.expert_target_modules}
    return LoraConfig(
        task_type=TaskType[lora_cfg.task_type],
        target_modules=all_targets,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        use_dora=lora_cfg.use_dora,  # QDoRA on the 4-bit base
        inference_mode=False,
    )


# =============================================================================
# Training Arguments
# =============================================================================


def build_training_args(cfg: SFTCoreConfig, config_path: str) -> TrainingArguments:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f).get("training", {})

    os.environ["WANDB_PROJECT"] = cfg.run.project
    os.environ["WANDB_ENTITY"] = cfg.run.entity
    os.environ["WANDB_RUN_GROUP"] = cfg.run.group

    return TrainingArguments(
        output_dir=raw.get("output_dir", "models/sft_checkpoints"),
        run_name=cfg.run.name,
        num_train_epochs=raw.get("num_train_epochs", 3),
        per_device_train_batch_size=raw.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=raw.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=raw.get("gradient_accumulation_steps", 8),
        # Gradient checkpointing is enabled on the model via
        # prepare_model_for_kbit_training; keep it off here to avoid double-wrapping.
        gradient_checkpointing=False,
        learning_rate=raw.get("learning_rate", 2e-4),
        lr_scheduler_type=raw.get("lr_scheduler_type", "cosine"),
        warmup_steps=raw.get("warmup_steps", 100),
        weight_decay=raw.get("weight_decay", 0.01),
        max_grad_norm=raw.get("max_grad_norm", 1.0),
        optim=raw.get("optim", "paged_adamw_8bit"),
        bf16=raw.get("bf16", True),
        fp16=raw.get("fp16", False),
        eval_strategy=raw.get("eval_strategy", "steps"),
        eval_steps=raw.get("eval_steps", 200),
        save_strategy=raw.get("save_strategy", "steps"),
        save_steps=raw.get("save_steps", 200),
        save_total_limit=raw.get("save_total_limit", 3),
        load_best_model_at_end=raw.get("load_best_model_at_end", False),
        metric_for_best_model=raw.get("metric_for_best_model", "eval_loss"),
        greater_is_better=raw.get("greater_is_better", False),
        logging_steps=raw.get("logging_steps", 10),
        report_to=raw.get("report_to", "wandb"),
        max_steps=raw.get("max_steps", -1),
        seed=cfg.run.seed,
        data_seed=cfg.run.seed,
        dataloader_num_workers=min(cfg.data.num_workers, 8),
        dataloader_pin_memory=True,
    )


# =============================================================================
# Trainer Assembly
# =============================================================================


def build_trainer(
    cfg: SFTCoreConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: DatasetDict,
    training_args: TrainingArguments,
    lora_config: LoraConfig,
) -> Trainer:
    # Checkpointing is mandatory: 48-layer activations OOM without it at any
    # useful batch. Throughput comes from a large batch amortizing the fixed
    # per-expert launch overhead, not from disabling recompute.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # peft (transformers>=5) auto-rewrites qwen3_moe expert target_modules
    # (gate_proj/up_proj/down_proj) into *fused* target_parameters
    # (gate_up_proj/down_proj). Our experts are deliberately unfused into
    # per-expert nn.Linear so bitsandbytes can quantize them, so we mask the
    # model_type during injection to disable that conversion and let the
    # unfused Linears match as ordinary target_modules.
    base_config = model.config
    original_model_type = base_config.model_type
    base_config.model_type = "qwen3_moe_unfused"
    try:
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    finally:
        base_config.model_type = original_model_type

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"{'DoRA' if lora_config.use_dora else 'LoRA'} applied: {trainable_params:,} trainable / "
        f"{total_params:,} total ({100 * trainable_params / total_params:.2f}%)"
    )

    collator = DataCollatorForSeq2Seq(tokenizer, padding="longest", label_pad_token_id=-100)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        processing_class=tokenizer,
    )
