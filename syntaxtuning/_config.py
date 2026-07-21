"""Syntax Tuning Core: Dataset loading, model setup, LoRA, and Trainer.

Stage 1 SFT — teaches Qwen3-Coder-30B to translate informal mathematical
statements into formal Lean 4 theorem statements using Herald + Lean Workbook.
"""

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger("SFTCore")


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class RunConfig:
    """WandB run identity configuration."""
    project: str = "mesh-autoformalizer"
    entity: str = ""
    group: str = "syntax-tuning"
    name: str = "run"  # Overridden by CLI argument
    seed: int = 42


@dataclass
class ModelConfig:
    """Model loading configuration."""
    base_path: str = "models/base_qwen"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    load_in_4bit: bool = False
    load_in_8bit: bool = False


@dataclass
class LoRAConfig:
    """PEFT LoRA adapter configuration.

    High rank (r=64) chosen to ensure sufficient capacity for adapting
    the model's latent space to Martin-Löf type theory structure.
    """
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DataConfig:
    """Dataset loading and tokenization configuration."""
    syntax_dir: str = "data/syntax"
    datasets: List[str] = field(default_factory=lambda: [
        "herald.jsonl", "lean_workbook.jsonl"
    ])
    max_seq_length: int = 1024
    val_split: float = 0.02
    num_workers: int = 4


@dataclass
class PromptConfig:
    """Prompt template configuration for autoformalization task."""
    system: str = (
        "You are an expert mathematician and Lean 4 programmer. Your task is to "
        "translate informal mathematical statements into formal Lean 4 theorem "
        "statements. Output only valid Lean 4 code."
    )


@dataclass
class SFTCoreConfig:
    """Aggregated configuration container."""
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)


# =============================================================================
# Config Loading
# =============================================================================


def load_config(config_path: str, run_name: Optional[str] = None) -> SFTCoreConfig:
    """Load and parse the YAML configuration into typed dataclasses.

    Args:
        config_path: Path to the YAML config file.
        run_name: Optional run name override (e.g., from CLI argument).

    Returns:
        A fully populated SFTCoreConfig instance.
    """
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
# Dataset Loading
# =============================================================================


def format_prompt(
    informal: str,
    formal: str,
    system_prompt: str,
    tokenizer: PreTrainedTokenizer,
) -> str:
    """Format a (informal, formal) pair into a chat-templated string.

    Uses the model's built-in chat template so that special tokens
    (e.g., <|im_start|>) are handled correctly for Qwen3.

    Args:
        informal: Natural language math statement (the prompt).
        formal: Lean 4 theorem statement (the completion to learn).
        system_prompt: System instruction defining the task.
        tokenizer: Tokenizer with a chat template.

    Returns:
        Fully formatted string ready for tokenization.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": informal},
        {"role": "assistant", "content": formal},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def load_syntax_dataset(
    data_cfg: DataConfig,
    prompt_cfg: PromptConfig,
    tokenizer: PreTrainedTokenizer,
) -> DatasetDict:
    """Load, merge, and format Herald + Lean Workbook datasets.

    The two datasets share the same JSONL schema:
      {problem_id, informal_statement, formal_statement, source}

    Skips any records where informal or formal statement is missing/empty,
    which could arise from malformed entries.

    Args:
        data_cfg: Data configuration (paths, split ratio).
        prompt_cfg: Prompt configuration (system message).
        tokenizer: Tokenizer for chat template formatting.

    Returns:
        A DatasetDict with 'train' and 'validation' splits.
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

        # Filter out malformed records
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
        raise FileNotFoundError(
            f"No dataset files found in {syntax_dir}. "
            "Please run src/data_loader.py first."
        )

    combined = concatenate_datasets(all_datasets)
    logger.info(f"Combined dataset size: {len(combined)} records")

    # Format each record into the chat-templated text
    def apply_template(batch: Dict[str, List]) -> Dict[str, List]:
        return {
            "text": [
                format_prompt(inf, form, prompt_cfg.system, tokenizer)
                for inf, form in zip(
                    batch["informal_statement"], batch["formal_statement"]
                )
            ]
        }

    combined = combined.map(
        apply_template,
        batched=True,
        batch_size=512,
        num_proc=data_cfg.num_workers,
        remove_columns=combined.column_names,
        desc="Formatting prompts",
    )

    # Deterministic train/val split for reproducibility
    split = combined.train_test_split(
        test_size=data_cfg.val_split, seed=42, shuffle=True
    )
    return DatasetDict({"train": split["train"], "validation": split["test"]})


# =============================================================================
# Model Loading
# =============================================================================


def load_model_and_tokenizer(
    model_cfg: ModelConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load the base model and tokenizer with the correct dtype and quantization.

    For the 30B MoE model on a single 32GB GPU:
    - We use bf16 WITHOUT quantization to preserve LoRA gradient quality.
    - Flash Attention 2 is enabled for memory efficiency on long sequences.

    Args:
        model_cfg: Model configuration.

    Returns:
        Tuple of (model, tokenizer).
    """
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_cfg.torch_dtype, torch.bfloat16)

    bnb_config = None
    if model_cfg.load_in_4bit:
        logger.info("Using 4-bit quantization (NF4)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    elif model_cfg.load_in_8bit:
        logger.info("Using 8-bit quantization")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # When using 4-bit with fp32 CPU offload, we provide a custom device_map
    # that keeps all transformer layers on GPU and offloads only embeddings to CPU.
    # This avoids the bnb meta-tensor bug while staying within 32 GB VRAM.
    if bnb_config is not None:
        from accelerate import infer_auto_device_map, init_empty_weights
        with init_empty_weights():
            empty_model = AutoModelForCausalLM.from_config(
                __import__("transformers").AutoConfig.from_pretrained(
                    model_cfg.base_path, trust_remote_code=True
                )
            )
        device_map = infer_auto_device_map(
            empty_model,
            max_memory={0: "28GiB", "cpu": "48GiB"},
            no_split_module_classes=["Qwen3MoeDecoderLayer"],
        )
        del empty_model
    else:
        device_map = "auto"

    logger.info(f"Loading model from: {model_cfg.base_path} (device_map={device_map if isinstance(device_map, str) else 'custom'})")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.base_path,
        dtype=torch_dtype,
        quantization_config=bnb_config,
        attn_implementation=model_cfg.attn_implementation,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.base_path,
        trust_remote_code=True,
        padding_side="right",  # Right-pad for decoder-only causal LM
    )

    # Qwen3 uses a specific EOS token; ensure pad token is correctly set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer


# =============================================================================
# LoRA Config
# =============================================================================


def build_lora_config(lora_cfg: LoRAConfig) -> LoraConfig:
    """Construct the PEFT LoraConfig from the hyperparameter dataclass.

    The config is passed directly to SFTTrainer via its peft_config argument.
    TRL handles prepare_model_for_kbit_training internally, which correctly
    integrates with the installed bitsandbytes and transformers versions.

    Args:
        lora_cfg: LoRA hyperparameter configuration.

    Returns:
        A LoraConfig ready to pass to SFTTrainer.
    """
    return LoraConfig(
        task_type=TaskType[lora_cfg.task_type],
        target_modules=lora_cfg.target_modules,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        inference_mode=False,
    )


# =============================================================================
# Training Arguments
# =============================================================================


def build_training_args(cfg: SFTCoreConfig) -> SFTConfig:
    """Construct the TRL SFTConfig from the top-level config.

    SFTConfig extends TrainingArguments with sequence-packing and
    response-only masking, which are critical here:
    - Packing: Maximizes GPU utilization by filling the context window.
    - Response-only masking: Loss is computed only on formal_statement tokens,
      preventing the model from "learning" the question (teacher forcing quality).

    Args:
        cfg: The full SFTCoreConfig.

    Returns:
        A populated SFTConfig ready to pass to SFTTrainer.
    """
    raw = yaml.safe_load(
        open("syntaxtuning/config.yaml")
    ).get("training", {})

    # WandB run name comes from the CLI arg passed at runtime
    os.environ["WANDB_PROJECT"] = cfg.run.project
    os.environ["WANDB_ENTITY"] = cfg.run.entity
    os.environ["WANDB_RUN_GROUP"] = cfg.run.group

    return SFTConfig(
        output_dir=raw.get("output_dir", "models/sft_checkpoints"),
        run_name=cfg.run.name,
        num_train_epochs=raw.get("num_train_epochs", 3),
        per_device_train_batch_size=raw.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=raw.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=raw.get("gradient_accumulation_steps", 16),
        learning_rate=raw.get("learning_rate", 2e-4),
        lr_scheduler_type=raw.get("lr_scheduler_type", "cosine"),
        warmup_ratio=raw.get("warmup_ratio", 0.05),
        weight_decay=raw.get("weight_decay", 0.01),
        max_grad_norm=raw.get("max_grad_norm", 1.0),
        bf16=raw.get("bf16", True),
        fp16=raw.get("fp16", False),
        eval_strategy=raw.get("eval_strategy", "steps"),
        eval_steps=raw.get("eval_steps", 500),
        save_strategy=raw.get("save_strategy", "steps"),
        save_steps=raw.get("save_steps", 500),
        save_total_limit=raw.get("save_total_limit", 3),
        load_best_model_at_end=raw.get("load_best_model_at_end", True),
        metric_for_best_model=raw.get("metric_for_best_model", "eval_loss"),
        greater_is_better=raw.get("greater_is_better", False),
        logging_steps=raw.get("logging_steps", 25),
        report_to=raw.get("report_to", "wandb"),
        max_length=cfg.data.max_seq_length,
        packing=raw.get("packing", True),
        dataset_text_field="text",
        seed=cfg.run.seed,
        data_seed=cfg.run.seed,
        dataloader_num_workers=cfg.data.num_workers,
    )


# =============================================================================
# Trainer Assembly
# =============================================================================


def build_trainer(
    cfg: SFTCoreConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: DatasetDict,
    training_args: SFTConfig,
    lora_config: LoraConfig,
) -> SFTTrainer:
    """Wrap the model with LoRA and assemble the SFTTrainer.

    Manual QLoRA setup:
    1. Freeze all base model parameters (no fp32 upcast, unlike
       prepare_model_for_kbit_training which OOMs on a full-VRAM 4-bit model).
    2. Enable gradient checkpointing with use_reentrant=False so that
       enable_input_require_grads() is not needed.
    3. Call get_peft_model with autocast_adapter_dtype=False to keep LoRA
       A/B matrices in bf16 instead of fp32, avoiding the 128+ MiB cast OOM.

    Args:
        cfg: Full configuration.
        model: The base quantized model (before LoRA wrapping).
        tokenizer: The model's tokenizer.
        dataset: DatasetDict with 'train' and 'validation' splits.
        training_args: SFTConfig training arguments.
        lora_config: The LoraConfig describing adapter hyperparameters.

    Returns:
        A configured SFTTrainer ready to call .train() on.
    """
    # Step 1: freeze base weights without upcasting to fp32
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: gradient checkpointing (use_reentrant=False avoids the
    # enable_input_require_grads hook that tries to allocate on meta tensors)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Step 3: inject LoRA adapters; keep adapter dtype in bf16
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"LoRA applied: {trainable_params:,} trainable / {total_params:,} total "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # Pass the already-wrapped PEFT model; no peft_config so TRL does not re-wrap
    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )
