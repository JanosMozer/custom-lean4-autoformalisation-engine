"""Stage 2 (RLCF): GRPO with a Lean-compiler reward.

Policy = Stage-1 QLoRA adapter, trained further so generated Lean 4 statements
type-check against Mathlib. Reward is the compiler verdict (the reference model
for KL is the same base with the adapter disabled, handled by TRL for PEFT).
"""
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from syntaxtuning._config import ModelConfig, load_model_and_tokenizer  # reuse GPU-only NF4 loader

# Make the venv-local Lean toolchain visible to reward subprocesses.
_ELAN = str(Path(__file__).resolve().parent.parent / "venv" / "elan" / "bin")
os.environ["PATH"] = _ELAN + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("ELAN_HOME", str(Path(_ELAN).parent))

logger = logging.getLogger("RLCFCore")


# ============================== Config ==============================


@dataclass
class RunConfig:
    project: str = "mesh-autoformalizer"
    entity: str = ""
    group: str = "rlcf"
    name: str = "run"
    seed: int = 42


@dataclass
class DataConfig:
    path: str = "data/rlcf/minif2f.jsonl"
    system: str = (
        "You are an expert mathematician and Lean 4 programmer. Translate the "
        "informal statement into a single formal Lean 4 theorem statement ending "
        "in ':= by sorry'. Output only Lean 4 code."
    )
    max_samples: int = 0  # 0 = all


@dataclass
class LeanConfig:
    mathlib_project: str = "leaneval"
    repl_bin: str = "repl_tool/.lake/build/bin/repl"  # persistent REPL; falls back to cold lean
    header: str = "import Mathlib"
    timeout: int = 60
    num_workers: int = 16          # parallel REPL workers / compiler checks
    # Reward is gated: well-formed-only -> r_wellformed; compiles -> w_compile +
    # w_faithful * similarity_to_reference. Faithfulness dominates once it compiles.
    r_wellformed: float = 0.1
    w_compile: float = 0.3
    w_faithful: float = 0.7


@dataclass
class RLCFCoreConfig:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lean: LeanConfig = field(default_factory=LeanConfig)
    adapter_path: str = "models/sft_checkpoints/exp-1-checkpoint-1650"


def load_config(config_path: str, run_name: Optional[str] = None) -> RLCFCoreConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    cfg = RLCFCoreConfig(
        run=RunConfig(**raw.get("run", {})),
        model=ModelConfig(**raw.get("model", {})),
        data=DataConfig(**raw.get("data", {})),
        lean=LeanConfig(**raw.get("lean", {})),
        adapter_path=raw.get("adapter_path", RLCFCoreConfig.adapter_path),
    )
    if run_name:
        cfg.run.name = run_name
    return cfg


# ============================== Data ==============================


def load_rlcf_dataset(data_cfg: DataConfig) -> Dataset:
    """Prompt-only dataset: informal statement -> conversational prompt.
    The reference formal statement is kept as a column for optional inspection."""
    rows = []
    with open(data_cfg.path) as fh:
        for line in fh:
            d = json.loads(line)
            if not d.get("informal_statement"):
                continue
            rows.append({
                "prompt": [
                    {"role": "system", "content": data_cfg.system},
                    {"role": "user", "content": d["informal_statement"]},
                ],
                "reference": d.get("formal_statement", ""),
            })
    if data_cfg.max_samples:
        rows = rows[: data_cfg.max_samples]
    logger.info(f"RLCF prompts: {len(rows)} from {data_cfg.path}")
    return Dataset.from_list(rows)


# ============================== Reward ==============================


def _clean_code(text: str) -> str:
    """Strip markdown fences / prose; keep the Lean code."""
    m = re.search(r"```(?:lean)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    return text.strip()


def _is_wellformed(c: str) -> bool:
    if not c:
        return False
    has_decl = ("theorem " in c) or ("lemma " in c) or ("example" in c)
    balanced = all(c.count(a) == c.count(b) for a, b in [("(", ")"), ("[", "]"), ("{", "}")])
    return has_decl and (":=" in c or " : " in c) and balanced


def _normalize(code: str) -> str:
    """Canonicalize a Lean statement for structural comparison: drop the theorem
    name, the proof body, hidden-answer placeholders, and whitespace."""
    c = re.sub(r"\b(theorem|lemma)\s+\S+", "theorem T", code)
    c = re.split(r":=", c)[0]                       # keep the statement, drop proof
    c = re.sub(r"answer\([^)]*\)", "ANS", c)         # miniF2F hides the answer
    return re.sub(r"\s+", " ", c).strip()


def _faithfulness(code: str, reference: str) -> float:
    """Structural similarity of the generated statement to the gold formalization
    in [0,1]. Penalizes hallucinated/unfaithful statements that still type-check.
    (Cannot verify hidden numeric answers; those are masked to ANS on both sides.)"""
    import difflib
    if not reference:
        return 0.0
    return difflib.SequenceMatcher(None, _normalize(code), _normalize(reference)).ratio()


def build_reward_fn(lean_cfg: LeanConfig):
    """reward = w_compile * type_checks + w_faithful * similarity_to_reference,
    gated on being well-formed.

    The compiler term rewards valid Lean; the faithfulness term anchors the
    policy to the reference so it cannot reward-hack by emitting trivially
    compilable but incorrect statements. Type-checking runs on a persistent
    Mathlib REPL pool (Mathlib loaded once per worker), parallelized across
    num_workers threads to eliminate cold-start idle time.
    """
    from rlcf.lean_server import make_checker
    check = make_checker(lean_cfg.repl_bin, lean_cfg.mathlib_project,
                         lean_cfg.num_workers, lean_cfg.timeout)
    rw, wc, wf = lean_cfg.r_wellformed, lean_cfg.w_compile, lean_cfg.w_faithful

    def reward(completions, reference=None, **kwargs) -> List[float]:
        codes = [_clean_code(c[0]["content"] if isinstance(c, list) else c) for c in completions]
        refs = reference if reference is not None else [""] * len(codes)
        with ThreadPoolExecutor(max_workers=lean_cfg.num_workers) as ex:
            compiled = list(ex.map(check, codes))
        scores = []
        for code, ok, ref in zip(codes, compiled, refs):
            if not _is_wellformed(code):
                scores.append(0.0)          # not even a valid statement
            elif not ok:
                scores.append(rw)           # well-formed but does not type-check
            else:
                scores.append(wc + wf * _faithfulness(code, ref))  # compiles: faithfulness dominates
        return scores

    reward.__name__ = "lean_compile_faithful_reward"
    return reward


# ============================== Model / Trainer ==============================


def load_policy(cfg: RLCFCoreConfig):
    model, tok = load_model_and_tokenizer(cfg.model)
    tok.padding_side = "left"  # decoder-only generation
    # Without autocast (GRPO generation), FA2's get_target_dtype falls back to the
    # bnb Linear4bit weight dtype (uint8) unless the config is flagged quantized.
    # Mark it so fp32 attention inputs are cast to bf16, not uint8.
    model.config._is_quantized = True
    model.config.dtype = torch.bfloat16
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    # Keep model_type masked for the whole run: peft's qwen3_moe adapter
    # conversion (invoked on load) assumes fused experts and is buggy.
    model.config.model_type = "qwen3_moe_unfused"
    model = PeftModel.from_pretrained(model, cfg.adapter_path, is_trainable=True)
    logger.info(f"Loaded Stage-1 policy adapter from {cfg.adapter_path}")
    return model, tok


def build_grpo_config(cfg: RLCFCoreConfig, config_path: str) -> GRPOConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f).get("training", {})
    os.environ["WANDB_PROJECT"] = cfg.run.project
    os.environ["WANDB_ENTITY"] = cfg.run.entity
    os.environ["WANDB_RUN_GROUP"] = cfg.run.group

    return GRPOConfig(
        output_dir=raw.get("output_dir", "models/rlcf_checkpoints"),
        run_name=cfg.run.name,
        num_generations=raw.get("num_generations", 8),
        per_device_train_batch_size=raw.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=raw.get("gradient_accumulation_steps", 4),
        max_completion_length=raw.get("max_completion_length", 256),
        temperature=raw.get("temperature", 1.0),
        beta=raw.get("beta", 0.04),  # KL coefficient vs the frozen Stage-1 policy
        learning_rate=raw.get("learning_rate", 1.0e-6),
        lr_scheduler_type=raw.get("lr_scheduler_type", "cosine"),
        warmup_steps=raw.get("warmup_steps", 20),
        max_grad_norm=raw.get("max_grad_norm", 0.2),
        num_train_epochs=raw.get("num_train_epochs", 1),
        max_steps=raw.get("max_steps", -1),
        bf16=raw.get("bf16", True),
        optim=raw.get("optim", "paged_adamw_8bit"),
        gradient_checkpointing=False,  # enabled on the model in load_policy
        logging_steps=raw.get("logging_steps", 1),
        save_strategy=raw.get("save_strategy", "steps"),
        save_steps=raw.get("save_steps", 50),
        save_total_limit=raw.get("save_total_limit", 3),
        report_to=raw.get("report_to", "wandb"),
        seed=cfg.run.seed,
    )


def build_trainer(cfg, model, tok, dataset, args, reward_fn) -> GRPOTrainer:
    return GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=args,
        train_dataset=dataset,
        processing_class=tok,
    )
