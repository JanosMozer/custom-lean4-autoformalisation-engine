"""Syntax Tuning Entry Point (Stage 1 SFT).

Usage:
    python syntaxtuning/main_sft.py <run_name>

Example:
    python syntaxtuning/main_sft.py test1

The run_name argument is passed directly to WandB as the run name, so that
each experiment is uniquely identified and traceable.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import wandb

# Make project root importable from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from syntaxtuning._config import (
    build_lora_config,
    build_trainer,
    build_training_args,
    load_config,
    load_model_and_tokenizer,
    load_syntax_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MainSFT")


def set_seed(seed: int) -> None:
    """Fix all random seeds for full reproducibility.

    Covers Python, NumPy, and PyTorch (CPU + CUDA). CUDA determinism
    is enabled for cuDNN, though it may slightly reduce throughput.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with run_name and optional config_path.
    """
    parser = argparse.ArgumentParser(
        description="Launch Syntax Tuning (Stage 1 SFT) for the Mesh Autoformalizer."
    )
    parser.add_argument(
        "run_name",
        type=str,
        help="Name for this training run. Logged to WandB as the run name.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="syntaxtuning/config.yaml",
        help="Path to the YAML configuration file (default: syntaxtuning/config.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    """Main training entrypoint.

    Sequence:
      1. Parse args & load config.
      2. Seed everything.
      3. Initialize WandB run.
      4. Load model + tokenizer.
      5. Apply LoRA adapters.
      6. Load & format datasets.
      7. Build training arguments.
      8. Build trainer.
      9. Train.
      10. Save final adapter weights.
      11. Finish WandB run.
    """
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Config
    # ------------------------------------------------------------------
    logger.info(f"Loading config from: {args.config}")
    cfg = load_config(args.config, run_name=args.run_name)
    logger.info(f"Starting run: '{cfg.run.name}' in group '{cfg.run.group}'")

    # ------------------------------------------------------------------
    # 2. Reproducibility
    # ------------------------------------------------------------------
    set_seed(cfg.run.seed)

    # ------------------------------------------------------------------
    # 3. WandB Initialisation
    # WandB run name == args.run_name; config dict is logged for
    # hyperparameter tracking and reproducibility.
    # ------------------------------------------------------------------
    run = wandb.init(
        project=cfg.run.project,
        entity=cfg.run.entity if cfg.run.entity else None,
        group=cfg.run.group,
        name=cfg.run.name,
        config={
            "model": vars(cfg.model),
            "lora": {
                "r": cfg.lora.r,
                "lora_alpha": cfg.lora.lora_alpha,
                "lora_dropout": cfg.lora.lora_dropout,
                "target_modules": cfg.lora.target_modules,
            },
            "data": vars(cfg.data),
            "seed": cfg.run.seed,
        },
    )
    logger.info(f"WandB run initialised: {run.url}")

    # ------------------------------------------------------------------
    # 4. Model + Tokenizer
    # ------------------------------------------------------------------
    logger.info("Loading base model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(cfg.model)

    # ------------------------------------------------------------------
    # 5. LoRA Config
    # ------------------------------------------------------------------
    logger.info("Building LoRA configuration...")
    lora_config = build_lora_config(cfg.lora)

    # ------------------------------------------------------------------
    # 6. Datasets
    # ------------------------------------------------------------------
    logger.info("Loading and formatting syntax datasets...")
    dataset = load_syntax_dataset(cfg.data, cfg.prompt, tokenizer)
    logger.info(
        f"Dataset ready: {len(dataset['train'])} train | "
        f"{len(dataset['validation'])} validation"
    )

    # ------------------------------------------------------------------
    # 7. Training Arguments
    # ------------------------------------------------------------------
    logger.info("Building SFT training arguments...")
    training_args = build_training_args(cfg)

    # ------------------------------------------------------------------
    # 8. Trainer (LoRA is applied internally via peft_config)
    # ------------------------------------------------------------------
    logger.info("Assembling SFTTrainer (applying LoRA via peft_config)...")
    trainer = build_trainer(cfg, model, tokenizer, dataset, training_args, lora_config)

    # ------------------------------------------------------------------
    # 9. Train
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"  Starting Syntax Tuning — Run: {cfg.run.name}")
    logger.info("=" * 60)
    trainer.train()

    # ------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------
    final_path = Path(cfg.model.__dict__.get("output_dir", "models/sft_checkpoints")) / "final"
    # Re-read output_dir from training args
    final_path = Path(training_args.output_dir) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"Final model adapter saved to: {final_path}")

    # ------------------------------------------------------------------
    # 11. WandB Finish
    # ------------------------------------------------------------------
    wandb.finish()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
