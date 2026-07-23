import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import wandb

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Syntax Tuning (Stage 1 SFT).")
    parser.add_argument("run_name", type=str, help="WandB run name.")
    parser.add_argument(
        "--config",
        type=str,
        default="syntaxtuning/config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint dir to resume training from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"Loading config from: {args.config}")
    cfg = load_config(args.config, run_name=args.run_name)
    logger.info(f"Starting run: '{cfg.run.name}' in group '{cfg.run.group}'")

    set_seed(cfg.run.seed)

    # wandb is initialised and owned by the Trainer (report_to="wandb"); a manual
    # wandb.init here would desync the step counter and drop per-step loss logs.

    logger.info("Loading base model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(cfg.model)

    logger.info("Building LoRA configuration...")
    lora_config = build_lora_config(cfg.lora)

    logger.info("Loading and formatting syntax datasets...")
    dataset = load_syntax_dataset(cfg.data, cfg.prompt, tokenizer)
    logger.info(f"Dataset ready: {len(dataset['train'])} train | {len(dataset['validation'])} validation")

    logger.info("Building SFT training arguments...")
    training_args = build_training_args(cfg, args.config)

    logger.info("Assembling SFTTrainer...")
    trainer = build_trainer(cfg, model, tokenizer, dataset, training_args, lora_config)

    logger.info("=" * 60)
    logger.info(f"  Starting Syntax Tuning — Run: {cfg.run.name}")
    logger.info("=" * 60)
    # peft's qwen3_moe adapter conversion (invoked on load_adapter during resume
    # and end-of-training best-model reload) assumes fused experts and is buggy.
    # Mask model_type for the whole train() so it uses our native unfused adapter.
    original_model_type = trainer.model.config.model_type
    trainer.model.config.model_type = "qwen3_moe_unfused"
    try:
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.train(resume_from_checkpoint=args.resume)
        else:
            trainer.train()
    finally:
        trainer.model.config.model_type = original_model_type

    final_path = Path(training_args.output_dir) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"Final model adapter saved to: {final_path}")

    wandb.finish()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
