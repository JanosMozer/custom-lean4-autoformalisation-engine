import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlcf._config import (
    build_grpo_config,
    build_reward_fn,
    build_trainer,
    load_config,
    load_policy,
    load_rlcf_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MainRLCF")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Global seed set to {seed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch RLCF (Stage 2, GRPO).")
    parser.add_argument("run_name", type=str, help="WandB run name.")
    parser.add_argument("--config", type=str, default="rlcf/config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint dir to resume from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, run_name=args.run_name)
    logger.info(f"Starting RLCF run: '{cfg.run.name}'")
    set_seed(cfg.run.seed)

    logger.info("Loading policy (base NF4 + Stage-1 adapter)...")
    model, tokenizer = load_policy(cfg)

    logger.info("Loading RLCF prompt dataset...")
    dataset = load_rlcf_dataset(cfg.data)

    logger.info("Building Lean-compiler reward and GRPO config...")
    reward_fn = build_reward_fn(cfg.lean)
    grpo_config = build_grpo_config(cfg, args.config)

    trainer = build_trainer(cfg, model, tokenizer, dataset, grpo_config, reward_fn)

    logger.info("=" * 60)
    logger.info(f"  Starting RLCF — Run: {cfg.run.name}")
    logger.info("=" * 60)

    original_model_type = model.config.model_type  # already masked; keep so through train()
    try:
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.train(resume_from_checkpoint=args.resume)
        else:
            trainer.train()
    finally:
        model.config.model_type = original_model_type

    final_path = Path(grpo_config.output_dir) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"Final RLCF adapter saved to: {final_path}")


if __name__ == "__main__":
    main()
