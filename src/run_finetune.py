"""CLI entry point for a single MPSelectTune fine-tuning stage.

Examples
--------
Stage 1 (multi-prompt tuning, all prompt types)::

    HF_TOKEN=hf_xxx python run_finetune.py \
        --npy-dir ../bios --idx-dir ../bios/prompt_selection_indices \
        --output-dir ./stage1_out

Stage 2 (selection tuning, worst prompt type only)::

    HF_TOKEN=hf_xxx python run_finetune.py \
        --idx-glob "worst_type_*.csv" --output-dir ./stage2_out
"""

from __future__ import annotations

import argparse

from mpselecttune.config import DataConfig, ModelConfig, PromptConfig, TrainConfig
from mpselecttune.pipeline import run_finetune


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MPSelectTune fine-tuning")
    p.add_argument("--model-id", default=ModelConfig.model_id)
    p.add_argument("--output-dir", default=TrainConfig.output_dir)
    p.add_argument("--epochs", type=int, default=TrainConfig.num_train_epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig.per_device_train_batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--no-log-generations", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = ModelConfig(model_id=args.model_id)
    train_cfg = TrainConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    run_finetune(
        data_cfg=DataConfig(),
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        prompt_cfg=PromptConfig(),
        log_generations=not args.no_log_generations,
    )


if __name__ == "__main__":
    main()
