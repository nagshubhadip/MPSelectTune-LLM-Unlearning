"""End-to-end pipeline wiring config -> data -> model -> trainer."""

from __future__ import annotations

from mpselecttune.callbacks import GenerationLoggingCallback
from mpselecttune.config import DataConfig, ModelConfig, PromptConfig, TrainConfig
from mpselecttune.data import build_dataset_dict
from mpselecttune.model import build_model_and_tokenizer
from mpselecttune.trainer import build_trainer


def run_finetune(
    data_cfg: DataConfig | None = None,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
    prompt_cfg: PromptConfig | None = None,
    log_generations: bool = True,
):
    """Run one fine-tuning stage (multi-prompt tuning or selection tuning).

    The stage is determined by which prompt-selection index CSVs are referenced
    in ``data_cfg`` — all prompt types for Stage 1, the worst prompt type only
    for Stage 2.
    """
    data_cfg = data_cfg or DataConfig()
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()
    prompt_cfg = prompt_cfg or PromptConfig()

    dataset = build_dataset_dict(data_cfg, prompt_cfg)
    model, tokenizer = build_model_and_tokenizer(model_cfg)

    callbacks = []
    if log_generations and "test" in dataset:
        callbacks.append(
            GenerationLoggingCallback(model, tokenizer, dataset["test"], prompt_cfg.output_marker)
        )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        extra_callbacks=callbacks,
    )

    trainer.train()
    trainer.model.save_pretrained(train_cfg.output_dir)
    tokenizer.save_pretrained(train_cfg.output_dir)
    return trainer
