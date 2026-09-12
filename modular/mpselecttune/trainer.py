"""SFT trainer construction for the multi-prompt / selection tuning stages."""

from __future__ import annotations

from typing import Optional

from transformers import TrainerCallback, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from mpselecttune.config import ModelConfig, TrainConfig
from mpselecttune.model import build_peft_config


def build_training_arguments(cfg: TrainConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        optim=cfg.optim,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        max_grad_norm=cfg.max_grad_norm,
        warmup_ratio=cfg.warmup_ratio,
        group_by_length=cfg.group_by_length,
        lr_scheduler_type=cfg.lr_scheduler_type,
        report_to="none",
    )


def build_completion_collator(tokenizer, marker: str = "\n### Model Output:"):
    """Collator that masks the loss on everything before the output marker.

    Only the tokens after ``### Model Output:`` contribute to the loss, so the
    model learns to produce the completion (task / concept / format objective).
    """
    template_ids = tokenizer.encode(marker, add_special_tokens=False)[2:]
    return DataCollatorForCompletionOnlyLM(template_ids, tokenizer=tokenizer)


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    max_seq_length: int = 2048,
    dataset_text_field: str = "text",
    extra_callbacks: Optional[list[TrainerCallback]] = None,
) -> SFTTrainer:
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=build_peft_config(model_cfg),
        dataset_text_field=dataset_text_field,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=build_training_arguments(train_cfg),
        data_collator=build_completion_collator(tokenizer),
        packing=False,
    )
    for cb in extra_callbacks or []:
        trainer.add_callback(cb)
    return trainer
