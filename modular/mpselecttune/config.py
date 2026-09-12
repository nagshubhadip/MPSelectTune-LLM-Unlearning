"""Configuration objects, prompt templates and the label space.

Everything that was previously hard-coded at module scope in the original
scripts is collected here so experiments can be configured explicitly instead
of by editing source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Label space (Bias-in-Bios)
# ---------------------------------------------------------------------------
# Main task = profession prediction; concept (to be unlearned) = gender.
PROFESSIONS: List[str] = [
    "psychologist", "poet", "photographer", "nurse", "software_engineer",
    "comedian", "pastor", "architect", "chiropractor", "dentist", "model",
    "interior_designer", "teacher", "accountant", "rapper", "yoga_teacher",
    "paralegal", "surgeon", "painter", "composer", "dj", "personal_trainer",
    "physician", "journalist", "dietitian", "filmmaker", "attorney", "professor",
]

# The output-format marker the model must learn to emit before its answer.
# The format loss encourages the model to respect this across prompt types.
OUTPUT_MARKER = "### Model Output:"


@dataclass
class PromptConfig:
    """Prompt-construction settings.

    Attributes
    ----------
    professions:
        The candidate profession list shown in the prompt header.
    output_marker:
        Marker placed before the model's answer (used by the completion-only
        collator to mask the prompt during loss computation).
    flip_train_gender:
        If ``True``, the gender shown in the *query* in-context answer is
        flipped for training examples. This is the adversarial signal that
        drives concept unlearning while preserving the task label.
    """

    professions: List[str] = field(default_factory=lambda: list(PROFESSIONS))
    output_marker: str = OUTPUT_MARKER
    flip_train_gender: bool = True

    @property
    def base_template(self) -> str:
        professions = " ".join(f"{p}," for p in self.professions)
        return (
            "The list of possible professions are:\n"
            f"[ {professions}]\n\n"
            "Examples:\n"
        )


@dataclass
class DataConfig:
    """Paths to the tokenized tensors and prompt-selection index CSVs."""

    ice_path: str = "ice_data.npy"
    train_path: str = "train_data.npy"
    val_path: str = "val_data.npy"
    test_path: str = "test_data.npy"

    train_idx_path: str = "train_idx_for_prompt_sel_from_ICE.csv"
    val_idx_path: str = "val_idx_for_prompt_sel_from_ICE.csv"
    test_idx_path: str = "test_idx_for_prompt_sel_from_ICE.csv"

    shuffle_seed: int = 42
    max_input_length: int = 2048
    max_target_length: int = 8


@dataclass
class ModelConfig:
    """Backbone + 4-bit QLoRA settings."""

    model_id: str = "meta-llama/Llama-2-7b-chat-hf"
    device: str = "cuda:0"

    # BitsAndBytes 4-bit
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = False

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_task_type: str = "CAUSAL_LM"


@dataclass
class TrainConfig:
    """HuggingFace ``TrainingArguments`` values used by the original pipeline."""

    output_dir: str = "./results_llama"
    save_dir: str = "./fine_tuned_llama2"
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    group_by_length: bool = True
    logging_steps: int = 25
