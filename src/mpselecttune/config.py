"""Configuration objects, prompt templates and the label space.

Everything that was previously hard-coded at module scope in the original
scripts is collected here so experiments can be configured explicitly instead
of by editing source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Label space (Bias-in-Bios) — kept for backward compatibility
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
class TaskConfig:
    """A concept-unlearning benchmark: a main task plus a concept to unlearn.

    MPSelectTune is benchmark-agnostic — the same two-stage pipeline applies to
    any (task, concept) pair. This object captures what changes between
    benchmarks so the rest of the package stays generic.

    Attributes
    ----------
    name:
        Short benchmark identifier (e.g. ``"bios"``, ``"adult"``).
    task_name:
        Human-readable name of the main task (used in the prompt question).
    label_space:
        Candidate labels for the main task, shown in the prompt header.
    concept_name:
        Name of the protected concept to unlearn (e.g. ``"gender"``).
    concept_classes:
        The possible concept values (e.g. ``["Male", "Female"]``).
    header:
        Optional prompt header. If ``None`` a default header listing
        ``label_space`` is generated.
    """

    name: str = "bios"
    task_name: str = "profession"
    label_space: List[str] = field(default_factory=lambda: list(PROFESSIONS))
    concept_name: str = "gender"
    concept_classes: List[str] = field(default_factory=lambda: ["Male", "Female"])
    header: str | None = None

    @property
    def base_template(self) -> str:
        if self.header is not None:
            return self.header
        labels = " ".join(f"{p}," for p in self.label_space)
        return (
            f"The list of possible {self.task_name}s are:\n"
            f"[ {labels}]\n\n"
            "Examples:\n"
        )


# ---------------------------------------------------------------------------
# Built-in benchmark presets
# ---------------------------------------------------------------------------
BIOS_TASK = TaskConfig(
    name="bios", task_name="profession", label_space=list(PROFESSIONS),
    concept_name="gender", concept_classes=["Male", "Female"],
)
ADULT_TASK = TaskConfig(
    name="adult", task_name="income", label_space=["<=50K", ">50K"],
    concept_name="race", concept_classes=["White", "Black", "Asian-Pac-Islander",
                                          "Amer-Indian-Eskimo", "Other"],
)
RT_GENDER_TASK = TaskConfig(
    name="rt_gender", task_name="response class", label_space=["0", "1"],
    concept_name="gender", concept_classes=["Male", "Female"],
)
JIGSAW_TASK = TaskConfig(
    name="jigsaw", task_name="toxicity", label_space=["non-toxic", "toxic"],
    concept_name="identity", concept_classes=["mentioned", "not-mentioned"],
)

TASK_PRESETS = {
    "bios": BIOS_TASK,
    "adult": ADULT_TASK,
    "rt_gender": RT_GENDER_TASK,
    "jigsaw": JIGSAW_TASK,
}


@dataclass
class PromptConfig:
    """Prompt-construction settings.

    Attributes
    ----------
    task:
        The benchmark's (task, concept) definition. Defaults to Bias-in-Bios.
    output_marker:
        Marker placed before the model's answer (used by the completion-only
        collator to mask the prompt during loss computation).
    flip_train_concept:
        If ``True``, the concept value shown in the *query* in-context answer is
        flipped for training examples. This is the adversarial signal that
        drives concept unlearning while preserving the task label.
    """

    task: TaskConfig = field(default_factory=lambda: BIOS_TASK)
    output_marker: str = OUTPUT_MARKER
    flip_train_concept: bool = True

    # Backwards-compatible alias for the old attribute name.
    @property
    def flip_train_gender(self) -> bool:
        return self.flip_train_concept

    @property
    def professions(self) -> List[str]:
        return self.task.label_space

    @property
    def base_template(self) -> str:
        return self.task.base_template


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
