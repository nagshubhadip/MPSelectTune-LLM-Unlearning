"""MPSelectTune — modular implementation.

A clean, config-driven reimplementation of the MPSelectTune concept-unlearning
pipeline. The reference implementation targets Bias-in-Bios, but the pipeline is
benchmark-agnostic: point :class:`~mpselecttune.config.TaskConfig` (or one of the
built-in presets) at another (task, concept) pair — e.g. Adult (income/race),
RT-Gender, or Jigsaw toxicity — to reuse the same two-stage approach. The
original research scripts live in the parent repository under ``bios/``,
``adult_census/``, ``fine_tune_filtered/``, ``jigsaw/`` and ``RT_gender/``.

Modules:

- :mod:`mpselecttune.config`   — dataclass configs, task presets, prompt templates
- :mod:`mpselecttune.data`     — data loading, prompt construction, tokenization
- :mod:`mpselecttune.model`    — 4-bit QLoRA model & tokenizer construction
- :mod:`mpselecttune.trainer`  — SFT trainer, completion collator, callbacks
- :mod:`mpselecttune.pipeline` — end-to-end multi-prompt fine-tuning driver
"""

from mpselecttune.config import (
    ADULT_TASK,
    BIOS_TASK,
    JIGSAW_TASK,
    RT_GENDER_TASK,
    TASK_PRESETS,
    DataConfig,
    ModelConfig,
    PromptConfig,
    TaskConfig,
    TrainConfig,
)

__all__ = [
    "DataConfig",
    "ModelConfig",
    "PromptConfig",
    "TaskConfig",
    "TrainConfig",
    "TASK_PRESETS",
    "BIOS_TASK",
    "ADULT_TASK",
    "RT_GENDER_TASK",
    "JIGSAW_TASK",
]
