"""MPSelectTune — modular implementation.

A clean, config-driven reimplementation of the MPSelectTune concept-unlearning
pipeline (Bias-in-Bios benchmark). The original research scripts live in the
parent repository under ``bios/``; this package refactors them into reusable
modules:

- :mod:`mpselecttune.config`   — dataclass configs, prompt templates, label space
- :mod:`mpselecttune.data`     — data loading, prompt construction, tokenization
- :mod:`mpselecttune.model`    — 4-bit QLoRA model & tokenizer construction
- :mod:`mpselecttune.trainer`  — SFT trainer, completion collator, callbacks
- :mod:`mpselecttune.pipeline` — end-to-end multi-prompt fine-tuning driver
"""

from mpselecttune.config import (
    DataConfig,
    ModelConfig,
    PromptConfig,
    TrainConfig,
)

__all__ = ["DataConfig", "ModelConfig", "PromptConfig", "TrainConfig"]
