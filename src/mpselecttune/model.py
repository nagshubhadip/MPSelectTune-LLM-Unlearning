"""Model and tokenizer construction (4-bit QLoRA).

Handles Hugging Face authentication via the ``HF_TOKEN`` environment variable —
never hard-code tokens.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from huggingface_hub.hf_api import HfFolder
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mpselecttune.config import ModelConfig


def authenticate() -> None:
    """Save the Hugging Face token from the ``HF_TOKEN`` environment variable."""
    token = os.environ.get("HF_TOKEN")
    if token:
        HfFolder.save_token(token)


def build_tokenizer(cfg: ModelConfig, padding_side: str = "right") -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def build_model_and_tokenizer(cfg: ModelConfig) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Load a 4-bit quantized backbone with a LoRA adapter, plus its tokenizer."""
    authenticate()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = True
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=cfg.lora_target_task_type,
    )
    model = get_peft_model(model, peft_config).cuda()

    tokenizer = build_tokenizer(cfg)
    return model, tokenizer


def build_peft_config(cfg: ModelConfig) -> LoraConfig:
    """Return the LoRA config (also passed to the SFT trainer)."""
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=cfg.lora_target_task_type,
    )
