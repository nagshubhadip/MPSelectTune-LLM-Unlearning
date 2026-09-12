"""Training callbacks."""

from __future__ import annotations

import random

import torch
from transformers import TrainerCallback


class GenerationLoggingCallback(TrainerCallback):
    """Periodically generate on a random test example and print the completion.

    Useful for eyeballing whether the model follows the required output format
    (the format-loss objective) during training.
    """

    def __init__(self, model, tokenizer, test_dataset, marker: str = "### Model Output:"):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.marker = marker
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or state.global_step % args.logging_steps != 0:
            return

        print(f"Step: {state.global_step}")
        for key, value in logs.items():
            print(f"{key}: {value}")

        sample = random.choice(self.test_dataset)
        inputs = self.tokenizer(
            sample["input_text"], return_tensors="pt", truncation=True,
            padding="max_length", max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=8, return_dict_in_generate=True)

        generated = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        start = generated.find(self.marker)
        extracted = generated[start + len(self.marker):].strip() if start != -1 else generated
        print(f"Generated: {extracted}")
        print(f"Target   : {sample['target_text']}")
        print("-" * 50)
