# MPSelectTune — Modular Package

A refactored, config-driven implementation of the MPSelectTune two-stage concept
unlearning pipeline used in the Bias-in-Bios experiments. It reorganizes the
original research script ([../bios/llama-2_fine_tune.py](../bios/llama-2_fine_tune.py))
into a reusable Python package.

## Method

Two-stage concept unlearning on `meta-llama/Llama-2-7b-chat-hf` with 4-bit QLoRA:

- **Stage 1 — Multi-Prompt Tuning:** fine-tune over multiple prompt types with a
  multi-task objective (task + concept + format + next-word prediction). The
  training query embeds a *flipped* protected attribute (gender) while the target
  label keeps the *true* value, producing the adversarial unlearning signal.
- **Stage 2 — Selection Tuning:** re-tune on the worst prompt type (highest
  concept accuracy) only, by pointing the data config at that prompt's index CSV.

## Layout

```
modular/
├── run_finetune.py          # CLI entry point (one fine-tuning stage)
├── requirements.txt
└── mpselecttune/
    ├── __init__.py
    ├── config.py            # PromptConfig, DataConfig, ModelConfig, TrainConfig
    ├── data.py              # prompt construction, .npy loading, DatasetDict build
    ├── model.py             # 4-bit QLoRA model + tokenizer (HF auth via env)
    ├── callbacks.py         # generation-logging callback
    ├── trainer.py           # SFTTrainer + completion-only collator
    └── pipeline.py          # end-to-end driver
```

## Usage

Authenticate with Hugging Face through the environment (never hard-code tokens):

```bash
export HF_TOKEN=hf_your_token_here
pip install -r requirements.txt
```

Run a stage:

```bash
python run_finetune.py --output-dir ./stage1_out --epochs 4 --batch-size 4
```

Or drive it programmatically:

```python
from mpselecttune.config import DataConfig, ModelConfig, TrainConfig, PromptConfig
from mpselecttune.pipeline import run_finetune

run_finetune(
    data_cfg=DataConfig(),
    model_cfg=ModelConfig(),
    train_cfg=TrainConfig(output_dir="./stage1_out"),
    prompt_cfg=PromptConfig(),
)
```

## Notes

- Data (`.npy` arrays and prompt-selection index CSVs) live under [../bios](../bios);
  adjust the paths in `DataConfig` to match your layout.
- The original, unmodified research scripts remain in [../bios](../bios) for reference.
