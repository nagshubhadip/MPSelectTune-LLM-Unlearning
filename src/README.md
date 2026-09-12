# MPSelectTune — Reusable Package (`src/`)

A refactored, config-driven implementation of the MPSelectTune two-stage concept
unlearning pipeline. It reorganizes the original research script
([../bios/llama-2_fine_tune.py](../bios/llama-2_fine_tune.py)) into a reusable
Python package (standard `src` layout).

The reference implementation targets **Bias-in-Bios**, but the pipeline is
**benchmark-agnostic**: point a `TaskConfig` (or a built-in preset) at another
`(task, concept)` pair — e.g. Adult (income/race), RT-Gender, or Jigsaw
toxicity — to reuse the same approach. The corresponding raw research scripts
live in [../adult_census](../adult_census), [../fine_tune_filtered](../fine_tune_filtered),
[../jigsaw](../jigsaw) and [../RT_gender](../RT_gender).

## Method

Two-stage concept unlearning on `meta-llama/Llama-2-7b-chat-hf` with 4-bit QLoRA:

- **Stage 1 — Multi-Prompt Tuning:** fine-tune over multiple prompt types with a
  multi-task objective (task + concept + format + next-word prediction). The
  training query embeds a *flipped* protected concept while the target label
  keeps the *true* value, producing the adversarial unlearning signal.
- **Stage 2 — Selection Tuning:** re-tune on the worst prompt type (highest
  concept accuracy) only, by pointing the data config at that prompt's index CSV.

## Layout

```
src/
├── run_finetune.py          # CLI entry point (one fine-tuning stage)
├── requirements.txt
└── mpselecttune/
    ├── __init__.py
    ├── config.py            # TaskConfig + presets, PromptConfig, DataConfig, ModelConfig, TrainConfig
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
    prompt_cfg=PromptConfig(),          # defaults to the Bias-in-Bios task
)
```

### Switching benchmarks

```python
from mpselecttune.config import PromptConfig, TASK_PRESETS, ADULT_TASK

# Use a built-in preset ...
prompt_cfg = PromptConfig(task=ADULT_TASK)          # Adult income / race
prompt_cfg = PromptConfig(task=TASK_PRESETS["rt_gender"])

# ... or define your own (task, concept) pair:
from mpselecttune.config import TaskConfig
my_task = TaskConfig(
    name="jigsaw", task_name="toxicity",
    label_space=["non-toxic", "toxic"],
    concept_name="identity", concept_classes=["mentioned", "not-mentioned"],
)
prompt_cfg = PromptConfig(task=my_task)
```

## Notes

- Data (`.npy` arrays and prompt-selection index CSVs) live under the benchmark
  folders (e.g. [../bios](../bios)); adjust the paths in `DataConfig` to match
  your layout.
- The original, unmodified research scripts remain in the benchmark folders for
  reference.
````
