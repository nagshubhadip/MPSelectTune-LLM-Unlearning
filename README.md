# MPSelectTune — Prompt-type Selection for Fine-tuning improves Concept Unlearning in LLMs

Official code for the paper:

> **MPSelectTune: Prompt-type Selection for Fine-tuning improves Concept
> Unlearning in LLMs**
> Shubhadip Nag, Srinjoy Das, Agniva Saha, Anushree Ghosh, Soumi Das,
> Tarun Kumar, Suparna Bhattacharya, Sourangshu Bhattacharya.
>
> **Venue:** NeurIPS 2025 — Reliable ML from Unreliable Data Workshop.
> **Paper:** https://openreview.net/forum?id=Jk8sOL97Tg

## Overview

Concept unlearning aims to erase a biased or harmful concept (e.g. **gender** in
profession prediction, or **bio-weapons** knowledge in scientific QA) from an LLM
while preserving its main-task ability. Existing unlearning methods largely
ignore that predictive performance depends strongly on the **prompt type** used
to elicit concept labels — so a model may appear to have unlearned a concept on
average, while some prompt types still recover it with high accuracy.

**MPSelectTune** is a two-stage approach:

1. **Stage 1 — Multi-Prompt Tuning (MPTune).** Fine-tune the model using
   *multiple joint-prediction prompt types* (varying the number and selection
   method of in-context examples) with a **multi-task loss** combining a main
   task loss, a concept loss, a novel **format loss** (which forces the LLM to
   follow the required output format across prompt types), and a next-word
   prediction loss.
2. **Stage 2 — Selection Tuning.** Identify the **worst prompt type** — the one
   with the *highest* concept accuracy after Stage 1 — and fine-tune to minimise
   its concept accuracy. Driving down the worst prompt type reduces concept
   accuracy across *all* prompt types, demonstrating genuine unlearning.

Across benchmarks this yields **2–15%** higher main-task accuracy while reducing
worst-case concept accuracy by up to **17%** vs. recent baselines, and a large
reduction (74%→23%) in the spurious correlation between task and concept
accuracy measured by the **spuriousness-score** metric.

## Repository structure

```
.
├── bios/                       # Main benchmark: Bias-in-Bios (profession + gender)
│   ├── llama-2_fine_tune.py            # MPTune: multi-prompt fine-tuning (Stage 1)
│   ├── llama-2_custom_loss_updated.py  # Custom multi-task loss (task+concept+format)
│   ├── sp_score_bios.ipynb             # Spuriousness-score metric computation
│   ├── test_llama2.ipynb               # Evaluation / inference across prompt types
│   └── prompt_selection_indices/       # In-context-example (ICE) selection indices
│       ├── train_idx_for_prompt_sel_from_ICE.csv
│       ├── val_idx_for_prompt_sel_from_ICE.csv
│       └── test_idx_for_prompt_sel_from_ICE.csv
│
├── adult_census/               # Adult-Census benchmark (income + protected attribute)
│   ├── fine_tune_adult.py
│   └── adult_data.csv
│
├── mmlu_test/                  # MMLU main-task capability evaluation
│   ├── run_inference.py                # 5-shot MMLU evaluation harness
│   ├── mmlu_results.json
│   └── mmlu_5shot_results.json
│
├── modular/                    # Refactored, config-driven package (see modular/README.md)
│   ├── run_finetune.py                 # CLI entry point for a fine-tuning stage
│   ├── requirements.txt
│   └── mpselecttune/                   # config / data / model / trainer / pipeline
│
└── data/README.md              # How to obtain the large .npy prompt tensors
```

The original research scripts (under `bios/`, `adult_census/`, `mmlu_test/`) are
kept as-is for reference. A cleaned-up, reusable implementation of the
Bias-in-Bios pipeline lives under [modular/](modular/README.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets peft trl huggingface_hub \
            numpy pandas scikit-learn tqdm
```

Set your Hugging Face access token via an environment variable (Llama-2 requires
authentication):

```bash
export HF_TOKEN=your_hf_token_here
```

> ⚠️ No tokens are stored in this repository. The scripts read `HF_TOKEN` from
> the environment.

## Data

The large tokenized prompt tensors (`ice_data.npy`, `train_data.npy`,
`val_data.npy`, `test_data.npy`, `one_hot_labels_tok.npy`) are **not** committed
due to their size (~340 MB). See [data/README.md](data/README.md) for how they
are structured and referenced. The lightweight ICE **selection indices** needed
to reproduce the prompt-type construction are included under
[bios/prompt_selection_indices/](bios/prompt_selection_indices).

## Running

Bias-in-Bios (Stage 1 multi-prompt fine-tuning):

```bash
cd bios
python llama-2_fine_tune.py            # expects the .npy tensors in this folder
```

Adult-Census:

```bash
cd adult_census
python fine_tune_adult.py
```

MMLU capability evaluation:

```bash
cd mmlu_test
python run_inference.py
```

The spuriousness-score metric (task–concept correlation) is computed in
[bios/sp_score_bios.ipynb](bios/sp_score_bios.ipynb).

> These are research scripts adapted from a compute-server workspace; some
> contain machine-specific paths that may need adjusting for your environment.

## Limitations and future work

- **Two-stage compute overhead.** Running both the multi-prompt tuning and
  selection tuning stages roughly *doubles* the fine-tuning time relative to
  single-stage baselines (see runtime comparison in the paper). This is the
  main efficiency trade-off for the improved worst-case unlearning.
- **Prompt-type selection.** The method depends on identifying the worst prompt
  type via evaluation on training data. If suitable worst-case prompt types are
  not present in the candidate set, unlearning effectiveness can degrade.
  Automated or online prompt-type selection is a promising direction.
- **Concept scope.** Experiments focus on largely binary concepts (e.g. gender,
  race, bio-weapon knowledge). Extending to multi-class or more abstract
  concepts remains open.

## Citation

```bibtex
@inproceedings{nag2025mpselecttune,
  title     = {MPSelectTune: Prompt-type Selection for Fine-tuning improves
               Concept Unlearning in LLMs},
  author    = {Nag, Shubhadip and Das, Srinjoy and Saha, Agniva and
               Ghosh, Anushree and Das, Soumi and Kumar, Tarun and
               Bhattacharya, Suparna and Bhattacharya, Sourangshu},
  booktitle = {NeurIPS 2025 Workshop on Reliable ML from Unreliable Data},
  year      = {2025},
  url       = {https://openreview.net/forum?id=Jk8sOL97Tg}
}
```
