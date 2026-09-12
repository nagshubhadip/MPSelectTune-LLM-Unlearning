"""Data loading, prompt construction and tokenization.

The raw arrays are object arrays where each row is ``[id, profession, gender,
bio]`` (gender in ``{"M", "F"}``). The prompt-selection index CSVs specify, for
each example, which in-context examples (ICE) to include and the *prompt type*
(``num_samples``, ``type``) — e.g. 2-random, 3-Sim-Dissim, 4-Half-Random.

The core MPSelectTune idea lives here: for each example we build a *joint*
task+concept prediction prompt across multiple prompt types.
"""

from __future__ import annotations

import ast
from typing import Dict, List

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

from mpselecttune.config import DataConfig, PromptConfig


def _gender_str(code: str, flip: bool = False) -> str:
    """Map a raw gender code (``M``/``F``) to a display string, optionally flipped."""
    male = code == "M"
    if flip:
        male = not male
    return "Male" if male else "Female"


def format_ice_example(bio: str, profession: str, gender_code: str, flip: bool = False) -> str:
    """Render one in-context example (with its gold answer)."""
    gender = _gender_str(gender_code, flip)
    return (
        f"\nBio: {bio}\n"
        "Q: Predict the profession and the gender of the above bio\n"
        f"A: {profession}, {gender}\n"
    )


def format_query_labeled(
    bio: str, profession: str, gender_code: str, marker: str, flip: bool = False
) -> str:
    """Render the query example *with* its answer (train/val)."""
    gender = _gender_str(gender_code, flip)
    return (
        f"\nBio: {bio}\n"
        "Q: Predict the profession and the gender of the above bio\n"
        f"{marker} {profession}, {gender}\n"
    )


def format_query_unlabeled(bio: str, marker: str) -> str:
    """Render the query example *without* its answer (test)."""
    return (
        f"\nBio: {bio}\n"
        "Q: Predict the profession and the gender of the above bio\n"
        f"{marker} \n"
    )


def build_prompts(
    idx_df: pd.DataFrame,
    data: np.ndarray,
    ice_data: np.ndarray,
    prompt_cfg: PromptConfig,
    is_test: bool = False,
) -> Dict[str, List[str]]:
    """Construct joint task+concept prompts for every row in ``idx_df``.

    Returns a dict with ``input_text``, ``target_text`` and ``combination``
    (the prompt type, e.g. ``"3, Sim-Dissim"``).
    """
    inputs, targets, combos = [], [], []

    for index, row in idx_df.iterrows():
        prompt = prompt_cfg.base_template
        for ice_i in ast.literal_eval(row["indices"]):
            prompt += format_ice_example(ice_data[ice_i][3], ice_data[ice_i][1], ice_data[ice_i][2])

        if is_test:
            prompt += format_query_unlabeled(data[index][3], prompt_cfg.output_marker)
        else:
            prompt += format_query_labeled(
                data[index][3], data[index][1], data[index][2],
                prompt_cfg.output_marker, flip=prompt_cfg.flip_train_gender,
            )

        inputs.append(prompt)
        # Target label is always the *true* (unflipped) profession + gender.
        gender = _gender_str(data[index][2], flip=False)
        targets.append(f"{data[index][1]}, {gender}")
        combos.append(f"{row['num_samples']}, {row['type']}")

    return {"input_text": inputs, "target_text": targets, "combination": combos}


def load_arrays(cfg: DataConfig):
    """Load the four object arrays (ICE, train, val, test)."""
    ice = np.load(cfg.ice_path, allow_pickle=True)
    train = np.load(cfg.train_path, allow_pickle=True)
    val = np.load(cfg.val_path, allow_pickle=True)
    test = np.load(cfg.test_path, allow_pickle=True)
    return ice, train, val, test


def load_index_frames(cfg: DataConfig):
    """Load the three prompt-selection index CSVs."""
    return (
        pd.read_csv(cfg.train_idx_path),
        pd.read_csv(cfg.val_idx_path),
        pd.read_csv(cfg.test_idx_path),
    )


def build_dataset_dict(data_cfg: DataConfig, prompt_cfg: PromptConfig) -> DatasetDict:
    """Build a shuffled train/val/test :class:`DatasetDict` of raw prompts."""
    ice, train, val, test = load_arrays(data_cfg)
    idx_train, idx_val, idx_test = load_index_frames(data_cfg)

    splits = {
        "train": build_prompts(idx_train, train, ice, prompt_cfg, is_test=False),
        "validation": build_prompts(idx_val, val, ice, prompt_cfg, is_test=False),
        "test": build_prompts(idx_test, test, ice, prompt_cfg, is_test=True),
    }

    dataset = {}
    for name, cols in splits.items():
        df = pd.DataFrame(cols).sample(frac=1, random_state=data_cfg.shuffle_seed).reset_index(drop=True)
        dataset[name] = Dataset.from_pandas(df)
    return DatasetDict(dataset)


def make_tokenize_fn(tokenizer, data_cfg: DataConfig):
    """Return a batched tokenization function for the prompt/target columns."""

    def tokenize(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=data_cfg.max_input_length,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            examples["target_text"],
            max_length=data_cfg.max_target_length,
            padding="max_length",
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return tokenize


def tokenize_dataset(dataset: DatasetDict, tokenizer, data_cfg: DataConfig) -> DatasetDict:
    """Tokenize all splits, preserving the raw text/combination columns."""
    return dataset.map(make_tokenize_fn(tokenizer, data_cfg), batched=True)
