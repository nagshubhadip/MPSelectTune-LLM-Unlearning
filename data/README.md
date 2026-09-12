# Data

The Bias-in-Bios experiments consume pre-tokenized prompt tensors that are too
large to commit to git (~340 MB total). These are excluded via
[.gitignore](../.gitignore) and must be placed alongside the scripts in
[../bios/](../bios) before running.

## Expected files (in `bios/`)

| File | Approx. size | Description |
|------|-------------|-------------|
| `ice_data.npy` | ~128 MB | In-Context-Example (ICE) pool used to build prompts |
| `train_data.npy` | ~128 MB | Tokenized training prompts |
| `val_data.npy` | ~43 MB | Tokenized validation prompts |
| `test_data.npy` | ~43 MB | Tokenized test prompts |
| `one_hot_labels_tok.npy` | ~1 MB | Tokenized one-hot label targets |

All are saved with `np.save(..., allow_pickle=True)` and loaded with
`np.load(..., allow_pickle=True)`.

## Prompt construction

The prompts are built by selecting in-context examples from the ICE pool using
the index CSVs in
[../bios/prompt_selection_indices/](../bios/prompt_selection_indices):

- `train_idx_for_prompt_sel_from_ICE.csv`
- `val_idx_for_prompt_sel_from_ICE.csv`
- `test_idx_for_prompt_sel_from_ICE.csv`

These indices define the different **prompt types** (e.g. 2-random,
3-Sim-Dissim, 4-Half-Random) that MPSelectTune tunes over. Regenerate the `.npy`
tensors from the raw Bias-in-Bios dataset using these indices, or contact the
authors for the preprocessed tensors.
