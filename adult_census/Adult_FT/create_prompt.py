import numpy as np
import os
import random
import pandas as pd
from datasets import Dataset, DatasetDict
import numpy as np
from huggingface_hub import login
import warnings
from transformers import BitsAndBytesConfig
import torch
import matplotlib.pyplot as plt
from huggingface_hub.hf_api import HfFolder
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback, Trainer
import transformers
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, PeftModel

# Suppress warnings
import json
import numpy as np
import pandas as pd
import random
from itertools import chain
from sentence_transformers import SentenceTransformer, util

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load and process the adult census dataset
file_path = "adult_data.csv"
data = pd.read_csv(file_path)

# Check for missing values and handle them
data.replace("?", np.nan, inplace=True)
data.fillna(data.mode().iloc[0], inplace=True)

# Filter to keep only Black and White races
data = data[data['race'].isin(['Black', 'White'])].reset_index(drop=True)

# Convert income to Yes/No format
data['income'] = data['income'].apply(lambda x: 'Yes' if x == '>50K' else 'No')

# Feature engineering - create combined features for embedding
# Exclude race and income since these are what we'll predict
data['combined_features'] = data.apply(
    lambda row: f"Age: {row['age']}, Workclass: {row['workclass']}, Education: {row['education']}, " +
                f"Marital Status: {row['marital.status']}, Occupation: {row['occupation']}, " +
                f"Relationship: {row['relationship']}, Sex: {row['sex']}, " +
                f"Hours per week: {row['hours.per.week']}, Native country: {row['native.country']}",
    axis=1
)

# Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the data
total_samples = len(data)
train_size = int(0.8 * total_samples)
in_context_size = int(0.8 * total_samples)
test_size = total_samples - train_size  # Ensure remaining samples are for testing

train_data = data[:train_size]  # First 80% for training
in_context_data = data[:in_context_size]  # First 80% for in-context (can overlap with train)
test_data = data[train_size:]  # Remaining for testing

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Encode the combined features into embeddings
train_embeddings = model.encode(train_data['combined_features'].tolist(), convert_to_tensor=True)
in_context_embeddings = model.encode(in_context_data['combined_features'].tolist(), convert_to_tensor=True)
test_embeddings = model.encode(test_data['combined_features'].tolist(), convert_to_tensor=True)

# Compute similarity matrices
similarity_matrix_train = util.cos_sim(train_embeddings, in_context_embeddings)
similarity_matrix_test = util.cos_sim(test_embeddings, in_context_embeddings)

# Combination configurations
combinations = [
    ("2-sim-dissim", 2, {"similar": 1, "dissimilar": 1, "random": 0}),
    ("2-half-random", 2, {"similar": 1, "dissimilar": 0, "random": 1}),
    ("2-random", 2, {"similar": 0, "dissimilar": 0, "random": 2}),
    ("3-sim-dissim", 3, {"similar": 2, "dissimilar": 1, "random": 0}),
    ("3-half-random", 3, {"similar": 2, "dissimilar": 0, "random": 1}),
    ("3-random", 3, {"similar": 0, "dissimilar": 0, "random": 3}),
    ("4-sim-dissim", 4, {"similar": 2, "dissimilar": 2, "random": 0}),
    ("4-half-random", 4, {"similar": 2, "dissimilar": 0, "random": 2}),
    ("4-random", 4, {"similar": 0, "dissimilar": 0, "random": 4}),
    ("5-sim-dissim", 5, {"similar": 3, "dissimilar": 2, "random": 0}),
    ("5-half-random", 5, {"similar": 3, "dissimilar": 0, "random": 2}),
    ("5-random", 5, {"similar": 0, "dissimilar": 0, "random": 5}),
]

# Split the training indices into groups
num_training = len(train_data)
group_size = num_training // len(combinations)
groups = [list(range(num_training))[i:i + group_size] for i in range(0, num_training, group_size)]

# Function to ensure demographic balance in in-context examples
def get_balanced_examples(indices, in_context_data, counts):
    """
    Select examples ensuring at least one Black example is included
    
    Parameters:
    indices - candidate indices from similarity/dissimilarity/random selection
    in_context_data - the dataset containing in-context examples
    counts - dict with counts for similar, dissimilar, and random examples
    
    Returns:
    List of selected indices with at least one Black example
    """
    # Convert indices to list if it's a tensor
    if hasattr(indices, 'tolist'):
        indices = indices.tolist()
    
    # Get the race of each candidate example
    races = [in_context_data.iloc[i]['race'] for i in indices]
    
    # Check if any example is Black
    if 'Black' not in races:
        # Find Black examples in the in_context_data
        black_indices = in_context_data[in_context_data['race'] == 'Black'].index.tolist()
        
        if not black_indices:
            print("Warning: No Black examples found in the in-context dataset!")
            return indices
        
        # Replace one random example with a random Black example
        # Prioritize replacing a random example if available, otherwise dissimilar, then similar
        replace_idx = None
        if counts["random"] > 0:
            # Find the index of a random example in the selected indices
            random_start_idx = counts["similar"] + counts["dissimilar"]
            if random_start_idx < len(indices):
                replace_idx = indices[random_start_idx]  # Replace the first random example
        elif counts["dissimilar"] > 0:
            # Find the index of a dissimilar example in the selected indices
            dissimilar_start_idx = counts["similar"]
            if dissimilar_start_idx < len(indices):
                replace_idx = indices[dissimilar_start_idx]  # Replace the first dissimilar example
        else:
            # Replace a similar example if no other types available
            replace_idx = indices[0]  # Replace the first similar example
            
        # Select a random Black example
        black_example_idx = random.choice(black_indices)
        
        # Replace the chosen example
        if replace_idx is not None:
            replace_pos = indices.index(replace_idx)
            indices[replace_pos] = black_example_idx
        
    return indices

# Final indices and combination types for training data
final_indices_train = []
combination_types_train = []

# Process each group for training data
for group, (comb_type, num_examples, counts) in zip(groups, combinations):
    for idx in group:
        similarities = similarity_matrix_train[idx]
        sorted_indices = similarities.argsort(descending=True)
        most_similar = sorted_indices[:counts["similar"]].tolist()
        most_dissimilar = sorted_indices[-counts["dissimilar"]:].tolist() if counts["dissimilar"] > 0 else []
        random_indices = random.sample(range(len(in_context_data)), counts["random"]) if counts["random"] > 0 else []
        selected_indices = list(chain(most_similar, most_dissimilar, random_indices))
        
        # Apply balancing to ensure at least one Black example
        selected_indices = get_balanced_examples(selected_indices, in_context_data, counts)
        
        random.shuffle(selected_indices)
        final_indices_train.append(selected_indices)
        combination_types_train.append(comb_type)

# Process test data
num_test = len(test_data)
group_size_test = num_test // len(combinations)
groups_test = [list(range(num_test))[i:i + group_size_test] for i in range(0, num_test, group_size_test)]

final_indices_test = []
combination_types_test = []

for group, (comb_type, num_examples, counts) in zip(groups_test, combinations):
    for idx in group:
        similarities = similarity_matrix_test[idx]
        sorted_indices = similarities.argsort(descending=True)
        most_similar = sorted_indices[:counts["similar"]].tolist()
        most_dissimilar = sorted_indices[-counts["dissimilar"]:].tolist() if counts["dissimilar"] > 0 else []
        random_indices = random.sample(range(len(in_context_data)), counts["random"]) if counts["random"] > 0 else []
        selected_indices = list(chain(most_similar, most_dissimilar, random_indices))
        
        # Apply balancing to ensure at least one Black example
        selected_indices = get_balanced_examples(selected_indices, in_context_data, counts)
        
        random.shuffle(selected_indices)
        final_indices_test.append(selected_indices)
        combination_types_test.append(comb_type)

# Merged lists
merged_list_train = list(zip(final_indices_train, combination_types_train))
merged_list_test = list(zip(final_indices_test, combination_types_test))

# Format a single record for prompting (excluding race and income)
def format_record(record):
    """Format a census record into a descriptive text, excluding race and income."""
    return (
        f"Person with age {record['age']}, {record['sex']} gender, "
        f"education level of {record['education']}, working as {record['occupation']} "
        f"in {record['workclass']} sector, {record['marital.status']} marital status, "
        f"relationship status of {record['relationship']}, "
        f"working {record['hours.per.week']} hours per week with "
        f"capital gain of {record['capital.gain']} and capital loss of {record['capital.loss']}, "
        f"native to {record['native.country']}."
    )

# Create prompts for joint prediction of income and race
base_template = """You are a socioeconomic analyst. Your task is to jointly predict both a person's race and whether they earn more than $50K per year based on their demographic data.

For each case, provide your prediction in this format:
Income: [Yes or No], Race: [Black or White]
Where "Yes" means income >$50K and "No" means income ≤$50K

Given the examples below, predict both race and income for the new case.

"""

def format_example(record):
    """Format a record for in-context learning examples."""
    description = format_record(record)
    income = record['income']  # Already "Yes" or "No"
    race = record['race']
    return f"""Description: {description}
### Model Output: {income}, {race}"""

def format_test_example(record, is_train=True):
    """Format a test record, optionally including the answer for evaluation."""
    description = format_record(record)
    if is_train:
        income = record['income']  # Already "Yes" or "No" 
        race = record['race']
        return f"""
Now, jointly predict race and income for this person:
Description: {description}
### Model Output: {income}, {race}"""
    else:
        return f"""
Now, jointly predict race and income for this person:
Description: {description}
### Model Output:"""

# Verify that each prompt includes Black examples
def verify_demographic_balance(prompts_data, in_context_data):
    """Verify that each prompt includes at least one Black example"""
    unbalanced_count = 0
    for idx, prompt in enumerate(prompts_data['input_text']):
        has_black = False
        # Check if any 'Black' race is mentioned in the prompt
        if 'Black' in prompt:
            has_black = True
        
        if not has_black:
            unbalanced_count += 1
            print(f"Warning: Prompt {idx} does not include Black examples")
    
    if unbalanced_count > 0:
        print(f"Warning: {unbalanced_count} prompts ({unbalanced_count/len(prompts_data['input_text'])*100:.2f}%) do not include Black examples")
    else:
        print("All prompts are demographically balanced with at least one Black example")
    
    return unbalanced_count

# Generate prompts for training and testing
def get_prompt(idx_list, data_set, in_context_set, is_train=True):
    prompts = []
    targets = []
    combinations = []
    
    for index, (indices, comb_type) in enumerate(idx_list):
        temp_template = base_template
        
        # Add in-context examples
        for i in indices:
            example = format_example(in_context_set.iloc[i])
            temp_template += example + "\n\n"
        
        # Add test example
        record = data_set.iloc[index]
        if is_train:
            temp_template += format_test_example(record, is_train=True)
        else:
            temp_template += format_test_example(record, is_train=False)
        
        prompts.append(temp_template)
        targets.append(f"{record['income']}, {record['race']}")
        combinations.append(comb_type)
    
    return {'input_text': prompts, 'target_text': targets, 'combination': combinations}

# Generate the datasets
train_prompts = get_prompt(merged_list_train, train_data, in_context_data)
test_prompts = get_prompt(merged_list_test, test_data, in_context_data, is_train=False)

# Verify demographic balance in prompts
print("Verifying demographic balance in training prompts:")
train_unbalanced = verify_demographic_balance(train_prompts, in_context_data)
print("\nVerifying demographic balance in testing prompts:")
test_unbalanced = verify_demographic_balance(test_prompts, in_context_data)

# Convert to DataFrames
train_df = pd.DataFrame(train_prompts)
test_df = pd.DataFrame(test_prompts)

# Shuffle for good measure
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
train_df.to_csv("census_joint_race_income_train_prompts.csv", index=False)
test_df.to_csv("census_joint_race_income_test_prompts.csv", index=False)

# Convert DataFrames to Dataset for Hugging Face integration
from datasets import Dataset, DatasetDict

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

print(f"Training examples: {len(train_df)}")
print(f"Test examples: {len(test_df)}")
print(f"Number of in-context examples: {len(in_context_data)}")
print("Data processing complete. Files saved as census_joint_race_income_train_prompts.csv and census_joint_race_income_test_prompts.csv")