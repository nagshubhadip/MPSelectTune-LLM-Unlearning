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
import torch.nn.functional as F
import json
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

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
train_size = int(0.6 * total_samples)
in_context_size = int(0.2 * total_samples)
test_size = total_samples - train_size - in_context_size

train_data = data[:train_size]
in_context_data = data[train_size:train_size + in_context_size]
test_data = data[train_size + in_context_size:]

# Initialize the SentenceTransformer model
from sentence_transformers import SentenceTransformer, util
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
### Model Output: Income: {income}, Race: {race}"""

def format_test_example(record, is_train=True):
    """Format a test record, optionally including the answer for evaluation."""
    description = format_record(record)
    if is_train:
        income = record['income']  # Already "Yes" or "No" 
        race = record['race']
        return f"""
Now, jointly predict race and income for this person:
Description: {description}
### Model Output: Income: {income}, Race: {race}"""
    else:
        return f"""
Now, jointly predict race and income for this person:
Description: {description}
### Model Output:"""

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
        targets.append(f"Income: {record['income']}, Race: {record['race']}")
        combinations.append(comb_type)
    
    return {'input_text': prompts, 'target_text': targets, 'combination': combinations}

# Generate the datasets
train_prompts = get_prompt(merged_list_train, train_data, in_context_data)
test_prompts = get_prompt(merged_list_test, test_data, in_context_data, is_train=False)

# Convert to DataFrames
train_df = pd.DataFrame(train_prompts)
test_df = pd.DataFrame(test_prompts)

# Shuffle for good measure
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Convert DataFrames to Dataset for Hugging Face integration
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# Set up GPU and environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Configure LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set your HuggingFace token
HfFolder.save_token(os.environ.get("HF_TOKEN"))

# Load model
model_id = "NousResearch/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map={"": 0}, 
    trust_remote_code=True
)

# Configure model
model.config.use_cache = True
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load another tokenizer for generation
tokenizer2 = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer2.pad_token = tokenizer2.eos_token
tokenizer2.padding_side = "left"

# Prepare model for training
model.enable_input_require_grads()
model = get_peft_model(model, peft_config).cuda()

# Analyze token lengths
def analyze_token_lengths(dataset, tokenizer):
    """Analyze the distribution of token lengths in the dataset"""
    lengths = []
    for example in dataset['input_text']:
        # Tokenize without padding or truncation to get actual length
        tokens = tokenizer(example, truncation=False, padding=False)['input_ids']
        lengths.append(len(tokens))
    
    # Convert to numpy array for statistical operations
    lengths = np.array(lengths)
    
    # Print statistics
    print("\n===== TOKEN LENGTH ANALYSIS =====")
    print(f"Average length: {np.mean(lengths):.2f}")
    print(f"Median length: {np.median(lengths):.2f}")
    print(f"90th percentile: {np.percentile(lengths, 90):.2f}")
    print(f"95th percentile: {np.percentile(lengths, 95):.2f}")
    print(f"99th percentile: {np.percentile(lengths, 99):.2f}")
    print(f"Maximum length: {np.max(lengths)}")
    
    # Optional: Plot histogram
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=50)
        plt.axvline(x=np.percentile(lengths, 95), color='r', linestyle='--', label='95th percentile')
        plt.axvline(x=np.percentile(lengths, 99), color='g', linestyle='--', label='99th percentile')
        plt.legend()
        plt.xlabel('Token Length')
        plt.ylabel('Frequency')
        plt.title('Distribution of Input Text Token Lengths')
        plt.savefig('token_length_distribution.png')
        print("Token length distribution plot saved as 'token_length_distribution.png'")
    except Exception as e:
        print(f"Could not create plot: {e}")
        
    return lengths

# Analyze token lengths
print("Analyzing token lengths in your dataset...")
train_lengths = analyze_token_lengths(dataset_dict['train'], tokenizer)

# Determine optimal max length (using 95th percentile)
optimal_max_length = int(np.percentile(train_lengths, 100))
# Round to the nearest multiple of 128 for better memory efficiency
optimal_max_length = ((optimal_max_length + 127) // 128) * 128
print(f"Using optimal max length: {optimal_max_length}")

# Function to get token IDs for output format components
def get_output_format_token_ids(tokenizer):
    """Get token IDs for output format components"""
    return {
        'yes': tokenizer.encode("Yes", add_special_tokens=False)[0],
        'no': tokenizer.encode("No", add_special_tokens=False)[0],
        'black': tokenizer.encode("Black", add_special_tokens=False)[0],
        'white': tokenizer.encode("White", add_special_tokens=False)[0],
        'comma': tokenizer.encode(",", add_special_tokens=False)[0],
        'income_prefix': tokenizer.encode("Income:", add_special_tokens=False),
        'race_prefix': tokenizer.encode("Race:", add_special_tokens=False),
        'model_output': tokenizer.encode("### Model Output:", add_special_tokens=False),
        'income_yes_full': tokenizer.encode("Income: Yes", add_special_tokens=False),
        'income_no_full': tokenizer.encode("Income: No", add_special_tokens=False),
        'race_black_full': tokenizer.encode("Race: Black", add_special_tokens=False),
        'race_white_full': tokenizer.encode("Race: White", add_special_tokens=False)
    }

# Initialize token IDs for formats
format_token_ids = get_output_format_token_ids(tokenizer)
print("Format token IDs:", format_token_ids)

# Create a function to save loss statistics
def save_loss_statistics(loss_dict, filepath='./results_llama2_adult_census/loss_statistics.json'):
    """Save loss statistics to a file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(loss_dict, f, indent=2)
        print(f"Loss statistics saved to {filepath}")
    except Exception as e:
        print(f"Error saving loss statistics: {e}")

