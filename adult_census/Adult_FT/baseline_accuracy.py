import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re
from huggingface_hub.hf_api import HfFolder
from collections import Counter
from datetime import datetime

# Step 1: Load the dataset
def load_dataset(csv_file, max_samples=50):
    """
    Load and prepare the dataset from a CSV file.
    """
    df = pd.read_csv(csv_file)
    if max_samples:
        df = df.head(max_samples)
    return df

# Step 2: Create a dataset class
class AdultCensusDataset(Dataset):
    def __init__(self, dataframe):
        self.input_text = dataframe['input_text'].tolist()
        self.target_text = dataframe['target_text'].tolist()
        if 'combination' in dataframe.columns:
            self.combination = dataframe['combination'].tolist()
        else:
            self.combination = None
    
    def __len__(self):
        return len(self.input_text)
    
    def __getitem__(self, idx):
        item = {
            'input_text': self.input_text[idx],
            'target_text': self.target_text[idx]
        }
        if self.combination is not None:
            item['combination'] = self.combination[idx]
        return item

# Step 3: Load the model and tokenizer
def load_model():
    """
    Load the pre-trained Llama-2-7B-chat model and tokenizer.
    """
    HfFolder.save_token(os.environ.get("HF_TOKEN"))
    model_name = "NousResearch/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

# Step 4: Inference function
def infer_batch(model, tokenizer, input_texts, batch_size=2, max_input_length=2048, return_sequences=7):
    """
    Run inference on a batch of inputs and return predictions.
    """
    predictions = []
    num_batches = (len(input_texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Processing Batches", unit="batch"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(input_texts))
            batch = input_texts[start_idx:end_idx]
            
            inputs = tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=max_input_length)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            
            outputs = model.generate(
                input_ids, 
                attention_mask=attention_mask, 
                max_new_tokens=20, 
                num_return_sequences=return_sequences
            )
            
            batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print(batch_predictions[0])
            for pred in batch_predictions:
                if 'Model Output:' in pred:
                    _, result = pred.split('Model Output:', 1)
                    predictions.append(result.strip())
                else:
                    predictions.append(pred.strip())
    
    return predictions

# Step 5: Extract predictions
def extract_predictions(text):
    """
    Extract income and race predictions from model output text.
    """
    pattern = r'(Yes|No),\s*(Black|White)'
    match = re.search(pattern, text)
    if match:
        return match.group(1), match.group(2)
    return None, None

# Step 6: Calculate accuracies
def calculate_accuracies(predictions, target_texts):
    """
    Calculate income and race accuracies separately.
    """
    total_samples = len(target_texts)
    income_correct = 0
    race_correct = 0
    joint_correct = 0
    
    for pred, true in zip(predictions, target_texts):
        pred_income, pred_race = extract_predictions(pred)
        true_income, true_race = extract_predictions(true)
        
        if pred_income == true_income:
            income_correct += 1
        if pred_race == true_race:
            race_correct += 1
        if pred_income == true_income and pred_race == true_race:
            joint_correct += 1
    
    income_accuracy = income_correct / total_samples if total_samples > 0 else 0
    race_accuracy = race_correct / total_samples if total_samples > 0 else 0
    joint_accuracy = joint_correct / total_samples if total_samples > 0 else 0
    
    return {
        'income_accuracy': income_accuracy,
        'race_accuracy': race_accuracy,
        'joint_accuracy': joint_accuracy,
        'income_correct': income_correct,
        'race_correct': race_correct,
        'joint_correct': joint_correct,
        'total_samples': total_samples
    }

def save_results(results, predictions, dataset_type):
    """
    Save results and predictions to files
    """
    # Save accuracies
    results_filename = f"results_{dataset_type}.txt"
    with open(results_filename, 'w') as f:
        f.write(f"Results for {dataset_type} dataset:\n")
        f.write(f"Income Accuracy: {results['income_accuracy']:.4f} ({results['income_correct']}/{results['total_samples']})\n")
        f.write(f"Race Accuracy: {results['race_accuracy']:.4f} ({results['race_correct']}/{results['total_samples']})\n")
        f.write(f"Joint Accuracy: {results['joint_accuracy']:.4f} ({results['joint_correct']}/{results['total_samples']})\n")
    
    # Save predictions
    predictions_filename = f"predictions_{dataset_type}.txt"
    with open(predictions_filename, 'w') as f:
        for i, pred in enumerate(predictions):
            f.write(f"Prediction {i+1}: {pred}\n")

def main(csv_file_path, dataset_type):
    """
    Main function to process a dataset
    """
    print(f"\n=== Processing {dataset_type} Data ===")
    print("-" * 30)
    
    # Load the dataset
    print(f"Loading dataset from {csv_file_path}...")
    df = load_dataset(csv_file_path)
    dataset = AdultCensusDataset(df)
    print(f"Loaded {len(dataset)} samples.")
    
    # Load the model
    print("Loading model...")
    model, tokenizer = load_model()
    print("Model loaded.")
    
    # Run inference
    print("Running inference...")
    predictions = infer_batch(
        model, 
        tokenizer, 
        dataset.input_text, 
        batch_size=2, 
        max_input_length=2048,
        return_sequences=7
    )
    
    # Calculate accuracies
    results = calculate_accuracies(predictions, dataset.target_text)
    
    # Print results
    print("\nResults:")
    print(f"Income Accuracy: {results['income_accuracy']:.4f} ({results['income_correct']}/{results['total_samples']})")
    print(f"Race Accuracy: {results['race_accuracy']:.4f} ({results['race_correct']}/{results['total_samples']})")
    print(f"Joint Accuracy: {results['joint_accuracy']:.4f} ({results['joint_correct']}/{results['total_samples']})")
    
    # Save results and predictions
    save_results(results, predictions, dataset_type)

if __name__ == "__main__":
  
    
    # Define file paths
    train_file_path = "census_joint_race_income_train_prompts.csv"
    test_file_path = "census_joint_race_income_test_prompts.csv"
    
    # Process training data
    # main(train_file_path, "train", timestamp)
    
    # Process test data
    main(test_file_path, "test")
    
    print("\nExecution completed!")
    print("Results and predictions have been saved to files.")