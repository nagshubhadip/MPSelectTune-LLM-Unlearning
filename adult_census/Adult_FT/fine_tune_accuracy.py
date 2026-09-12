import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import re
import os
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

# Step 3: Load the model and tokenizer with PEFT
def load_fine_tuned_model(base_model_name, peft_model_path):
    """
    Load the fine-tuned model using PEFT.
    """
    HfFolder.save_token(os.environ.get("HF_TOKEN"))
    
    # Load the base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    print(f"Loading base model: {base_model_name}")
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:1"
    )
    
    print(f"Loading PEFT model from: {peft_model_path}")
    print(f"Files in directory: {os.listdir(peft_model_path)}")
    
    # Check if adapter_model.safetensors exists and rename it if needed
    if "adapter_model.safetensors" in os.listdir(peft_model_path) and not "adapter_model.bin" in os.listdir(peft_model_path):
        print("Using adapter_model.safetensors for loading")
        # No need to rename, PEFT can now handle .safetensors files
    
    try:
        # Try first with existing config
        print("Attempting to load model with PeftModel.from_pretrained")
        model = PeftModel.from_pretrained(
            base_model, 
            peft_model_path,
            device_map="cuda:1"
        )
    except Exception as e:
        print(f"Error loading with PeftModel: {e}")
        # If that fails, try a different approach
        print("Attempting to load model with AutoModelForCausalLM.from_pretrained")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                peft_model_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda:1"
            )
        except Exception as e2:
            print(f"Error with alternative loading approach: {e2}")
            raise RuntimeError(f"Failed to load model: {e}\nAnd alternative approach: {e2}")
    
    model.eval()
    print("Model loaded successfully")
    return model, tokenizer

# Step 4: Inference function with improved error handling
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
            
            # Add error handling around tokenization and generation
            try:
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
                
                # Process each prediction
                for j, pred in enumerate(batch_predictions):
                    if j < len(batch): # Only take the first prediction for each input
                        if 'Model Output:' in pred:
                            _, result = pred.split('Model Output:', 1)
                            predictions.append(result.strip())
                        else:
                            # Extract only the generated part after the input text
                            input_text = batch[j % len(batch)]
                            if input_text in pred:
                                result = pred[len(input_text):].strip()
                                predictions.append(result)
                            else:
                                # If we can't find the input text exactly, just use the tail end
                                predictions.append(pred.strip())
            
            except Exception as e:
                print(f"Error during inference for batch {i}: {e}")
                # Add placeholder predictions for this batch
                for _ in range(start_idx, end_idx):
                    predictions.append("Error: Failed to generate prediction")
    
    return predictions

# Step 5: Extract predictions
def extract_predictions(text):
    """
    Extract income and race predictions from model output text.
    """
    # More robust pattern matching
    pattern = r'(Yes|No)\s*,\s*(Black|White)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)
    
    # Try alternative formats
    alt_pattern = r'Income:\s*(Yes|No).*Race:\s*(Black|White)'
    alt_match = re.search(alt_pattern, text, re.IGNORECASE)
    if alt_match:
        return alt_match.group(1), alt_match.group(2)
    
    # If no match found
    print(f"Warning: Could not extract prediction from text: '{text}'")
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
        
        if pred_income and true_income and pred_income.lower() == true_income.lower():
            income_correct += 1
        if pred_race and true_race and pred_race.lower() == true_race.lower():
            race_correct += 1
        if (pred_income and true_income and pred_income.lower() == true_income.lower() and 
            pred_race and true_race and pred_race.lower() == true_race.lower()):
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

def save_results(results, predictions, target_texts, dataset_type):
    """
    Save results and predictions to files
    """
    # Save accuracies
    results_filename = f"results_{dataset_type}_fine_tuned.txt"
    with open(results_filename, 'w') as f:
        f.write(f"Results for {dataset_type} dataset (fine-tuned model):\n")
        f.write(f"Income Accuracy: {results['income_accuracy']:.4f} ({results['income_correct']}/{results['total_samples']})\n")
        f.write(f"Race Accuracy: {results['race_accuracy']:.4f} ({results['race_correct']}/{results['total_samples']})\n")
        f.write(f"Joint Accuracy: {results['joint_accuracy']:.4f} ({results['joint_correct']}/{results['total_samples']})\n")
    
    # Save detailed predictions
    predictions_filename = f"predictions_{dataset_type}_fine_tuned.txt"
    with open(predictions_filename, 'w') as f:
        for i, (pred, target) in enumerate(zip(predictions, target_texts)):
            f.write(f"Example {i+1}:\n")
            f.write(f"Prediction: {pred}\n")
            f.write(f"Target: {target}\n")
            pred_income, pred_race = extract_predictions(pred)
            true_income, true_race = extract_predictions(target)
            f.write(f"Extracted - Pred: Income={pred_income}, Race={pred_race} | True: Income={true_income}, Race={true_race}\n")
            f.write("-" * 50 + "\n")

def main(csv_file_path, dataset_type, base_model_name, peft_model_path):
    """
    Main function to process a dataset using the fine-tuned model
    """
    print(f"\n=== Processing {dataset_type} Data with Fine-tuned Model ===")
    print("-" * 50)
    
    # Load the dataset
    print(f"Loading dataset from {csv_file_path}...")
    df = load_dataset(csv_file_path)
    dataset = AdultCensusDataset(df)
    print(f"Loaded {len(dataset)} samples.")
    
    # Load the fine-tuned model
    print(f"Loading fine-tuned model from {peft_model_path}...")
    model, tokenizer = load_fine_tuned_model(base_model_name, peft_model_path)
    print("Fine-tuned model loaded.")
    
    # Run inference
    print("Running inference with fine-tuned model...")
    predictions = infer_batch(
        model, 
        tokenizer, 
        dataset.input_text, 
        batch_size=2, 
        max_input_length=2048,
        return_sequences=1  # Changed from 7 to 1 for better extraction
    )
    
    # Calculate accuracies
    results = calculate_accuracies(predictions, dataset.target_text)
    
    # Print results
    print("\nResults (Fine-tuned Model):")
    print(f"Income Accuracy: {results['income_accuracy']:.4f} ({results['income_correct']}/{results['total_samples']})")
    print(f"Race Accuracy: {results['race_accuracy']:.4f} ({results['race_correct']}/{results['total_samples']})")
    print(f"Joint Accuracy: {results['joint_accuracy']:.4f} ({results['joint_correct']}/{results['total_samples']})")
    
    # Save results and predictions
    save_results(results, predictions, dataset.target_text, dataset_type)

if __name__ == "__main__":
    # Define file paths
    train_file_path = "census_joint_race_income_train_prompts.csv"
    test_file_path = "census_joint_race_income_test_prompts.csv"
    
    # Define model paths
    base_model_name = "NousResearch/Llama-2-7b-chat-hf"
    peft_model_path = "./fine_tuned_llama2_adult_census_final"  # Use a relative path with ./ prefix
    
    # Process test data with fine-tuned model
    main(test_file_path, "test", base_model_name, peft_model_path)
    
    print("\nExecution completed!")
    print("Results and predictions have been saved to files.")