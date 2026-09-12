import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
import argparse
from typing import List, Dict, Tuple
import gc
import random

class MistralMMLU5ShotEvaluator:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", device: str = "auto"):
        """
        Initialize the Mistral MMLU evaluator with 5-shot prompting.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
            
        self.model.eval()
        
        # Store few-shot examples for each subject
        self.few_shot_examples = {}
        
    def load_few_shot_examples(self, subject: str, num_shots: int = 5) -> List[Dict]:
        """
        Load few-shot examples for a given subject from the development set.
        
        Args:
            subject: MMLU subject name
            num_shots: Number of examples to load
            
        Returns:
            List of example dictionaries
        """
        if subject in self.few_shot_examples:
            return self.few_shot_examples[subject]
            
        try:
            # Load development set for few-shot examples
            dev_dataset = load_dataset("cais/mmlu", subject)['dev']
        except:
            # Fallback to alternative dataset name format
            dev_dataset = load_dataset("hendrycks_test", subject)['dev']
        
        # Randomly sample examples (with fixed seed for reproducibility)
        random.seed(42)
        if len(dev_dataset) >= num_shots:
            indices = random.sample(range(len(dev_dataset)), num_shots)
        else:
            indices = list(range(len(dev_dataset)))
            
        examples = []
        choice_labels = ['A', 'B', 'C', 'D']
        
        for idx in indices:
            example = dev_dataset[idx]
            examples.append({
                'question': example['question'],
                'choices': example['choices'],
                'answer': choice_labels[example['answer']]
            })
        
        self.few_shot_examples[subject] = examples
        return examples
        
    def format_few_shot_prompt(self, question: str, choices: List[str], subject: str) -> str:
        """
        Format the prompt with 5-shot examples for Mistral Instruct model.
        
        Args:
            question: The question text
            choices: List of answer choices
            subject: MMLU subject for getting relevant examples
            
        Returns:
            Formatted prompt string with few-shot examples
        """
        choice_labels = ['A', 'B', 'C', 'D']
        
        # Get few-shot examples
        few_shot_examples = self.load_few_shot_examples(subject, num_shots=5)
        
        # Build the prompt with strict answer format instructions first
        prompt_parts = [
            "<s>[INST] You are a helpful assistant. Answer the following multiple choice questions by selecting the correct answer.",
            "",
            "IMPORTANT ANSWER FORMAT:",
            "Please respond with only the letter of the correct answer (A, B, C, or D).",
            "DON'T PRINT ANYTHING ELSE (ONLY 'A' OR 'B' OR 'C' OR 'D').",
            "DON'T EVEN PRINT THE ANSWER LIKE 'A. 4' - THIS IS WRONG.",
            "CORRECT FORMAT IS JUST 'A' (single letter only).",
            "",
            "Here are some examples following this format:",
            ""
        ]
        
        # Add few-shot examples
        for i, example in enumerate(few_shot_examples, 1):
            example_choices = '\n'.join([f"{label}. {choice}" 
                                       for label, choice in zip(choice_labels, example['choices'])])
            
            prompt_parts.extend([
                f"Example {i}:",
                f"Question: {example['question']}",
                f"Choices:",
                example_choices,
                f"Answer: {example['answer']}",
                ""
            ])
        
        # Add the actual question
        current_choices = '\n'.join([f"{label}. {choice}" 
                                   for label, choice in zip(choice_labels, choices)])
        
        prompt_parts.extend([
            "Now answer this question following the same format (single letter only):",
            f"Question: {question}",
            f"Choices:",
            current_choices,
            "Answer: [/INST]"
        ])
        
        return '\n'.join(prompt_parts)
    
    def extract_answer(self, response: str) -> str:
        """
        Extract the answer choice from the model response.
        Searches from the start and returns the first occurrence of A, B, C, or D.
        
        Args:
            response: Model's generated response
            
        Returns:
            Extracted answer choice ('A', 'B', 'C', 'D', or 'INVALID')
        """
        print(f"Raw model response: {response}")
        response = response.strip().upper()
        
        # Find the first occurrence of A, B, C, or D from the start
        valid_choices = ['A', 'B', 'C', 'D']
        first_choice = None
        first_position = len(response)  # Initialize with max possible position
        
        for choice in valid_choices:
            position = response.find(choice)
            if position != -1 and position < first_position:
                first_position = position
                first_choice = choice
        
        return first_choice if first_choice is not None else 'INVALID'
    
    def predict_single(self, question: str, choices: List[str], subject: str) -> Tuple[str, str]:
        """
        Make a prediction for a single question using 5-shot prompting.
        
        Args:
            question: The question text
            choices: List of answer choices
            subject: MMLU subject name
            
        Returns:
            Tuple of (predicted_answer, raw_response)
        """
        prompt = self.format_few_shot_prompt(question, choices, subject)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract answer
        answer = self.extract_answer(response)

        print(f"Predicted answer: {answer}")
        
        return answer, response.strip()
    
    def evaluate_subject(self, subject: str, max_samples: int = None, verbose: bool = False) -> Dict:
        """
        Evaluate on a specific MMLU subject using 5-shot prompting.
        
        Args:
            subject: MMLU subject name
            max_samples: Maximum number of samples to evaluate (for testing)
            verbose: Whether to print individual predictions
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating on subject: {subject} (5-shot)")
        
        # Load dataset for specific subject
        try:
            dataset = load_dataset("cais/mmlu", subject)['test']
        except:
            # Fallback to alternative dataset name format
            dataset = load_dataset("hendrycks_test", subject)['test']
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            
        correct = 0
        total = 0
        predictions = []
        
        # Load few-shot examples once for this subject
        few_shot_examples = self.load_few_shot_examples(subject)
        print(f"Using {len(few_shot_examples)} few-shot examples for {subject}")
        
        for example in tqdm(dataset, desc=f"Processing {subject}"):
            question = example['question']
            choices = example['choices']
            correct_answer = ['A', 'B', 'C', 'D'][example['answer']]

            print(f"\nQuestion: {question}")
            print(f"Choices: {choices}")
            print(f"Correct answer: {correct_answer}")
            
            predicted_answer, raw_response = self.predict_single(question, choices, subject)
            
            is_correct = predicted_answer == correct_answer
            if is_correct:
                correct += 1
            total += 1
            
            prediction_data = {
                'question': question,
                'choices': choices,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'raw_response': raw_response,
                'is_correct': is_correct
            }
            predictions.append(prediction_data)
            
            if verbose:
                status = "✓" if is_correct else "✗"
                print(f"{status} Q{total}: Predicted {predicted_answer}, Correct {correct_answer}")
                if not is_correct:
                    print(f"  Raw response: {raw_response}")
            
            # Memory cleanup
            if total % 20 == 0:
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'subject': subject,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'few_shot_examples': few_shot_examples,
            'predictions': predictions
        }
    
    def evaluate_all_subjects(self, max_samples_per_subject: int = None, verbose: bool = False) -> Dict:
        """
        Evaluate on all MMLU subjects using 5-shot prompting.
        
        Args:
            max_samples_per_subject: Maximum samples per subject (for testing)
            verbose: Whether to print individual predictions
            
        Returns:
            Dictionary with overall results
        """
        # MMLU subject list
        subjects = [
            'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
            'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
            'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
            'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
            'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
            'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
            'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
            'high_school_physics', 'high_school_psychology', 'high_school_statistics',
            'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
            'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning',
            'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes',
            'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
            'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations',
            'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
        ]
        
        all_results = []
        total_correct = 0
        total_questions = 0
        
        for subject in subjects:
            try:
                result = self.evaluate_subject(subject, max_samples_per_subject, verbose)
                all_results.append(result)
                total_correct += result['correct']
                total_questions += result['total']
                
                print(f"{subject}: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
                
            except Exception as e:
                print(f"Error evaluating {subject}: {str(e)}")
                continue
        
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
        
        return {
            'evaluation_type': '5-shot',
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_questions': total_questions,
            'subject_results': all_results
        }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Mistral 7B Instruct v0.3 on MMLU with 5-shot prompting")
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.3",
                       help="HuggingFace model name")
    parser.add_argument('--device', type=str, default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use for inference")
    parser.add_argument('--max_samples', type=int, default=None,
                       help="Maximum samples per subject (for testing)")
    parser.add_argument('--subject', type=str, default=None,
                       help="Evaluate on a specific subject only")
    parser.add_argument('--output_file', type=str, default="mmlu_5shot_results.json",
                       help="Output file for results")
    parser.add_argument('--verbose', action='store_true',
                       help="Print individual predictions")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MistralMMLU5ShotEvaluator(model_name=args.model_name, device=args.device)
    
    # Run evaluation
    if args.subject:
        # Evaluate single subject
        results = evaluator.evaluate_subject(args.subject, args.max_samples, args.verbose)
        print(f"\n5-Shot Results for {args.subject}:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Correct: {results['correct']}/{results['total']}")
        
        # Show few-shot examples used
        print(f"\nFew-shot examples used:")
        for i, example in enumerate(results['few_shot_examples'], 1):
            print(f"Example {i}: {example['question'][:60]}... -> {example['answer']}")
            
    else:
        # Evaluate all subjects
        results = evaluator.evaluate_all_subjects(args.max_samples, args.verbose)
        print(f"\nOverall 5-Shot Results:")
        print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
        print(f"Total Correct: {results['total_correct']}/{results['total_questions']}")
        
        # Print subject-wise results
        print(f"\nSubject-wise Results:")
        for result in results['subject_results']:
            print(f"{result['subject']}: {result['accuracy']:.4f}")
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()