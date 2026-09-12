import numpy as np
import os
import random
import pandas as pd
from datasets import Dataset, DatasetDict, load_metric
import numpy as np
from huggingface_hub import login
import warnings
from transformers import BitsAndBytesConfig
import torch
from huggingface_hub.hf_api import HfFolder
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback, Trainer
import transformers
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, PeftModel

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import numpy as np
import pandas as pd
import random
from itertools import chain
from sentence_transformers import SentenceTransformer, util


random.seed(42)
np.random.seed(42)

# Load the CSV file
file_path = 'annotations.csv'  
df = pd.read_csv(file_path)
df = df.dropna()

data_as_list = df.values.tolist()

final_data = []
for data in data_as_list:
    if data[1] == 'W':
        gender = 'Female'
    else:
        gender = 'Male'
    final_data.append(["Post_text: "+data[2] + "\nResponse_text: "+data[3], data[4], gender])

formatted_data = np.array(final_data)

np.random.shuffle(formatted_data)

train_data = formatted_data[:6000]
ice_data = formatted_data[6000:12000]
test_data = formatted_data[12000:13200]



model = SentenceTransformer('bert-base-nli-mean-tokens')

def extract_questions(data):
    return [item[0] for item in data]

train_text = extract_questions(train_data)
in_context_text = extract_questions(ice_data)

train_embeddings = model.encode(train_text, convert_to_tensor=True)
in_context_embeddings = model.encode(in_context_text, convert_to_tensor=True)

similarity_matrix = util.cos_sim(train_embeddings, in_context_embeddings)

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

num_training = len(train_text)
groups = [list(range(num_training))[i:i + 500] for i in range(0, num_training, 500)]

final_indices = []
combination_types = []


for group, (comb_type, num_examples, counts) in zip(groups, combinations):
    for idx in group:
        similarities = similarity_matrix[idx]
        sorted_indices = similarities.argsort(descending=True)
        most_similar = sorted_indices[:counts["similar"]].tolist()
        most_dissimilar = sorted_indices[-counts["dissimilar"]:].tolist() if counts["dissimilar"] > 0 else []
        random_indices = random.sample(range(len(in_context_text)), counts["random"]) if counts["random"] > 0 else []
        selected_indices = list(chain(most_similar, most_dissimilar, random_indices))
        random.shuffle(selected_indices)
        final_indices.append(selected_indices)
        combination_types.append(comb_type)

merged_list_train = list(zip(final_indices, combination_types))



# Function to process test data for a dataset
def process_test_data(test_data, in_context_data, similarity_matrix, combinations):
    num_test_examples = len(test_data)
    group_size = num_test_examples // len(combinations)  # 22 examples per combination
    print(group_size)
    groups = [list(range(num_test_examples))[i:i + group_size] for i in range(0, num_test_examples, group_size)]

    final_indices = []
    combination_types = []

    for group, (comb_type, num_examples, counts) in zip(groups, combinations):
        for idx in group:
            similarities = similarity_matrix[idx]
            sorted_indices = similarities.argsort(descending=True)
            most_similar = sorted_indices[:counts["similar"]].tolist()
            most_dissimilar = sorted_indices[-counts["dissimilar"]:].tolist() if counts["dissimilar"] > 0 else []
            random_indices = random.sample(range(len(in_context_data)), counts["random"]) if counts["random"] > 0 else []
            selected_indices = list(chain(most_similar, most_dissimilar, random_indices))
            random.shuffle(selected_indices)
            final_indices.append(selected_indices)
            combination_types.append(comb_type)

    return final_indices, combination_types

test_text = extract_questions(test_data)
test_embeddings = model.encode(test_text, convert_to_tensor=True)
similarity_matrix_test = util.cos_sim(test_embeddings, in_context_embeddings)

final_indices_test, combination_types_test = process_test_data(
    test_text, in_context_text, similarity_matrix_test, combinations
)

# Merged list for test data in both datasets
merged_list_test = list(zip(final_indices_test, combination_types_test))



import numpy as np

import warnings
warnings.filterwarnings('ignore')



base_template = """Instruction:
You are an expert assistant trained to jointly predict the sentiment and the gender for the given input from social media post and its response. 

Possible types of sentiment are: 'Mixed', 'Negative', 'Neutral', and 'Positive'.  
Possible types of gender are: 'Male' and 'Female'.  

Output Format:
The output should be in the format: 'sentiment, gender'.

Examples:
"""

def put_example(data):
    
    template = """
{post}
Q: Predict the sentiment and the gender of the above post and response in the format sentiment, gender.
Answer: {sent}, {gend}
"""
    result_string = template.format(post=data[0], sent=data[1], gend=data[2])
    
    return result_string

def put_example_test(data):
    
    template = """
Now, solve for this example:
{post}
Q: Predict the sentiment and the gender of the above post and response in the format sentiment, gender.
Model Answer: {sent}, {gend}
"""
    result_string = template.format(post=data[0], sent=data[1], gend=data[2])
    
    return result_string

def put_example_test_for_test_data(data):
    template = """
Now, solve for this example:
{post}
Q: Predict the sentiment and the gender of the above post and response in the format sentiment, gender.
Model Answer: """
    result_string = template.format(post=data[0])
    
    return result_string





import ast
from datasets import Dataset, DatasetDict
def get_prompt(idx, data, Test=0):
    result = []
    temp = []
    target = []
    for index, id in enumerate(idx):
        temp_template = base_template
        indices_list = id[0]
        for i in range(len(indices_list)):
           example = put_example(ice_data[indices_list[i]])
           temp_template = temp_template + example
        
        if Test == 0:
            temp_template = temp_template + put_example_test(data[index])
        else:
            temp_template = temp_template + put_example_test_for_test_data(data[index])
        temp.append(temp_template)
        
        target.append((f'{data[index][1]}, {data[index][2]}'))
        result.append(f"{id[1]}")

    return {'input_text': temp, 'target_text': target, 'combination': result}
            
            
            
train_data = get_prompt(merged_list_train, train_data)
test_data = get_prompt(merged_list_test, test_data, Test=1)


train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)


#Convert DataFrames to Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Create DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})


from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub.hf_api import HfFolder
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

peft_config = LoraConfig(
    r= 8,          
    lora_alpha= 64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


HfFolder.save_token(os.environ.get("HF_TOKEN"))
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0}, trust_remote_code=True,)

model.config.use_cache = True # silence the warnings
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()


tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


tokenizer2 = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer2.pad_token = tokenizer.eos_token
tokenizer2.padding_side = "left"
#tokenizer.add_tokens(new_tokens)
#model.resize_token_embeddings(len(tokenizer))
model.enable_input_require_grads()


model = get_peft_model(model, peft_config).cuda()
#model = PeftModel.from_pretrained(model, 'fine_tuned_llama2_test').cuda()



def preprocess_function(example, tokenizer):
    #tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.bos_token
    model_inputs = tokenizer(example['input_text'], max_length=1024, padding='max_length', truncation=True)
    #tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    labels = tokenizer(example['target_text'], max_length=10, truncation=True, padding="max_length")["input_ids"]
    labels = [(label if label != tokenizer.pad_token_id else -100) for label in labels]
       
    
    
    model_inputs['labels'] = labels
    model_inputs['combination'] = example['combination']
    model_inputs['input_text'] = example['input_text']
    model_inputs['target_text'] = example['target_text']
    return model_inputs

def preprocess_dataset_individual(dataset):
    # Extract the examples
    examples = dataset.to_pandas().to_dict(orient='records')  # Convert to list of dicts

    # Process each example
    processed_examples = []
    for example in examples:
        processed_example = preprocess_function(example, tokenizer)
        processed_examples.append(processed_example)

    # Convert the processed examples back to a dataset
    processed_df = pd.DataFrame(processed_examples)
    return Dataset.from_pandas(processed_df)

# Process each split individually
processed_train_dataset = preprocess_dataset_individual(dataset_dict['train'])
processed_test_dataset = preprocess_dataset_individual(dataset_dict['test'])

processed_dataset_dict = DatasetDict({
    'train': processed_train_dataset,
    'test': processed_test_dataset
})



training_args = TrainingArguments(
    output_dir = "./results_llama",
    num_train_epochs = 10,
    fp16 = False,
    bf16 = False,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    #gradient_accumulation_steps = 25,
    gradient_checkpointing = True,
    max_grad_norm = 0.3,
    learning_rate = 2e-4,
    weight_decay = 0.001,
    optim = "paged_adamw_32bit",
    lr_scheduler_type = "cosine",
    max_steps = -1,
    warmup_ratio = 0.03,
    group_by_length = True,
    save_steps = 0,
    logging_steps = 20,
    logging_dir='./logs',
    report_to='none'
)

class CustomCallback(TrainerCallback):
    def __init__(self, model, tokenizer, test_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % args.logging_steps == 0:
            if logs:
                print(f"Step: {state.global_step}")
                for key, value in logs.items():
                    print(f"{key}: {value}")
                
                # Select a random input text from the training data
                sample = random.choice(self.test_dataset)
                
                input_text = sample['input_text']
                # Tokenize and generate prediction
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024).to(self.device)
                
                outputs = self.model.generate(**inputs, max_new_tokens=5)
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                start_idx = generated_text.find('Model Answer:') + len('Model Answer:')
                extracted_text = generated_text[start_idx:].strip()
                

                #print(f"Input text: {input_text}")
                #modified_generated_text = generated_text.replace(input_text, "").strip()
                print(f"Generated text: {extracted_text}")
                print(f"Target text: {sample['target_text']}\n")
                print('-------------------------------------------------')

from transformers import Trainer
custom_callback = CustomCallback(model, tokenizer2, processed_dataset_dict['test'])

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

template_ids = tokenizer.encode("Model Answer:", add_special_tokens=False)[2:]
collator = DataCollatorForCompletionOnlyLM(template_ids, tokenizer=tokenizer)




trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset_dict['train'],
    eval_dataset=processed_dataset_dict['test'],  # Validation dataset
    peft_config=peft_config,
    #data_collator=collator,
    tokenizer=tokenizer,
    args=training_args,
    callbacks=[custom_callback]
)

trainer.train()


trainer.model.save_pretrained('./fine_tuned_llama3_rtgen_10epoch')
tokenizer.save_pretrained('./fine_tuned_llama3_rtgen_10epoch')

test_results = trainer.evaluate(eval_dataset=processed_dataset_dict['test'])
print(f"Test Loss: {test_results['eval_loss']}")
