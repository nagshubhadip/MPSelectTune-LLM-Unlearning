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
from huggingface_hub.hf_api import HfFolder
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback, Trainer
import transformers
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, PeftModel

import numpy as np
#torch.cuda.set_device(1)
ICE_data = np.load('ice_data.npy', allow_pickle=True)
Train_data = np.load('train_data.npy', allow_pickle=True)
Val_data = np.load('val_data.npy', allow_pickle=True)
Test_data = np.load('test_data.npy', allow_pickle=True)

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
idx_train = pd.read_csv('train_idx_for_prompt_sel_from_ICE.csv')
idx_val = pd.read_csv('val_idx_for_prompt_sel_from_ICE.csv')
idx_test = pd.read_csv('test_idx_for_prompt_sel_from_ICE.csv')


base_template = """The list of possible professions are:
[ psychologist, poet, photographer, nurse, software_engineer, comedian, pastor, architect, chiropractor, dentist, model, interior_designer, teacher, accountant, rapper, yoga_teacher, paralegal, surgeon, painter, composer, dj, personal_trainer, physician, journalist, dietitian, filmmaker, attorney, professor,]

Examples:
"""

def put_example(bio, prof, gend, flip=0):
    if flip == 1:
        if gend == 'M': gender = 'Female'
        else: gender = 'Male'
    else:
        if gend == 'M': gender = 'Male'
        else: gender = 'Female'
    template = """
Bio: {b1}
Q: Predict the profession and the gender of the above bio
A: {p1}, {g1}
"""
    result_string = template.format(b1=bio, p1=prof, g1=gender)
    
    return result_string

def put_example_test(bio, prof, gend, flip=0):
    if flip == 1:
        if gend == 'M': gender = 'Female'
        else: gender = 'Male'
    else:
        if gend == 'M': gender = 'Male'
        else: gender = 'Female'
    template = """
Now, predict for this given example
Bio: {b1}
Q: Predict the profession and the gender of the above bio
 ### Model Output: {p1}, {g1}
"""
    result_string = template.format(b1=bio, p1=prof, g1=gender)
    
    return result_string

def put_example_test_for_test_data(bio):
    template = """
Now, predict for this given example
Bio: {b1}
Q: Predict the profession and the gender of the above bio
 ### Model Output: """
    result_string = template.format(b1=bio)
    
    return result_string


import ast
from datasets import Dataset, DatasetDict
def get_prompt(idx, data, Test=0):
    result = []
    temp = []
    target = []
    for index, row in idx.iterrows():
        temp_template = base_template
        indices_list = ast.literal_eval(row['indices'])
        for i in indices_list:
           example = put_example(ICE_data[i][3], ICE_data[i][1], ICE_data[i][2])
           temp_template = temp_template + example
        
        if Test == 0:
            temp_template = temp_template + put_example_test(data[index][3], data[index][1], data[index][2], flip=0)
        else:
            temp_template = temp_template + put_example_test_for_test_data(data[index][3])
        temp.append(temp_template)
        gender = 'Male' if data[index][2] == 'M' else 'Female'
        target.append((f'{data[index][1]}, {gender}</s>'))
        result.append(f"{row['num_samples']}, {row['type']}")

    return {'input_text': temp, 'target_text': target, 'combination': result}
            
            
            
train_data = get_prompt(idx_train, Train_data)
val_data = get_prompt(idx_val, Val_data)
test_data = get_prompt(idx_test, Test_data, Test=1)


train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
val_df = pd.DataFrame(val_data)

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)


#Convert DataFrames to Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Create DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})


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

#device_map = {"": "cuda:1"}

model_id = "meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map = {"": "cuda:0"}, trust_remote_code=True,)

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

import numpy as np
from datasets import Dataset, DatasetDict
import random
random.seed(42)
def preprocess_function(example, tokenizer):
    #tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.bos_token
    model_inputs = tokenizer(example['input_text'], max_length=2048, padding='max_length', truncation=True)
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
processed_val_dataset = preprocess_dataset_individual(dataset_dict['validation'])
processed_test_dataset = preprocess_dataset_individual(dataset_dict['test'])

processed_dataset_dict = DatasetDict({
    'train': processed_train_dataset,
    'validation': processed_val_dataset,
    'test': processed_test_dataset
})

training_args = TrainingArguments(
    output_dir = "./analysis/results_llama",
    num_train_epochs = 3,
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
    save_steps = 1000,  # Save checkpoint every 1000 steps
    logging_steps = 25,
    logging_dir='./logs',
    report_to='none'
)

import torch
import torch.nn.functional as F


template_ids = tokenizer.encode("\n ###Model Output:", add_special_tokens=False)[2:]
collator = DataCollatorForCompletionOnlyLM(template_ids, tokenizer=tokenizer)



import torch
import torch.nn.functional as F

one_hot = np.
