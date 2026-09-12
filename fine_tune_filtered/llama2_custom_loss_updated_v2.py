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


new_tokens = [
    'psychologist,', 'poet,', 'photographer,', 'nurse,', 'software_engineer,',
    'comedian,', 'pastor,', 'architect,', 'chiropractor,', 'dentist,', 'model,',
    'interior_designer,', 'teacher,', 'accountant,', 'rapper,', 'yoga_teacher,',
    'paralegal,', 'surgeon,', 'painter,', 'composer,', 'dj,', 'personal_trainer,',
    'physician,', 'journalist,', 'dietitian,', 'filmmaker,', 'attorney,', 'professor,', 'Male', 'Female'
]

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
HfFolder.save_token(os.environ.get("HF_TOKEN"))
model_id = "meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map = {'': torch.cuda.current_device()}, trust_remote_code=True,)

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
    output_dir = "./results_llama",
    num_train_epochs = 3,
    fp16 = False,
    bf16 = False,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
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
import json
with open('possible_token_stream.json', "r") as f:
    token_dict = json.load(f)
def create_one_hot_vector(tokens, seq_len, vocab_size):
    one_hot_matrix = np.zeros((seq_len, vocab_size), dtype=np.float32)
    
    for i, token_list in enumerate(tokens):
        for token in token_list:
            one_hot_matrix[i][token] = 1
    
    return one_hot_matrix

one_hot_dict = {}
for key, token_lists in token_dict.items():
    one_hot_dict[key] = create_one_hot_vector(token_lists, 9, 32000)

#target_token_sets = np.load("token_sequences.npy", allow_pickle=True)


import torch
import torch.nn.functional as F

#tok_dict = {'prof_token': [1153, 24452, 261, 7047, 391, 12042, 652, 270, 272, 11665, 2706, 23187, 277, 287, 419, 25252, 7333, 1190, 424, 300, 6956, 14895, 3633, 1336, 2496, 13892, 713, 1098, 3018, 4940, 19915, 8910, 28107, 17739, 11643, 15703, 4824, 344, 343, 5595, 12251, 9821, 29918, 28891, 610, 29926, 10599, 744, 23014, 13290, 1904, 371, 18422, 4983, 25339, 18558], 'gend_token': [27208, 19361, 744], 'dilim': [29892]}
#one_hot = np.load('one_hot_labels_tok.npy')
loss = {'gender': [], 'prof': [], 'format': [], 'total': []}

#one_hot = torch.from_numpy(one_hot).cuda()
def custom_loss_function(outputs, labels, specific_token_ids=[27208, 19361, 744]):
    logits = outputs.logits
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    batch_size, seq_len, num_classes = logits.size()
    op_pos = []
    dilim_pos = []
    gend_end_pos = []
    
    gender_mask = torch.zeros_like(labels, dtype=torch.bool)
    prof_mask = torch.zeros_like(labels, dtype=torch.bool)

    for j in range(len(labels)):
        for i in range(len(labels[0])-1, -1, -1):
            if (labels[j][i] == 27208 or labels[j][i] == 744):
                 gend_end_pos.append(i)
                 pos = i
                 break
        if labels[j][pos] == 27208:
            gender_mask[j][pos] = True
            gender_mask[j][pos+1] = True
        elif labels[j][pos] == 744:
            gender_mask[j][pos] = True
            gender_mask[j][pos-1] = True
            gender_mask[j][pos+1] = True
            pos = pos-1
        
        for i in range(pos-1, -1, -1):
            if labels[j][i] != 29901:
                prof_mask[j][i] = True
            else:
                #op_pos.append(i+1)
                break

        for i in range(pos-1, -1, -1):
            if labels[j][i] == 29892 :
                dilim_pos.append(i)
                pos = i
                break
        for i in range(pos-1, -1, -1):
            if labels[j][i] == 29901 :
                op_pos.append(i+1)
                pos = i+1
                break
    

    #gender part
    modified_labels = labels.clone()
    modified_labels[~gender_mask] = -100 
     
    
    loss_gend = F.cross_entropy(logits.view(-1, num_classes), modified_labels.view(-1), ignore_index=-100, reduction='mean')


    #Profession part


    modified_labels_2 = labels.clone()
    modified_labels_2[~prof_mask] = -100
    
    loss_prof = F.cross_entropy(logits.view(-1, num_classes), modified_labels_2.view(-1), ignore_index=-100, reduction='mean')
    

    #format part
    extracted_logits = []
    one_hot_list = []

    for i in range(batch_size):
        start_idx = op_pos[i]
        end_idx = start_idx + 9

        logits_slice = logits[i, start_idx:end_idx, :]
        extracted_logits.append(logits_slice)

        label_key = int(labels[i][start_idx])  # Convert to int if label tensor is in a different format
        one_hot_matrix = torch.tensor(one_hot_dict[str(label_key)], dtype=torch.float32).cuda()  # Create one-hot matrix dynamically

        one_hot_list.append(one_hot_matrix)

    extracted_logits = torch.stack(extracted_logits)  # Shape: (batch_size, 9, num_classes)
    one_hot_tensor = torch.stack(one_hot_list)        # Shape: (batch_size, 9, num_classes)

    softmax_logits = torch.nn.functional.softmax(extracted_logits, dim=-1)

    masked_probs = softmax_logits * one_hot_tensor  # Now one_hot_tensor is dynamic per batch

    valid_prob_mass = masked_probs.sum(dim=-1)

    loss_format = -torch.log(valid_prob_mass + 1e-8).mean()



    lambda_1 = 0.4
    lambda_2 = 0.4
    lambda_3 = 0.2*100
    
    loss['lambda_1'] = lambda_1
    loss['lambda_2'] = lambda_2
    loss['lambda_3'] = lambda_3/100

    total_loss = lambda_1*loss_format + lambda_2*loss_prof + lambda_3*(1-torch.sigmoid(loss_gend))
    print(f'Total Loss = {total_loss.item()}, Format_loss = {(lambda_1*loss_format).item()}, Prof_loss = {(lambda_2*loss_prof).item()}, Gend_loss = {(lambda_3*(1-torch.sigmoid(loss_gend))).item()}')
    


    loss['gender'].append(lambda_3*loss_gend.item())
    loss['prof'].append(lambda_2*loss_prof.item())
    loss['format'].append(lambda_1*loss_format.item())
    loss['total'].append(total_loss.item())

    return total_loss





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
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=2048).to(self.device)
                
                outputs = self.model.generate(**inputs, max_new_tokens=8)
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                start_idx = generated_text.find('### Model Output:') + len('### Model Output:')
                extracted_text = generated_text[start_idx:].strip()
                

                #print(f"Input text: {input_text}")
                #modified_generated_text = generated_text.replace(input_text, "").strip()
                print(f"Generated text: {extracted_text}")
                print(f"Target text: {sample['target_text']}\n")
                print('-------------------------------------------------')

from transformers import Trainer
custom_callback = CustomCallback(model, tokenizer2, processed_dataset_dict['test'])
class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        labels = inputs.get("labels")
        #print(inputs.keys())
        #print(all(inputs['input_ids'][0]==inputs['labels'][0]))
        #print(inputs['labels'][0].tolist())
        loss = custom_loss_function(outputs, labels)
        return (loss, outputs) if return_outputs else loss


trainer = CustomSFTTrainer(
    model=model,
    train_dataset=processed_dataset_dict['train'],
    eval_dataset=processed_dataset_dict['validation'],
    peft_config=peft_config,
    #data_collator=collator,
    tokenizer=tokenizer,
    args=training_args,
    #callbacks=[custom_callback]
)

trainer.train()

import json
file_path = 'fine_tuned_llama2_test_custom_loss_r8_a64_tok_32000_new_format_loss_v2.json'

with open(file_path, 'w') as file:
    json.dump(loss, file, indent=4) 


trainer.model.save_pretrained('./fine_tuned_llama2_test_custom_loss_r8_a64_tok_32000_new_format_loss_v2')
tokenizer.save_pretrained('./fine_tuned_llama2_test_custom_loss_r8_a64_tok_32000_new_format_loss_v2')

val_results = trainer.evaluate(eval_dataset=processed_dataset_dict['validation'])
print(f"Validation Loss: {val_results['eval_loss']}")

# test_results = trainer.evaluate(eval_dataset=processed_dataset_dict['test'])
# print(f"Test Loss: {test_results['eval_loss']}")


log_history_df = pd.DataFrame(trainer.state.log_history)

# Save DataFrame to a CSV file
log_history_df.to_json('LOG____fine_tuned_llama2_test_custom_loss_r8_a64_tok_32000_new_format_loss_v2.json', orient='records', lines=True)