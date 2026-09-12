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
Bio: {b1}
Q: Predict the profession and the gender of the above bio
### Model Output: {p1}, {g1}
"""
    result_string = template.format(b1=bio, p1=prof, g1=gender)
    
    return result_string

def put_example_test_for_test_data(bio):
    template = """
Bio: {b1}
Q: Predict the profession and the gender of the above bio
### Model Output: 
"""
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
        
        #add here according to soham da
        if Test == 0:
            temp_template = temp_template + put_example_test(data[index][3], data[index][1], data[index][2], flip=1)
        else:
            temp_template = temp_template + put_example_test_for_test_data(data[index][3])
        temp.append(temp_template)
        gender = 'Male' if data[index][2] == 'M' else 'Female'
        target.append((f'{data[index][1]}, {gender}'))
        result.append(f"{row['num_samples']}, {row['type']}")

    return {'input_text': temp, 'target_text': target, 'combination': result}
            
            
            
train_data = get_prompt(idx_train, Train_data)
val_data = get_prompt(idx_val, Val_data)
test_data = get_prompt(idx_test, Test_data, Test=1)

#------Flipping 50%

# import ast
# from datasets import Dataset, DatasetDict
# def get_prompt(idx, data, Indices=[], Test=0):
#     result = []
#     temp = []
#     target = []
#     flip = 0
#     c = 0
#     for index, row in idx.iterrows():
#         if(index in Indices and Test == 0):
#             flip = 1
#             c+=1
#         else:
#             flip = 0
#         temp_template = base_template
#         indices_list = ast.literal_eval(row['indices'])
#         for i in indices_list:
#             example = put_example(ICE_data[i][3], ICE_data[i][1], ICE_data[i][2])
#             temp_template = temp_template + example
        
#         if Test == 0:
#             temp_template = temp_template + put_example_test(data[index][3], data[index][1], data[index][2], flip=flip)
#         else:
#             temp_template = temp_template + put_example_test_for_test_data(data[index][3])
#         # if flip ==1:
#         #     print(temp_template)
#         #     print(x)
#         temp.append(temp_template)
#         gender = 'Male' if data[index][2] == 'M' else 'Female'
#         target.append((f'{data[index][1]}, {gender}'))
#         result.append(f"{row['num_samples']}, {row['type']}")

#     return {'input_text': temp, 'target_text': target, 'combination': result}
            
# np.random.seed(42)
# #indices_to_flip = np.random.choice(len(Train_data), int(len(Train_data)*0.5), replace=False)                      
# train_data = get_prompt(idx_train, Train_data, Indices=indices_to_flip)

# indices_to_flip = np.random.choice(len(Val_data), int(len(Val_data)*0.5), replace=False)                      
# val_data = get_prompt(idx_val, Val_data, Indices=indices_to_flip)

# test_data = get_prompt(idx_test, Test_data, Test=1)

#------------------------------------------------------------------------

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
# new_tokens = [
#     ' psychologist,', ' poet,', ' photographer,', ' nurse,', ' software_engineer,',
#     ' comedian,', ' pastor,', ' architect,', ' chiropractor,', ' dentist,', ' model,',
#     ' interior_designer,', ' teacher,', ' accountant,', ' rapper,', ' yoga_teacher,',
#     ' paralegal,', ' surgeon,', ' painter,', ' composer,', ' dj,', ' personal_trainer,',
#     ' physician,', ' journalist,', ' dietitian,', ' filmmaker,', ' attorney,', ' professor,', ' Male', ' Female', ', '
# ]


model_id = "meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0}, trust_remote_code=True,)

model.config.use_cache = True # silence the warnings
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()


tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
#tokenizer.add_tokens(new_tokens)
#model.resize_token_embeddings(len(tokenizer))
model.enable_input_require_grads()


model = get_peft_model(model, peft_config).cuda()
#model = PeftModel.from_pretrained(model, 'fine_tuned_llama2_test').cuda()



def preprocess_function(examples):
    inputs = examples['input_text']
    targets = examples['target_text']
    model_inputs = tokenizer(inputs, max_length=2048, padding='max_length', truncation=True)
    labels = tokenizer(targets, max_length=8, padding='max_length', truncation=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)



# Set training arguments
training_args = TrainingArguments(
    output_dir = "./results_llama",
    num_train_epochs = 4,
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
    logging_steps = 25,
    logging_dir='./logs',
    report_to='none'
)





class CustomCallback(TrainerCallback):
    def __init__(self, model, tokenizer, test_dataset, new_tokens):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.new_token_ids = [self.tokenizer.encode(token, add_special_tokens=False)[0] for token in new_tokens]

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
                #print(inputs['input_ids'].shape)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=8, output_scores=True, return_dict_in_generate=True)
                
                token_ids = outputs.sequences[0]
                scores = outputs.scores  # This contains the logits for each step
                generated_tokens = [self.tokenizer.decode(token_id) for token_id in token_ids[-2:]]
                print(generated_tokens)
                # Find the most probable new token for each generated token
                for i, (token_id, logits) in enumerate(zip(token_ids[-8:], scores)):
                    print(token_id, logits)
                    
                    
                # Print the full generated text
                generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                start_idx = generated_text.find('##Model Output:') + len('##Model Output:')
                extracted_text = generated_text[start_idx:].strip()
                
                print(f"Generated text: {extracted_text}")
                print(f"Target text: {sample['target_text']}\n")
                print('-------------------------------------------------')


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

template_ids = tokenizer.encode("\n### Model Output:", add_special_tokens=False)[2:]
collator = DataCollatorForCompletionOnlyLM(template_ids, tokenizer=tokenizer)



custom_callback = CustomCallback(model, tokenizer, tokenized_datasets['test'], new_tokens)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],  # Validation dataset
    peft_config=peft_config,
    data_collator=collator,
    tokenizer=tokenizer,
    args=training_args,
    #callbacks=[loss_logger]
)

trainer.train()

val_results = trainer.evaluate(eval_dataset=tokenized_datasets['validation'])
print(f"Validation Loss: {val_results['eval_loss']}")

test_results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
print(f"Test Loss: {test_results['eval_loss']}")

trainer.model.save_pretrained('./fine_tuned_llama2_test_v2')
tokenizer.save_pretrained('./fine_tuned_llama2_test_v2')

