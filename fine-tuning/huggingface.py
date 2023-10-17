'''
import dependencies
'''
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from peft import LoraConfig
from trl import SFTTrainer

from auto_gptq import exllama_set_max_input_length

'''
Load model
'''
model_name = "TheBloke/Llama-2-70B-chat-GPTQ"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # device_map="auto",
    device_map={"": 0},
    trust_remote_code=False,
    revision="gptq-4bit-64g-actorder_True"
)
model = exllama_set_max_input_length(model, 8192) # Need to set when using LLaMa models
model.enable_input_require_grads() # Need to enable when training!

'''
Load dataset
'''
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

print('successfully loaded model and tokenzier')

lora_config = LoraConfig(
    # target_modules=["q_proj", "k_proj"],
    init_lora_weights=False,
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir='./train-output',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim='paged_adamw_32bit',
    save_steps=25,
    logging_steps=25,
    learning_rate=1e-3,
    weight_decay=1e-3,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type='constant',
    report_to="tensorboard"
)

'''
Load and wrap the dataset to fine-tune on
'''
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from datasets import Dataset
import json

template = yaml.safe_load(open('./templates/college_confidential.yaml'))
df = pd.read_csv('./data/college_confidential/dataset.csv')

def wrap_task(alt_a, alt_b, text, label):
    prompt = '### System:\n'
    prompt += template['instruction']
    prompt += '\n\n'
    prompt += '### User:\n'
    prompt += template['task'].replace(
        '{alternative_a}', alt_a
    ).replace(
        '{alternative_b}', alt_b
    ).replace(
        '{text}', text
    )
    prompt += '\n\n'
    prompt += '### Response:\n'
    prompt += template['label'][label]
    return prompt

input_prompts = [
    wrap_task(alt_a, alt_b, text, label)
    for alt_a, alt_b, text, label in df[['alternative_a', 'alternative_b','text','label']].values
]

X_train, X_test, y_train, y_test = train_test_split(
    input_prompts,
    df['label'],
    test_size = .2,
    random_state = 0,
)

dataset = Dataset.from_dict(dict(
    text = X_train
))

json.dump(X_train, open('./data/college_confidential/train.json','w'))
json.dump(X_test, open('./data/college_confidential/test.json','w'))

print('loaded dataset')

trainer = SFTTrainer(
    model = model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

trainer.train()
trainer.model.save_pretrained('ft-college-confidential')

