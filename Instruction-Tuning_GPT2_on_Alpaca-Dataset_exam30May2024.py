'''
Instruction Tuning GPT2 on Alpaca Dataset
Author: Sovit Ranjan Rath _ May 6, 2024
Practice: Mr. Jack _ May30, 2024
Link: https://debuggercafe.com/instruction-tuning-gpt2-on-alpaca-dataset/

Fine-tuning language models to follow instructions is a major step in making them more useful. In this article, we will train the GPT2 model for following simple instructions. Instruction tuning GPT2 on the Alpaca dataset will reveal how well very small language models perform at following instructions.

In particular, we will train the GPT2 base model which contains just 124 million parameters. This is much smaller than what the industry considers as SLMs (Small Language Models), which us typically 7 bllion (7B) parameters. In fact, any language model below 3 billion parameters can be a challenge to to train for instruction following. However, in future posts, we will train many such models and see how far we can push the envelope forward. This post is a starting point for this.
'''

import os, sys
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging,
)
from trl import SFTTrainer


# --------- Training and Dataset Configurations ---------
batch_size = 1 # 16
num_workers = os.cpu_count()
max_steps = 8 # 3000
bf16 = False
fp16 = False # True
gradient_accumulation_steps = 1 # 2
context_length = 256
logging_steps = 1 # 500
save_steps = 8 # 500
learning_rate = 0.0001
model_name = 'gpt2' # 'openai-community/gpt2'
out_dir = 'outputs/gpt2_alpaca_preprocess_fn'


# Loading the Alpaca Instruction Tuning Dataset
dataset = load_dataset('tatsu-lab/alpaca')
print(dataset)

'''
Downloading readme: 100%|█████████████████████████████████████████████████| 7.47k/7.47k [00:00<00:00, 5.57MB/s]
Downloading data: 100%|███████████████████████████████████████████████████| 24.2M/24.2M [00:04<00:00, 4.96MB/s]
Downloading data files: 100%|████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.08s/it]
Extracting data files: 100%|███████████████████████████████████████████████████| 1/1 [00:00<00:00, 1128.71it/s]
Generating train split: 52002 examples [00:00, 349609.93 examples/s]
DatasetDict({
    train: Dataset({
        features: ['instruction', 'input', 'output', 'text'],
        num_rows: 52002
    })
})
'''

# https://huggingface.co/docs/datasets/loading
dataset.save_to_disk('tatsu-lab-alpaca')
# Saving the dataset (1/1 shards): 100%|████████████████████████████████████████████████████| 52002/52002 [00:00<00:00, 156045.99 examples/s]

dataset = load_from_disk('tatsu-lab-alpaca')['train']
print("\n",dataset) # ~> num_rows: 52002

# https://huggingface.co/docs/datasets/process#shard
# dataset = dataset.shard(num_shards=1000, index=0)
# print("\n",dataset) # ~> num_rows: 53

dataset = dataset.select(range(50))
print("\n",dataset) # ~> num_rows: 50

# full_dataset = dataset['train'].train_test_split(test_size=0.05, shuffle=True)
full_dataset = dataset.train_test_split(test_size=0.2, shuffle=True)
dataset_train = full_dataset['train']
dataset_valid = full_dataset['test']
 
print(dataset_train)
print(dataset_valid)

# ~> dataset_train 40
# ~> dataset_valid 10


# --------- The Preprocessing Function ---------
def preprocess_function(example):
    """
    Formatting function returning a list of samples (kind of necessary for SFT API).
    """
    output_texts = []
    for i, exam in enumerate(example['instruction']):
	    text = f"### Instruction:\n{example['instruction'][i]}\n\n### Input:\n{example['input'][i]}\n\n### Response:\n{example['output'][i]}"
	    output_texts.append(text)
    return output_texts


# --------- Initializing the GPT2 Base Model for Instruction Tuning ---------
if bf16:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(dtype=torch.bfloat16)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name)
print(model)
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# 124,439,808 total parameters.
# 124,439,808 training parameters.


# --------- Initializing the Tokenizer ---------
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True,
    use_fast=False
)
# tokenizer.pad_token = tokenizer.eos_token # <|endoftext|>

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token # <|endoftext|>


# --------- Training the GPT2 Model on the Alpaca Dataset ---------
training_args = TrainingArguments(
    output_dir=f"{out_dir}/logs",
    evaluation_strategy='steps',
    eval_steps=save_steps,
    weight_decay=0.01,
    load_best_model_at_end=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_strategy='steps',
    save_strategy='steps',
    logging_steps=logging_steps,
    save_steps=save_steps,
    save_total_limit=2,
    # bf16=bf16,
    # fp16=fp16,
    # report_to='tensorboard',
    max_steps=max_steps,
    # dataloader_num_workers=num_workers,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    lr_scheduler_type='constant',
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_valid,
    max_seq_length=context_length,
    tokenizer=tokenizer,
    args=training_args,
    formatting_func=preprocess_function,
    # packing=True
)

dataloader = trainer.get_train_dataloader()
for i, sample in enumerate(dataloader):
    print(tokenizer.decode(sample['input_ids'][0]))
    print('#'*50)
    if i == 5:
        break

history = trainer.train()

model.save_pretrained(f"{out_dir}/best_model")
tokenizer.save_pretrained(f"{out_dir}/best_model")


# --------- Inference using the Instruction Tuned GPT2 Base Model ---------
# from transformers import (
#     AutoModelForCausalLM, 
#     logging, 
#     pipeline,
#     AutoTokenizer
# )
# import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained('outputs/gpt2_alpaca_preprocess_fn/best_model/')
tokenizer = AutoTokenizer.from_pretrained('outputs/gpt2_alpaca_preprocess_fn/best_model/')
tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    task='text-generation', 
    model=model, 
    tokenizer=tokenizer, 
    max_length=256, # Prompt + new tokens to generate.
    device_map=device
)

template = """### Instruction:
{}
### Input:
{}
### Response:
{}"""

instructions = 'Write three tips for staying healthy.'
inputs = ''
response = ''
prompt = template.format(instructions, inputs, response)

outputs = pipe(
    prompt, 
    do_sample=True, 
    temperature=0.7, 
    top_k=50, 
    top_p=0.95,
    repetition_penalty=1.1,
)
print(outputs[0]['generated_text'])

instructions = 'How can I become better at speaking?'
inputs = ''
response = ''
prompt = template.format(instructions, inputs, response)
outputs = pipe(
    prompt, 
    do_sample=True, 
    temperature=0.7, 
    top_k=50, 
    top_p=0.95,
    repetition_penalty=1.1,
)
print(outputs[0]['generated_text'])
