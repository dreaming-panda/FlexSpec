from transformers import LlamaConfig, DataCollatorForLanguageModeling, AutoTokenizer, Trainer, TrainingArguments, LlamaForCausalLM
import argparse
import wandb
import os

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="Llama3"
# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="false"
# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"
from datasets import load_dataset,concatenate_datasets
def convert_dataset_train(tokenizer, seq_len = 1024):
    dataset = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.0001*-of-01024.json.gz'}, split='train')
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text', 'timestamp', 'url'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

def convert_dataset_eval(tokenizer, seq_len = 1024):
    dataset = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text', 'timestamp', 'url'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset
def convert_dataset_train_forcode(tokenizer, seq_len = 1024):
    dataset = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.0002*-of-01024.json.gz'}, split='train')
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text', 'timestamp', 'url'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    code = load_dataset('jtatman/python-code-dataset-500k', split='train[:500000]')
    def tokenize_function_code(examples):
            return tokenizer(examples["output"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    code = code.map(tokenize_function_code, batched=True, remove_columns=['output', 'instruction', 'system'])
    code.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataset = concatenate_datasets([dataset, code])
    return dataset

def convert_dataset_eval_forcode(tokenizer, seq_len = 1024):
    dataset = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00002-of-00008.json.gz'}, split='validation')
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text', 'timestamp', 'url'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    code = load_dataset('jtatman/python-code-dataset-500k', split='train[500000:]')
    def tokenize_function_code(examples):
            return tokenizer(examples["output"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    code = code.map(tokenize_function_code, batched=True, remove_columns=['output', 'instruction', 'system'])
    code.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataset = concatenate_datasets([dataset, code])
    return dataset

def convert_dataset_train_comprehensive(tokenizer, seq_len = 1024):
    dataset1 = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.0003*-of-01024.json.gz'}, split='train')
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset1 = dataset1.map(tokenize_function, batched=True, remove_columns=['text', 'timestamp', 'url'])
    dataset1.set_format(type='torch', columns=['input_ids', 'attention_mask'])


    dataset2 = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.0002*-of-01024.json.gz'}, split='train')
    dataset2 = dataset2.map(tokenize_function, batched=True, remove_columns=['text', 'timestamp', 'url'])
    dataset2.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    dataset3 = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.0001*-of-01024.json.gz'}, split='train')
    dataset3 = dataset3.map(tokenize_function, batched=True, remove_columns=['text', 'timestamp', 'url'])
    dataset3.set_format(type='torch', columns=['input_ids', 'attention_mask'])


    code0 = load_dataset('jtatman/python-code-dataset-500k', split='train[:500000]')
    def tokenize_function_code0(examples):
            return tokenizer(examples["output"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    code0 = code0.map(tokenize_function_code0, batched=True, remove_columns=['output', 'instruction', 'system'])
    code0.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    code = load_dataset('semeru/code-text-python', split='train')
    def tokenize_function_code(examples):
            return tokenizer(examples["original_string"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    code = code.map(tokenize_function_code, batched=True)
    code.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    math = load_dataset('peiyi9979/Math-Shepherd', split='train[:420000]')
    def tokenize_function_math(examples):
            return tokenizer(examples["label"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    math = math.map(tokenize_function_math, batched=True, remove_columns=['input', 'label'])
    math.set_format(type='torch', columns=['input_ids', 'attention_mask'])


    paper = load_dataset('CShorten/ML-ArXiv-Papers', split='train[:100000]')
    def tokenize_function_paper(examples):
            return tokenizer(examples["abstract"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    paper = paper.map(tokenize_function_paper, batched=True, remove_columns=['title', 'abstract', 'Unnamed: 0.1', 'Unnamed: 0'])
    paper.set_format(type='torch', columns=['input_ids', 'attention_mask'])


    poems = load_dataset('othorizedshogun/poems_dataset', split='train')
    def tokenize_function_poem(examples):
        return tokenizer(examples["input"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    poems = poems.map(tokenize_function_poem, batched=True, remove_columns=['poem', 'input', 'form', 'topic'])
    poems.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    dataset = concatenate_datasets([dataset1, dataset2, dataset3, code0, code, math, paper, poems])
    return dataset

def convert_dataset_eval_comprehensive(tokenizer, seq_len = 1024):
    dataset = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00003-of-00008.json.gz'}, split='validation')
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text', 'timestamp', 'url'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    code0 = load_dataset('jtatman/python-code-dataset-500k', split='train[500000:]')
    def tokenize_function_code0(examples):
            return tokenizer(examples["output"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    code0 = code0.map(tokenize_function_code0, batched=True, remove_columns=['output', 'instruction', 'system'])
    code0.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    code = load_dataset('semeru/code-text-python', split='validation')
    def tokenize_function_code(examples):
            return tokenizer(examples["original_string"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    code = code.map(tokenize_function_code, batched=True)
    code.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    math = load_dataset('peiyi9979/Math-Shepherd', split='train[420000:]')
    def tokenize_function_math(examples):
            return tokenizer(examples["label"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    math = math.map(tokenize_function_math, batched=True, remove_columns=['input', 'label'])
    math.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    paper = load_dataset('CShorten/ML-ArXiv-Papers', split='train[100000:]')
    def tokenize_function_paper(examples):
            return tokenizer(examples["abstract"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    paper = paper.map(tokenize_function_paper, batched=True, remove_columns=['title', 'abstract', 'Unnamed: 0.1', 'Unnamed: 0'])
    paper.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    poems = load_dataset('othorizedshogun/poems_dataset', split='test')
    def tokenize_function_poem(examples):
        return tokenizer(examples["input"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    poems = poems.map(tokenize_function_poem, batched=True, remove_columns=['poem', 'input', 'form', 'topic'])
    poems.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    dataset = concatenate_datasets([dataset, code0, code, math, paper, poems])
    return dataset

def train(args):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
            num_hidden_layers=args.L, hidden_size=args.S, intermediate_size=args.D,
            num_attention_heads=args.H, num_key_value_heads=args.H)
    config.pad_token_id = config.eos_token_id

    tokenized_dataset_train = convert_dataset_train_comprehensive(tokenizer=tokenizer)
    tokenized_dataset_eval = convert_dataset_eval_comprehensive(tokenizer=tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


    model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", config=config, ignore_mismatched_sizes=True)

    args = TrainingArguments(
    output_dir="Llama3-L{}-S{}-H{}-D{}".format(args.L, args.S, args.H, args.D),
    logging_dir="Llama3-L{}-S{}-H{}-D{}".format(args.L, args.S, args.H, args.D),
    per_device_train_batch_size=args.B,
    per_device_eval_batch_size=args.B,
    evaluation_strategy="steps",
    logging_steps=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=1e-4,
    bf16=True,
    push_to_hub=False,
    report_to="wandb",
    save_strategy='steps',
    log_level='info',
    load_best_model_at_end=True,
    eval_steps=2000,
    save_steps=2000,
    save_total_limit=3,
    )
    trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_eval
    )
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, default=4, help='hidden layers')
    parser.add_argument('--S', type=int, default=1024, help='hidden size')
    parser.add_argument('--H', type=int, default=16, help='attention heads')
    parser.add_argument('--D', type=int, default=3072, help='mlp size')
    parser.add_argument('--B', type=int, default=4, help='batch size')
    args = parser.parse_args()
    train(args)
