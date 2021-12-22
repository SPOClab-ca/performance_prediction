import argparse 
import transformers
import datasets
from datasets import load_dataset, load_metric 
from pathlib import Path 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader
import torch
import wandb
from utils import init_or_resume_wandb_run, timed_func


import tokenizers 
print("""Implementation note: This script depends a on some new features. 
Here are the versions of the huggingface tools I developed this script on.
If your version is smaller than these, there might be various bugs.""")
print("transformers version (developed on 4.14.1):", transformers.__version__) 
print("datasets version (developed on 1.16.1):", datasets.__version__)
print("tokenizers version (developed on 0.10.3):", tokenizers.__version__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 


def tokenize_function(examples, tokenizer, task):
    #  examples is a batch of 
    processed_examples = []
    if task in ["cola", "sst2"]:
        processed_examples = examples["sentence"]
    elif task in ["ax", "mnli"]:
        for p, h in zip(examples["premise"], examples["hypothesis"]):
            processed_examples.append(p + tokenizer.sep_token + h)
    elif task in ["mrpc", "rte", "stsb", "wnli"]:
        for s1, s2 in zip(examples["sentence1"], examples["sentence2"]):
            processed_examples.append(s1 + tokenizer.sep_token + s2) 
    elif task in ["qnli"]:
        for q, s in zip(examples["question"], examples["sentence"]):
            processed_examples.append(q + tokenizer.sep_token + s)    
    elif task in ["qqp"]:
        for s1, s2 in zip(examples["question2"], examples["question2"]):
            processed_examples.append(s1 + tokenizer.sep_token + s2)
    elif task in ["mnli_matched", "mnli_mismatched"]:
        raise ValueError("Use task=mnli instead. mnli_matched and mnli_mismatched only contains validation and test")
    else:
        raise ValueError("tokenization for task {} currently not supported".format(task)) 
    return tokenizer(processed_examples, padding="max_length", truncation=True)


@timed_func
def train_glue_task(args):
    raw_datasets = load_dataset("glue", args.task) 
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenize_function(examples, tokenizer, args.task), batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    #tokenized_datasets.set_format("torch")  # This causes problems; I'll manually set the format in iteration

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)
    model.to(device) 

    optimizer = AdamW(model.parameters(), lr=args.init_lr)
    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    model.train()
    for epoch in range(args.num_epochs):
        for b in train_dataloader:
            batch = {
                "input_ids": torch.stack(b["input_ids"]).transpose(0,1).to(device),
                "attention_mask": torch.stack(b["attention_mask"]).transpose(0,1).to(device),
                "labels": b["labels"].to(device)
            }
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        epoch_metric = evaluate(model, eval_dataloader)
        wandb.log({
            "epoch": epoch,
            "dev_acc": epoch_metric["accuracy"]
        })
        epoch_metric['epoch'] = epoch 


def evaluate(model, eval_dataloader):
    metric= load_metric("accuracy")
    model.eval()
    for b in eval_dataloader:
        batch = {
            "input_ids": torch.stack(b["input_ids"]).transpose(0,1).to(device),
            "attention_mask": torch.stack(b["attention_mask"]).transpose(0,1).to(device),
            "labels": b["labels"].to(device)
        }
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    return metric.compute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="cola")
    parser.add_argument("--model", type=str, default="roberta-base")
    parser.add_argument("--init_lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--slurm_id", type=str, default=0)
    parser.add_argument("--checkpoint_dir", type=str, default="")
    args = parser.parse_args()

    args.num_labels = {"mnli": 3,
        "mrpc": 2, "qnli": 2, "qqp": 2, "rte": 2, "sst2": 2,
        "cola": 2, "wnli": 2}[args.task]
    print(args)
    init_or_resume_wandb_run(
        wandb_id_file_path=Path(args.checkpoint_dir, "wandb_id.txt"),
        project_name="probing-shortcuts",
        config={
            "task": args.task,
            "dataset_seed": args.dataset_seed,
            "model": args.model,
            "slurm_id": args.slurm_id,
            "init_lr": args.init_lr,
            "batch_size": args.batch_size
        })

    computed_metrics = train_glue_task(args)
    print(computed_metrics)
