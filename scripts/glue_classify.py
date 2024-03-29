import argparse 
import transformers
import datasets
from datasets import load_dataset, load_metric 
from pathlib import Path 
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
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
    #  examples contains a batch of data.
    processed_examples = []
    if task in ["cola", "sst2"]:
        processed_examples = examples["sentence"]
    elif task in ["ax", "mnli_m", "mnli_mm"]:
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
    if tokenizer.model_max_length > 512:
        # xlnet-base-cased needs a manual specification of max_length; otherwise it doesn't pad (since its tokenizer.model_max_length is about 1e32)
        return tokenizer(processed_examples, max_length=512, padding="max_length", truncation=True)
    else:
        return tokenizer(processed_examples, padding="max_length", truncation=True)


def prepare_datasets_models(args):
    """
    Prepares everything needed for this experiment.
    Also reads from checkpoint if exists.
    """
    if args.task in ["mnli_m", "mnli_mm"]:
        raw_datasets = load_dataset("glue", "mnli")
    else:
        raw_datasets = load_dataset("glue", args.task) 
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenize_function(examples, tokenizer, args.task), batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    #tokenized_datasets.set_format("torch")  # This causes problems; I'll manually set the format in iteration

    train_dataset = tokenized_datasets["train"]
    if args.task == "mnli_m":
        eval_dataset = tokenized_datasets["validation_matched"]
        test_dataset = tokenized_datasets["test_matched"]
    elif args.task == "mnli_mm":
        eval_dataset = tokenized_datasets["validation_mismatched"]
        test_dataset = tokenized_datasets["test_mismatched"]
    else:
        eval_dataset = tokenized_datasets["validation"]
        test_dataset = tokenized_datasets["test"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    #test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)  # Not used
    if args.corruption_step > 0:
        model = AutoModelForSequenceClassification.from_pretrained(
            "../data/corrupted_{}_checkpoints/checkpoint-{}".format(
                args.model.replace("-", "_").replace("/", "_"), 
                args.corruption_step))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)
    model.to(device) 
    
    optimizer = AdamW(model.parameters(), lr=args.init_lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.num_epochs * len(train_dataloader)
    )

    train_dataloader_start_batch_idx = 0
    start_epoch = 0

    checkpoint_path = Path(args.checkpoint_dir, "checkpoint.pt") 
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["train_start_epoch"]
        train_dataloader_start_batch_idx = ckpt["train_dataloader_batch_idx"]
        print("Loaded from checkpoint")
    else:    
        print("Prepared experiment material from scratch")
    model.train()
    return train_dataloader, eval_dataloader, model, optimizer, lr_scheduler, start_epoch, train_dataloader_start_batch_idx


@timed_func
def train_glue_task(args):
    train_dataloader, eval_dataloader, model, optimizer, lr_scheduler, start_epoch, train_dataloader_start_batch_idx = prepare_datasets_models(args)
    last_ckpt_time = time.time()

    for epoch in range(start_epoch, args.num_epochs):
        for batch_idx, b in enumerate(train_dataloader):
            if batch_idx < train_dataloader_start_batch_idx:
                # Skip this batch
                continue 
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

            # Checkpoint at the end of this batch
            if time.time() - last_ckpt_time > 600:  
                torch.save({
                     "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "scheduler_state_dict": lr_scheduler.state_dict(),
                     "train_start_epoch": epoch,
                     "train_dataloader_batch_idx": batch_idx,
                }, Path(args.checkpoint_dir, "checkpoint.pt"))
                last_ckpt_time = time.time()

        epoch_metric = evaluate(model, eval_dataloader)
        wandb.log({
            "epoch": epoch,
            "dev_acc": epoch_metric["accuracy"]
        })
        epoch_metric['epoch'] = epoch 
        train_dataloader_start_batch_idx = 0

def evaluate(model, eval_dataloader):
    metric = load_metric("accuracy")
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
    parser.add_argument("--corruption_step", type=int, default=0)
    parser.add_argument("--init_lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dataset_seed", type=int, default=42)
    parser.add_argument("--slurm_id", type=str, default=0)
    parser.add_argument("--checkpoint_dir", type=str, default="")
    args = parser.parse_args()

    args.num_labels = {"mnli_m": 3, "mnli_mm": 3,
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
            "corruption_step": args.corruption_step,
            "slurm_id": args.slurm_id,
            "init_lr": args.init_lr,
            "batch_size": args.batch_size
        })

    torch.manual_seed(args.dataset_seed)
    computed_metrics = train_glue_task(args)
    print(computed_metrics)
