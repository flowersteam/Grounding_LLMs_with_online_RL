import transformers
import datasets
from transformers import AdamW, get_scheduler, set_seed, AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator
from datasets import Dataset, DatasetDict

from accelerate import Accelerator

accelerator = Accelerator(split_batches=False)

from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch
import numpy as np
import logging
import argparse
from copy import deepcopy
import os


def load_dataset(dir, file_name, file_id):
    _inputs = np.load(f"{dir}/{file_name}_prompts_{file_id}.npy")
    _outputs = np.load(f"{dir}/{file_name}_actions_{file_id}.npy")

    _train_dataset = Dataset.from_dict({
        "input": _inputs[:400000],
        "output": _outputs[:400000]
    })

    _eval_dataset = Dataset.from_dict({
        "input": _inputs[400000:],
        "output": _outputs[400000:]
    })

    return DatasetDict({
        "train": _train_dataset,
        "test": _eval_dataset
    })

def tokenize_dataset(dataset, tokenizer):
    tokenized_datasets = dataset.map(
        lambda examples: tokenizer(examples["input"], padding="max_length", max_length=1024),
        batched=True,
        desc="Running tokenizer on inputs",
        remove_columns=["input"]
    )

    # max_length = 3 as longest sequence is [<pad>, <turn>, <left>] (same with "turn right" or "go forward")
    tokenized_datasets = tokenized_datasets.map(
        lambda examples: {"labels": tokenizer(examples["output"], padding="max_length", max_length=3)["input_ids"]},
        batched=True,
        desc="Running tokenizer on outputs",
        remove_columns=["output"]
    )

    return tokenized_datasets


def setup_logging(logging_folder, args):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    if accelerator.is_main_process:  # we only want to setup logging once
        tb_writer = SummaryWriter(log_dir=logging_folder)
        hyperparams = deepcopy(args)
        for hyperparam, value in hyperparams.items():
            if isinstance(value, list):
                hyperparams[hyperparam] = ','.join(str(value))
        tb_writer.add_hparams(hyperparams, {'0': 0})
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        tb_writer = None
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger, tb_writer


def get_grouped_params(model, config, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [{'params': params_with_wd, 'weight_decay': config["weight_decay"]},
            {'params': params_without_wd, 'weight_decay': 0.0}]


def log_metrics(logger, tb_writer, step, metrics):
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        [tb_writer.add_scalar(k, v, step) for k, v in metrics.items()]


def evaluate(model, eval_dataloader, config):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss.repeat(config["per_device_batch_size"])
        losses.append(accelerator.gather(loss))
        if config["max_eval_steps"] > 0 and step >= config["max_eval_steps"]: break
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


def launch_training(args):
    torch.cuda.set_device(accelerator.device)
    raw_datasets = load_dataset(args.data_dir, args.file_name, args.file_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    processed_datasets = tokenize_dataset(raw_datasets, tokenizer)

    config = {
        "weight_decay": 0.0,
        "learning_rate": 5e-4,  # same as Flan paper
        "lr_scheduler_type": "cosine",
        "n_epochs": 1,
        "evaluation_steps": 250,
        "gradient_accumulation_steps": args.gradient_accumulation_steps
    }

    config["per_device_batch_size"] = args.per_device_batch_size
    config["full_batch_size"] = args.per_device_batch_size * accelerator.num_processes
    updates_batch_size = config["full_batch_size"] * args.gradient_accumulation_steps
    # Use the same number of samples for evaluation than for updates
    config["max_eval_steps"] = args.per_device_batch_size * args.gradient_accumulation_steps
    config["num_warmup_steps"] = len(processed_datasets["train"]) // updates_batch_size * 0.01  # => 1% of total number of steps

    output_dir = args.output_dir
    # Sanity checks
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    logger, tb_writer = setup_logging(output_dir + "/logs/", config)

    set_seed(args.seed)

    train_dataloader = DataLoader(processed_datasets["train"], collate_fn=default_data_collator,
                                  batch_size=config["per_device_batch_size"])
    eval_dataloader = DataLoader(processed_datasets["test"], collate_fn=default_data_collator,
                                 batch_size=config["per_device_batch_size"])
    n_train_steps = len(processed_datasets["train"]) / updates_batch_size * config["n_epochs"]

    # Prepare the optimizer and learning rate scheduler
    optimizer = AdamW(get_grouped_params(model, config), lr=config["learning_rate"], eps=1e-8)
    lr_scheduler = get_scheduler(name=config["lr_scheduler_type"], optimizer=optimizer,
                                 num_warmup_steps=config["num_warmup_steps"],
                                 num_training_steps=n_train_steps)

    def get_lr():
        return optimizer.param_groups[0]['lr']

    # Prepare everything with our `accelerator`.
    logger.info("Accelerate preparing...")
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader)

    # Train model
    logger.info("Training model!")
    model.train()
    completed_steps = 0
    for epoch in range(config["n_epochs"]):
        for step, batch in enumerate(train_dataloader, start=1):
            input_ids = torch.tensor(batch["input_ids"])
            if step == 1:
                print(f"Input size: {len(input_ids)}")
            attention_mask = torch.tensor(batch["attention_mask"])
            labels = torch.tensor(batch["labels"])
            # labels[labels == tokenizer.pad_token_id] = -100
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            log_metrics(logger, tb_writer, step, {'lr': get_lr(), 'samples': step * config["full_batch_size"],
                                                  'steps': completed_steps, 'loss/train': loss.item()})
            loss = loss / config["gradient_accumulation_steps"]
            accelerator.backward(loss)
            if step % config["gradient_accumulation_steps"] == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
            if step % config["evaluation_steps"] == 0:
                logger.info('Evaluating model')
                eval_loss, perplexity = evaluate(model, eval_dataloader, config)
                log_metrics(logger, tb_writer, step, {'loss/eval': eval_loss, 'perplexity': perplexity})
                logger.info('Saving model checkpoint')
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                if accelerator.is_main_process:
                    torch.save(unwrapped_model.state_dict(), args.output_dir + "/model.checkpoint")
                model.train()

    # Evaluate and save the last checkpoint
    logger.info('Evaluating and saving model after training')
    eval_loss, perplexity = evaluate(model, eval_dataloader, config)
    log_metrics(logger, tb_writer, step, {'loss/eval': eval_loss, 'perplexity': perplexity})
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        torch.save(unwrapped_model.state_dict(), args.output_dir + "/model.checkpoint")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune a LLM on transitions")
    parser.add_argument(
        "--data_dir",
        type=str
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="trajectories"
    )
    parser.add_argument(
        "--file_id",
        type=str,
        default="13"
    )
    parser.add_argument(
        "--model_dir",
        type=str
    )
    parser.add_argument(
        "--output_dir",
        type=str
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int
    )
    parser.add_argument(
        "--seed",
        type=int
    )

    args = parser.parse_args()
    launch_training(args)