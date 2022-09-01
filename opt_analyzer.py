"""
opt analyzer: analyzer sparsity of opt
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, DatasetDict
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd

import argparse as ag
import os
import sys
import random
from datetime import datetime
import math
import glob
from textwrap import wrap
from itertools import compress, product
import json

PARAM_PATH = "./params/"
DATA_PATH = "./data"
CONTEXT_LEN = 512
MODEL_NAME = "facebook/opt-125m"

def tokenize(element):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    outputs = tokenizer(
        element["context"],
        truncation=True,
        padding='max_length',
        max_length=CONTEXT_LEN,
        return_overflowing_tokens=True,
        return_length=True,
    )

    return {"input_ids": outputs["input_ids"][:100]}

def prepare_dataset_and_tokenize():
    ed_dataset_train = load_dataset("empathetic_dialogues", split="train")
    ed_dataset_valid = load_dataset("empathetic_dialogues", split="validation")
    raw_datasets = DatasetDict({
        "train": ed_dataset_train,
        "valid": ed_dataset_valid,
    })

    print(raw_datasets)

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    return tokenized_datasets

def evaluate_model():
    loss = 0.0
    losses = []

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", cache_dir=".opt_cache")
    tokenized_dataset = prepare_dataset_and_tokenize()
    print(tokenized_dataset)
    eval_dataloader = DataLoader(tokenized_dataset["valid"], batch_size = 32)
    print(f"len of eval data: {len(eval_dataloader)}")
    # run model
    for step, batch in enumerate(eval_dataloader):
        print(f"step {step} :")
        with torch.no_grad():
            input_ids_tensor = torch.stack(batch["input_ids"])
            model_output = model(input_ids_tensor, \
                                    output_hidden_states=True, \
                                    labels=input_ids_tensor)
        print(model_output.loss)
        losses.append(model_output.loss.item())
    
    loss = np.mean(losses)
    try:
        pplx = np.exp(loss)
    except OverflowError:
        pplx = float("inf")

    return pplx.item()

def main():
    pplx = evaluate_model()
    print("ppl: ", pplx)


if __name__ == "__main__":
    main()