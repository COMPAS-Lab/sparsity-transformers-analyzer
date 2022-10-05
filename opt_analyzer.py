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
CONTEXT_LEN = 1024
MODEL_NAME = "facebook/opt-30b"

def tokenize(element):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    outputs = tokenizer(
        element,
        # truncation=False,
        # padding='max_length',
        # max_length=CONTEXT_LEN,
        # return_overflowing_tokens=True,
        # return_length=True,
        return_tensors="pt"
    )

    return outputs.input_ids

def prepare_dataset_and_tokenize():
    azreview_dataset_valid = load_dataset("amazon_reviews_multi", split="test")
    azreview_dataset_valid = azreview_dataset_valid.filter(lambda x: x["language"] == "en")
    print(azreview_dataset_valid)

    tokenized_datasets = [tokenize(i) for i in azreview_dataset_valid["review_body"]]
    # tokenized_datasets = azreview_dataset_valid.map(tokenize, batched=False)
    return tokenized_datasets

def evaluate_model():
    loss = 0.0
    losses = []

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", cache_dir=".opt_cache")
    tokenized_dataset = prepare_dataset_and_tokenize()
    # eval_dataloader = DataLoader(tokenized_dataset, batch_size = 4)
    # print(f"len of eval data: {len(eval_dataloader)}")
    # run model
    for step, input_ids_tensor in enumerate(tokenized_dataset):
        print(f"step {step} :")
        with torch.no_grad():
            input_ids_tensor = input_ids_tensor.to(model.device)
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
