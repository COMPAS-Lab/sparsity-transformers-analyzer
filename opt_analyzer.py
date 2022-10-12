"""
opt analyzer: analyzer sparsity of opt
"""
from lib2to3.pgen2 import token
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
    wikitext_valid = load_dataset("wikitext", "wikitext-103-v1", split="test")
    # wikitext_valid = wikitext_valid.filter(lambda x: x["language"] == "en")
    print(wikitext_valid)

    tokenized_datasets, tmp_long_seq = [], []
    for i in wikitext_valid:
        if len(i["text"]) > 600:
            tmp_long_seq.append(i["text"])
        if len(tmp_long_seq) is 5:
            tokenized_datasets.append(tokenize(" ".join(tmp_long_seq)))
            tmp_long_seq = []
    # tokenized_datasets = azreview_dataset_valid.map(tokenize, batched=False)
    print("num insts: ", len(tokenized_datasets))
    return tokenized_datasets

def evaluate_model():
    loss = 0.0
    losses = []

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", cache_dir=".opt_cache")
    tokenized_dataset = prepare_dataset_and_tokenize()
    # eval_dataloader = DataLoader(tokenized_dataset, batch_size = 4)
    # print(f"len of eval data: {len(eval_dataloader)}")
    # run model
    for step, input_ids_tensor in enumerate(tokenized_dataset[:100]):
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
