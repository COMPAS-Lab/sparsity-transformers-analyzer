"""
opt analyzer: analyzer sparsity of opt
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, DatasetDict
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd
from math import isnan

from pprint import pprint

PARAM_PATH = "./params/"
DATA_PATH = "./data"
CONTEXT_LEN = 1024
MODEL_NAME = "facebook/opt-30b"
NUM_LAYERS = 48

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
        if len(tmp_long_seq) == 3:
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

    all_attn = []
    max_seq_len = 0
    for step, input_ids_tensor in enumerate(tokenized_dataset):
        print(f"step {step} :")
        with torch.no_grad():
            input_ids_tensor = input_ids_tensor.to(model.device)
            model_output = model(input_ids_tensor, \
                                    output_hidden_states=False, output_attentions=True, \
                                    labels=input_ids_tensor)
        losses.append(model_output.loss.item())
        curr_attn = torch.stack(list(model_output.attentions)).to("cpu")
        curr_attn = torch.squeeze(curr_attn)

        if isnan(losses[-1]):
            print(model_output.logits)
            exit()

        print(curr_attn.size())
        if max_seq_len < curr_attn.size()[-1]:
            max_seq_len = curr_attn.size()[-1]
        all_attn.append(curr_attn)

    # prepare all attention and save them
    # for l in range(NUM_LAYERS):
    #     attns_from_same_layer = []
    #     seq_lens = []
    #     for inst in all_attn:
    #         zeros_to_pad = max_seq_len - inst.size()[-1]
    #         padded_inst = torch.nn.functional.pad(\
    #                         inst[l], (0, zeros_to_pad, 0, zeros_to_pad, 0, 0), "constant", 0)
    #         attns_from_same_layer.append(padded_inst)
    #         seq_lens.append(inst.size()[-1])

    #     attns_from_same_layer = torch.stack(attns_from_same_layer)
    #     seq_lens = torch.tensor(seq_lens).type(torch.int)
    #     print(f"layer {l} attn shape: {attns_from_same_layer.size()}")
    #     torch.save(attns_from_same_layer, f"./temp_dat/bfp_attn/{l}-0.pt")
    #     torch.save(seq_lens, f"./temp_dat/seqlen/{l}-0.pt")
    
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
