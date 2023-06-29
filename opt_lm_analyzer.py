"""
opt analyzer: analyzer sparsity of opt
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
import random
from utils import move_to
from sparse_tensor_analyzer import get_mat_sparsity

PARAM_PATH = "./params/"
DATA_PATH = "./data"
CONTEXT_LEN = 2048
MODEL_NAME = "facebook/opt-350m"
NUM_LAYERS = 24

def tokenize(element):
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b")
    outputs = tokenizer(
        element["article"],
        truncation=True,
        # padding='max_length',
        max_length=CONTEXT_LEN,
        stride = 5, 
        # return_overflowing_tokens=True,
        # return_length=True,
    )

    return outputs

def prepare_dataset_and_tokenize():
    wikitext_valid = load_dataset("wikitext", "wikitext-103-v1", split="test")
    # wikitext_valid = wikitext_valid.filter(lambda x: x["language"] == "en")
    print(wikitext_valid)

    tokenized_datasets, tmp_long_seq = [], []
    for i in wikitext_valid:
        # select sentences larger than 10
        if (len(i["text"].split()) > 10):
            tokenized_datasets.append({"input_ids": tokenize(i["text"])})
        # if len(i["text"]) > 600:
        #     tmp_long_seq.append(i["text"])
        # if len(tmp_long_seq) == 3:
        #     tokenized_datasets.append(tokenize(" ".join(tmp_long_seq)))
        #     tmp_long_seq = []

    print("num insts: ", len(tokenized_datasets))
    return tokenized_datasets

def prepare_scipaper_dataset_and_tokenize(nsamples = -1):
    sci_paper_data = load_dataset("scientific_papers", "pubmed", split="train")

    if nsamples > 0:
        total_len = len(sci_paper_data)
        selected_idx = random.sample(list(range(total_len)), nsamples)
        sci_paper_data = sci_paper_data.select(selected_idx)

    column_names = sci_paper_data.column_names
    tokenized_data = sci_paper_data.map(
        tokenize,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True
    )
    tokenized_data.set_format("torch")

    tokenized_dataloader = DataLoader(
        tokenized_data, collate_fn=default_data_collator, batch_size=1, shuffle=False
    )

    print(f"selected {len(tokenized_dataloader)} insts")
    return tokenized_dataloader

def evaluate_model(data_fn, nsamples=-1):
    loss = 0.0
    losses = []

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    tokenized_dataset = data_fn(nsamples)
    # eval_dataloader = DataLoader(tokenized_dataset, batch_size = 4)
    # print(f"len of eval data: {len(eval_dataloader)}")
    # run model

    all_attn = []
    max_seq_len = 0
    attn_sparsities = []
    for step, batch in enumerate(tokenized_dataset):
        print(f"--- step {step} ---")
        print(f'input shape: {batch["input_ids"].size()}')
        if batch["input_ids"].size()[-1] < 2:
            continue 
        batch = move_to(batch, model.device)
        gen_params = {
            "input_ids": batch["input_ids"], 
            "attention_mask": batch["attention_mask"],
            "output_attentions": True,
            "output_hidden_states": False,
            "labels": batch["input_ids"],
        }
        with torch.no_grad():
            model_output = model(**gen_params)
        losses.append(model_output.loss.item())
        curr_attn = torch.stack(list(model_output.attentions)).to("cpu")
        curr_attn = torch.squeeze(curr_attn)

        if max_seq_len < curr_attn.size()[-1]:
            max_seq_len = curr_attn.size()[-1]
        attn_sparsities.append(get_mat_sparsity(curr_attn, causal_mask=True))

        # prepare all attention and save them
        print(f"saving {step} with size {curr_attn.size()}")
        torch.save(curr_attn, f"/chronos_data/tji/opt_350m_sparse_attn/attn_s{step}b0.pt")
        
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

    return pplx.item(), attn_sparsities

def main():
    pplx, attn_sparsity = evaluate_model(prepare_scipaper_dataset_and_tokenize, 100)
    print("ppl: ", pplx)
    print("average sparsity: ", np.mean(attn_sparsity))


if __name__ == "__main__":
    main()
