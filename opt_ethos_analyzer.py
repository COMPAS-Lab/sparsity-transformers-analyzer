"""
opt analyzer: analyzer sparsity of opt
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
                            TrainingArguments, Trainer, DataCollatorWithPadding, \
                            AdamW, get_scheduler
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict
import evaluate
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd
from torch import nn

from pprint import pprint

PARAM_PATH = "./params/"
DATA_PATH = "./data"
CONTEXT_LEN = 128
MODEL_NAME = "facebook/opt-6.7b"

def analyze_model_params(model):
    for name, params in model.named_parameters():
        # print(name, ":", params.shape)
        if (name == "score.weight"):
            print(params.shape)
            print(params[0])

def prepare_dataset_and_tokenize_for_evaluation(split="train"):
        
    def tokenize(element):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        outputs = tokenizer(
            element,
            # return_overflowing_tokens=True,
            # return_length=True,
            return_tensors="pt",
        )
        return outputs.input_ids

    ethos_dat = load_dataset("ethos", "binary", split=split)
    print(ethos_dat)

    tokenized_datasets = []
    for i in ethos_dat:
        tokenized_datasets.append({"ids": tokenize(i["text"]), "label": i["label"]})
    # tokenized_datasets = ethos_dat.map(tokenize, batched=False)

    print("num insts: ", len(tokenized_datasets))
    return tokenized_datasets

def prepare_dataset_and_tokenize_for_training():
    
    def tokenize(element):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        outputs = tokenizer(
            element["text"],
            truncation=True,
            padding='max_length',
            max_length=CONTEXT_LEN,
            # return_overflowing_tokens=True,
            # return_length=True
        )
        return outputs

    ethos_dat_training = load_dataset("ethos", "binary", split="train[:90%]")
    ethos_dat_eval = load_dataset("ethos", "binary", split="train[:-10%]")

    train_dataset = ethos_dat_training.map(tokenize, batched=True)
    train_dataset.set_format("pt", columns=["input_ids", "attention_mask"], output_all_columns=True)

    eval_dataset = ethos_dat_training.map(tokenize, batched=True)
    eval_dataset.set_format("pt", columns=["input_ids", "attention_mask"], output_all_columns=True)

    return train_dataset, eval_dataset

def finetune_model():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred_res = np.argmax(logits, axis=-1)
        metric = evaluate.load("accuracy")
        return metric.compute(predictions=pred_res, references=labels)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=".opt_cache", num_labels=2)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    train_dataset, eval_dataset = prepare_dataset_and_tokenize_for_training()
    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=16)

    accelerator = Accelerator()

    train_dataset, eval_dataset, model, optimizer = accelerator.prepare(
        train_dataset, eval_dataset, model, optimizer
    )

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataset)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataset:
            outputs = model(batch["input_ids"].to(model.device), labels = batch["label"], attention_mask=batch["attention_mask"].to(model.device), output_hidden_states=False, output_attentions=True)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    model.save_pretrained(".opt_cache/opt-6.7b-finetuned")
    
def evaluate_model():
    losses = []
    res_probs = []
    predicted_labels = []
    num_labels_as_one = 0.0

    model = AutoModelForSequenceClassification.from_pretrained(".opt_cache/opt-1.3b-finetuned", num_labels=2)
    tokenized_dataset = prepare_dataset_and_tokenize_for_evaluation()
    # eval_dataloader = DataLoader(tokenized_dataset, batch_size = 4)
    # print(f"len of eval data: {len(eval_dataloader)}")
    # run model
    analyze_model_params(model)

    all_attn = []
    max_seq_len = 0
    for step, dat in enumerate(tokenized_dataset):
        print(f"step {step} :")
        with torch.no_grad():
            input_ids_tensor, l = \
                dat["ids"].to(model.device), \
                torch.tensor(dat["label"], dtype=torch.long,  device="cpu")
            model_output = model(input_ids_tensor, \
                                    output_hidden_states=False, output_attentions=True, \
                                    labels=l)
        losses.append(model_output.loss.item())
        prob = nn.functional.softmax(torch.squeeze(model_output.logits), dim=-1)
        res_probs.append(prob)
        predicted_label = 0 if prob[0].item() > prob[1].item() else 1
        predicted_labels.append(predicted_label)
        if dat["label"] == 1:
            num_labels_as_one += 1.0

        if predicted_label != dat["label"]:
            print(f"find mismatch: {predicted_label} vs. ", dat["label"])
            print(f"logits:{torch.squeeze(model_output.logits)}, prob: {prob}")

        # curr_attn = torch.stack(list(model_output.attentions)).to("cpu")
        # curr_attn = torch.squeeze(curr_attn)
        # print(curr_attn.size())
        # if max_seq_len < curr_attn.size()[-1]:
        #     max_seq_len = curr_attn.size()[-1]
        # all_attn.append(curr_attn)

    print(f"number of hate speeches: {num_labels_as_one / len(tokenized_dataset)}")
    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")
    res_f1 = f1_metric.compute(predictions=predicted_labels, 
                                references = [i["label"] for i in tokenized_dataset])
    res_accu = acc_metric.compute(predictions=predicted_labels, 
                                references = [i["label"] for i in tokenized_dataset])
    return res_probs, res_accu, res_f1

    # prepare all attention and save them
    for l in range(48):
        attns_from_same_layer = []
        seq_lens = []
        for inst in all_attn:
            zeros_to_pad = max_seq_len - inst.size()[-1]
            print("size before padding: ", inst[l].size())
            padded_inst = torch.nn.functional.pad(\
                            inst[l], (0, zeros_to_pad, 0, zeros_to_pad, 0, 0), "constant", 0)
            print("size after padding: ", padded_inst.size())
            attns_from_same_layer.append(padded_inst)
            seq_lens.append(inst.size()[-1])

        attns_from_same_layer = torch.stack(attns_from_same_layer)
        seq_lens = torch.tensor(seq_lens).type(torch.int)
        print(f"layer {l} attn shape: {attns_from_same_layer.size()}")
        torch.save(attns_from_same_layer, f"./temp_dat/bfp_attn/{l}-0.pt")
        torch.save(seq_lens, f"./temp_dat/seqlen/{l}-0.pt")

    return res_probs, res_accu

def main():
    finetune_model()
    logits, accu, f1 = evaluate_model()
    print("accu: ", accu, "f1: ", f1)


if __name__ == "__main__":
    main()
