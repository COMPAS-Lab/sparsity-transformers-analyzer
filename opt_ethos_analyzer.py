"""
opt analyzer: analyzer sparsity of opt
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
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
MODEL_NAME = "facebook/opt-30b"

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
    print(ethos_dat_training)
    ethos_dat_eval = load_dataset("ethos", "binary", split="train[:-10%]")

    tokenized_datasets = {"train": ethos_dat_training.map(tokenize, batched=True), 
                            "test": ethos_dat_eval.map(tokenize, batched=True)}

    return tokenized_datasets

def finetune_model():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred_res = np.argmax(logits, axis=-1)
        metric = evaluate.load("accuracy")
        return metric.compute(predictions=pred_res, references=labels)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=".opt_cache", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tokenized_dataset = prepare_dataset_and_tokenize_for_training()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
                        output_dir="./params",
                        learning_rate=2e-5,
                        do_train=True,
                        do_eval=True,
                        per_device_train_batch_size=16,
                        num_train_epochs=3,
                        weight_decay=0.01,
                        evaluation_strategy="epoch",
                        fp16=True,
                    )

    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=tokenized_dataset["train"],
                        eval_dataset=tokenized_dataset["test"],
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics
                    )

    trainer.train()
    analyze_model_params(model)
    trainer.save_model(".opt_cache/opt-1.3b-finetuned")

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
