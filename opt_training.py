from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    OPTForCausalLM,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import logging as hf_logging
from datasets import load_dataset
from promptsource.templates import DatasetTemplates
from torch.utils.data import DataLoader
import evaluate
import torch
from torch import nn
from torch.optim import AdamW
from utils import move_to, extract_param_names
import numpy as np
# from llm_serving.model.wrapper import get_model
import random, logging, sys, re
from tqdm.auto import tqdm
from sparse_tensor_analyzer import get_mat_sparsity
from transformer_visualization import plot_heatmap
from accelerate import Accelerator, find_executable_batch_size
from functools import reduce

def load_and_prepare_stance_det_trset(topic, split="train", use_promptsource=False):
    dataset_path = "/chronos_data/tji/.huggingface_cache/datasets/twitter_stance/"
    test_dat = dataset_path + f"stance_{topic}_test_with_history_v2.csv"
    train_dat = dataset_path + f"stance_{topic}_train_with_history_v2.csv"
    raw_data = load_dataset("csv", data_files={"train": train_dat}, split=split)
    TOPIC_LST = ["abortion", "atheism", "climate", "clinton", "feminist"]
    assert topic in TOPIC_LST, f"selected topic not in the support list!\n Supported topics are: {TOPIC_LST}"

    insts = raw_data.filter(lambda example: not example["stance"] is None)

    def remove_duplicate(x):
        res = []
        for i in x:
            if i not in res:
                res.append(i)
        return res

    concat_tweets = {}
    for uid in insts["user_id"]:
        curr_uid_insts = raw_data.filter(lambda example: example["user_id"] == uid)
        message_list = remove_duplicate(curr_uid_insts["message"])
        # new_history_tweets = " ".join(message_list[:50])
        new_history_tweets = message_list[:50]
        concat_tweets[uid] = new_history_tweets

    def replace_tweet(inst):
        text_his = concat_tweets[inst["user_id"]]
        text_anchor = inst["message"]
        text = '\n'.join(text_his + [text_anchor])
        st_str_list = {-1: "against", 1: "favor", 0: "neutral"}
        st_list = {-1: 1, 1: 2, 0: 0}
        text_his = ". ".join(text_his)
        ans = inst["stance"]

        if not use_promptsource:
            # text = f"Task: Based on the tweets below, is the person's stance in favor, neutral or against the topic {topic}? " + \
            #         f"Answer \"favor\", \"neutral\" or \"against\". \n" + \
            #         f"Tweet: {text} \n" + \
            #         f"Answer: "
            text = f"Tweet history: {text_his} \n" + \
                    f"Anchor Tweet: {text_anchor} \n" + \
                    f"Question: Based on the history of tweets from a person, is his/her stance pro, neutral or against {topic}? \n" + \
                    f"Answer: {st_str_list[ans]}"
            # text = f"Tweets: {text}\n" + \
            #         f"Question: In the tweets above, what is the author's stance on the {topic}? Answer with neutral, against or favor.\n" + \
            #         "Answer:\n\n"
            # text = f"Question: Does the author express any stance about {topic} in the following text? Answer favor, against or none. We say God bless America but we kill 4,000 babies a year. #SemST \n" + \
            #         f"Answer: \nagainst \n" + \
            #         f"Question: Does the author express any stance about {topic} in the following text? Answer favor, against or none. @user @user @user Yup. One of the MANY reasons I changed parties. #SemST \n" + \
            #         f"Answer: \nnone \n" + \
            #         f"Question: Does the author express any stance about {topic} in the following text? Answer favor, against or none. " + \
            #         f"Progress for #AfricanAmericans check. Progress for #Gay people check. Progress for #Women. Waiting waiting waiting.... #SemST \n" + \
            #         f"Answer: \nfavor \n" + \
            #         f"Question: Does the author express any stance about {topic} in the following text? Answer favor, against or none. " + \
            #         f"{' '.join(text_his)} " + f"{text_anchor} \n" + \
            #         "Answer: \n"
            return {"text": text, "ans": ans}

    column_names = insts.column_names
    insts = insts.map(replace_tweet, remove_columns=column_names)

    insts.to_csv(dataset_path + f"{topic}_stance_postprocessing.csv")

    return

def test_load_stance_det(topic, split="train"):
    dataset_path = "/chronos_data/tji/.huggingface_cache/datasets/twitter_stance/"
    train_dat = dataset_path + f"{topic}_stance_postprocessing.csv"
    raw_data = load_dataset("csv", data_files={"train": train_dat}, split=split)

    print(raw_data[0]["text"])
    print(raw_data[0]["ans"])

if __name__ == "__main__":
    load_and_prepare_stance_det_trset("abortion")
    test_load_stance_det("abortion")