'''utilities for the experiments'''

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM, 
    LlamaTokenizer,
    OPTForCausalLM,
    PreTrainedTokenizer,
)
from datasets import load_dataset
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
import torch
from tqdm import tqdm

def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        print(type(obj))
        raise TypeError("Invalid type for move_to")

def estimate_mem_usage(model_name):
    if "llama" in model_name:
        model = LlamaForCausalLM.from_pretrained(model_name)
    elif "opt" in model_name:
        model = OPTForCausalLM.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
    estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)

def extract_param_names(model_name):
    if "llama" in model_name:
        model = LlamaForCausalLM.from_pretrained(model_name)
    elif "opt" in model_name:
        model = OPTForCausalLM.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    fname = model_name.split("/")[-1] + "-paramlist.txt"
    with open(fname, "w+") as f:
        f.write(str(model.config))
        f.write("\n\n")
        for name, params in model.named_parameters():
            f.write(name + ": " + str(params.size()) + "\n")

def examine_dataset_seqlen(tokenizer_name: str, 
                           dataset_name: str,
                           seqlen_bar: int,
                           dataset_key = None,
                           split = "validation",
                           ):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = load_dataset(dataset_name, split=split)

    dataset_len = len(dataset)
    insts_longer_than_bar = 0
    for i in tqdm(list(range(dataset_len)), total=dataset_len):
        if dataset_key is None:
            tokenized_dat = tokenizer(dataset[i])
        else:
            tokenized_dat = tokenizer(dataset[i][dataset_key])

        if len(tokenized_dat["input_ids"]) > seqlen_bar:
            insts_longer_than_bar += 1
    
    print('''inst length detection: '''
            f'''{insts_longer_than_bar}/{dataset_len} longer than {seqlen_bar}''')