'''utilities for the experiments'''

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM, 
    LlamaTokenizer,
    OPTForCausalLM,
)
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
import torch

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