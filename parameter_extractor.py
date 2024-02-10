'''
extract parameters from
'''
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM, 
    LlamaTokenizer,
    OPTForCausalLM,
    default_data_collator
)
import torch

from sparsemat_hw_modeling import get_mat_sparsity
from transformer_visualization import plot_heatmap

from analyze_tcblock_vs_matsize import closest_factors_to_target
from sparsemat_hw_modeling import single_case_analyzing

MODEL_NAME = "decapoda-research/llama-7b-hf"
# MODEL_NAME = "facebook/opt-13b"



def main():
    '''
    Examine only yes or no answers
    '''
    transformers.logging.set_verbosity_info()

    l1_fc1_w = None
    # Load the model.
    model = OPTForCausalLM.from_pretrained(MODEL_NAME)
    with torch.no_grad():
        fname = MODEL_NAME.split("/")[-1]+"-paramlist.txt"
        with open(fname, mode="w+") as f:
            for name, param in model.named_parameters():
                f.write(f"{name}: {param.size()}\n")
                if name == "model.decoder.layers.0.self_attn.k_proj.weight":
                    l1_fc1_w = param.numpy()

    single_case_analyzing(l1_fc1_w, 
                          (l1_fc1_w.shape[1], 5120), 
                          hw_array_shape=closest_factors_to_target(24, 3960.0), 
                          chain_len = 24)


if __name__ == "__main__":
    main()