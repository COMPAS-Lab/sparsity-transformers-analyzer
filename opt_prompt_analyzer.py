from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    OPTForCausalLM,
    default_data_collator
)
from transformers.utils import logging as hf_logging
from datasets import load_dataset
from torch.utils.data import DataLoader
import evaluate
import torch
from torch import nn
from utils import move_to
import numpy as np
# from llm_serving.model.wrapper import get_model
import random, logging, sys, re
from tqdm import tqdm
from sparse_tensor_analyzer import get_mat_sparsity
from transformer_visualization import plot_heatmap

# MODEL_NAME = "facebook/opt-iml-max-1.3b"
MODEL_NAME = "facebook/opt-13b"
OPT_CACHE = "/chronos_data/tji/.huggingface_cache/"

# init log
log_fhandler = logging.FileHandler("opt_prompt_analyzer.log", mode="w", encoding="utf-8")
log_stdout_handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger("opt_prompt_analyzer")
logger.setLevel(logging.INFO)
logger.addHandler(log_fhandler)
logger.propagate = False
hf_logging.disable_default_handler()
hf_logging.add_handler(log_stdout_handler)

def load_and_prepare_dataset(tokenizer, num_examples=-1, split="validation", pad_on_right=True, batch_size=3, max_seq_len=1024, stride=32):
    raw_data = load_dataset("boolq", split=split)

    column_names = raw_data.column_names
    question_column_name = "question_wq"
    context_column_name = "passage"
    answer_column_name = "answer"

    def tokenize_data(examples):
        # examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            # examples["full"],
            # truncation="only_first",
            max_length=max_seq_len,
            stride=stride,
            padding="max_length",
        )

        return tokenized_examples

    if num_examples > -1:
        selected_idx = random.sample(range(len(raw_data)), num_examples)
        raw_data = raw_data.select(selected_idx)
        # raw_data = raw_data.select(range(num_examples))

    # append sep token to question and contexts
    def prompt_ques(a):
        context_part = "Read the following text and choose \"Yes\" or \"No\" to answer the question: " + a["passage"] + "\n"
        ques_part = "Question: " + a["question"] + "?\n"
        ans_part = "Answer: \n\n"
        return {"full": context_part + ques_part + ans_part}

    raw_data = raw_data.map(
        lambda a: {"question_wq": f"\nQuestion: " + a["question"] + "?\nAnswer (\"Yes\" or \"No\"): \n\n"}
        # lambda a: {"question_wq": "\nBased on the above paragraph, answer the following question with \"Yes\" or \"No\": " + a["question"] + "?\n\n"}
        # prompt_ques
    )
    
    tokenized_data = raw_data.map(
        tokenize_data, 
        batched=True,
        remove_columns=["passage", "question", "question_wq"],
        load_from_cache_file=True
    )
    tokenized_data.set_format("torch")
    print(tokenized_data)

    tokenized_dataloader = DataLoader(
        tokenized_data, shuffle=False, collate_fn=default_data_collator, batch_size=batch_size
    )

    return tokenized_dataloader

def infer_ans_directly(
        model_res, ref,
        model_res_avaliable_ans_only, ref_ans_only,
        prompt_ans_dict: dict
        ):
    generated_string = prompt_ans_dict.get("generated_string", None)
    answers = prompt_ans_dict.get("answers", None)
    
    if generated_string is None or answers is None:
        raise TypeError("Invalid answers to infer ans directly!")
    else:
        for i, answer in zip(generated_string, answers):
            ans_str = i.split("\n\n")[-1].lower()
            re_code = r"[_|\W]*(yes|no)[_|\W]*"
            stripped_ans = "".join(re.findall(re_code, ans_str))

            if "yes" == stripped_ans:
                model_res.append("True")
                model_res_avaliable_ans_only.append(True)
                ref_ans_only.append(answer.item())
                num_true += 1
            elif "no" == stripped_ans:
                model_res.append("False")
                model_res_avaliable_ans_only.append(False)
                ref_ans_only.append(answer.item())
                num_false += 1
            else:
                model_res.append("Neither")

        ref.append(str(answer.item()))

    return model_res, ref, model_res_avaliable_ans_only, ref_ans_only

def infer_by_logits(
        model_res, ref,
        model_res_avaliable_ans_only, ref_ans_only,
        prompt_ans_dict: dict      
    ):

    out_strs = prompt_ans_dict.get("out_strs", None)
    transition_probs = prompt_ans_dict.get("transition_probs", None)
    answers = prompt_ans_dict.get("answers", None)
    yes_ids = prompt_ans_dict.get("yes_ids", None)
    no_ids = prompt_ans_dict.get("no_ids", None)

    for generated_str, probability, answer in zip(out_strs, transition_probs, answers):
        generated_str = generated_str.lower()
        
        re_yes_match = re.fullmatch(r"[_|\W]*(yes)[_|\W]*", generated_str)
        re_no_match = re.fullmatch(r"[_|\W]*(no|not)[_|\W]*", generated_str)
        
        if re_yes_match:
            model_res.append("True")
            model_res_avaliable_ans_only.append(True)
            ref_ans_only.append(answer.item())
        elif re_no_match:
            model_res.append("False")
            model_res_avaliable_ans_only.append(False)
            ref_ans_only.append(answer.item())
        else:
            yes_prob_sum = sum([probability[idx] for idx in yes_ids])
            no_prob_sum = sum([probability[idx] for idx in no_ids])
            model_res.append(str((yes_prob_sum > no_prob_sum).item()))

        ref.append(str(answer.item()))

    return model_res, ref, model_res_avaliable_ans_only, ref_ans_only

def run_prob_eval(test_data, tokenizer, num_examples):
    # Load the model. Alpa automatically downloads the weights to the specificed path
    model = OPTForCausalLM.from_pretrained(MODEL_NAME)
    # model = get_model(model_name="alpa/opt-30b", path=OPT_CACHE)

    # Generate
    model_res, ref = [], []
    num_true, num_false = 0, 0
    model_res_avaliable_ans_only, ref_ans_only = [], []

    for step, batch in enumerate(test_data):
        output = model.generate(
            batch["input_ids"], 
            attention_mask=batch["attention_mask"], 
            max_new_tokens=1, do_sample=True
        )

        actual_input_len = torch.count_nonzero(batch["attention_mask"], dim=-1)

        generated_string = tokenizer.batch_decode(output, skip_special_tokens=True)
        logger.info(f"example {step}:")

        for i, answer in zip(generated_string, batch["answer"]):
            ans_str = i.split("\n\n")[-1]
            stripped_ans = ans_str.strip().lower()

            if "yes" == stripped_ans:
                model_res.append("True")
                model_res_avaliable_ans_only.append(True)
                ref_ans_only.append(answer.item())
                num_true += 1
            elif "no" == stripped_ans:
                model_res.append("False")
                model_res_avaliable_ans_only.append(False)
                ref_ans_only.append(answer.item())
                num_false += 1
            else:
                model_res.append("Neither")

            ref.append(str(answer.item()))

    print(model_res, ref)
    print(model_res_avaliable_ans_only, ref_ans_only)
     
    em_metric = evaluate.load("exact_match")
    res_em = em_metric.compute(predictions=model_res, 
                                references=ref)
    logger.info(f"exact match: {res_em}")
    print("exact match: ", res_em)

    higher_freq_true = None
    if (len(model_res_avaliable_ans_only) > 0):
        f1_metric = evaluate.load("f1")
        res_f1 = f1_metric.compute(predictions=model_res_avaliable_ans_only, 
                                    references=ref_ans_only)
        acc_metric = evaluate.load("accuracy")
        res_acc = acc_metric.compute(predictions=model_res_avaliable_ans_only,
                                    references=ref_ans_only)

        logger.info(f"accuracy: {res_acc}")
        print("accuracy: ", res_acc)
        logger.info(f"accuracy: {res_f1}")
        print("accuracy: ", res_f1)

        print(f"Freq of True: {num_true/num_examples}, False: {num_false/num_examples}")
        higher_freq_true = (num_true >  num_false)
    
    return higher_freq_true

def run_eval_with_constraints(
        test_data, 
        tokenizer, 
        eval_method,
        yes_ids, no_ids,
        device="cuda:0", 
        is_forcing_words=False
        ):
    '''
    Examine only yes or no answers
    '''
    # Load the model. Alpa automatically downloads the weights to the specificed path
    model = OPTForCausalLM.from_pretrained(MODEL_NAME)
    model = model.to(device)
    # model = get_model(model_name="alpa/opt-30b", path=OPT_CACHE)

    # Generate
    model_res, ref = [], []
    num_true, num_false = 0, 0
    model_res_avaliable_ans_only, ref_ans_only = [], []
    attn_sparsities = []
    
    force_words_ids = None
    if is_forcing_words:
        single_word_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b", add_prefix_space=True)
        force_words = ["Yes", "No"]
        force_words_ids = single_word_tokenizer(force_words, add_special_tokens=False).input_ids
        print(force_words_ids)

    num_examples = 0
    for step, batch in tqdm(enumerate(test_data), total=len(test_data)):
        batch = move_to(batch, device)
        num_beams = 1
        batch_size = len(batch["input_ids"])
        gen_params = {
            "inputs": batch["input_ids"], 
            "attention_mask": batch["attention_mask"],
            "max_new_tokens": 1,
            "num_beams": num_beams,
            "output_attentions": True,
            "output_scores": True,
            "return_dict_in_generate": True,
            }
        if is_forcing_words:
            gen_params.update({
                "force_words_ids": force_words_ids
            })
        
        output = model.generate(**gen_params)
        seq_ids = output.sequences
        transition_probs = nn.functional.softmax(output.scores[0], dim=-1)

        ori_prompt = tokenizer.batch_decode(seq_ids)
        out_strs = []
        for out_inst in ori_prompt:
            out_strs.append(out_inst.split("\n\n")[-1])
        
        # check attention
        # expected atten size: layer_size, num_beamsxbatch_sizexhead_size, len, len
        attens = [i.to("cpu") for i in output.attentions[0]]
        attens = torch.stack(attens)
        layer_size, head_size, seq_len, _ = attens.size()
        attens = attens.view(layer_size, head_size // (num_beams*batch_size), 
                                num_beams*batch_size, seq_len, seq_len)
        for i in range(num_beams*batch_size):
            actual_input_len = torch.count_nonzero(batch["attention_mask"][i], dim=-1).item()
            curr_attens = torch.squeeze(attens[:,:,i,-actual_input_len:,-actual_input_len:])
            attn_sparsities.append(get_mat_sparsity(curr_attens, causal_mask=True))
        
        # batch_idx, layer_idx = 0, 10
        # ori_prompt = tokenizer.decode(seq_ids[batch_idx][-actual_input_len:])
        # logger.info(f"Prompt: {ori_prompt}")
        # true_ans = batch["answer"][batch_idx]
        # logger.info(f"true answer: {true_ans}")
        # logger.info(f"actual len: {actual_input_len}")

        # sampled_attn = attens[:, :, batch_idx, -actual_input_len:, -actual_input_len:].numpy()
        # plot_heatmap(sampled_attn, sparsity_bar=0.0, auto_scale=False, binarize=False, fig_gird=(4, 8))
        # col_max_idx = np.argmax(np.sum(sampled_attn, axis=-2, keepdims=False), axis=-1, keepdims=False)
        # row_max_idx = np.argmax(np.sum(sampled_attn, axis=-1, keepdims=False), axis=-1, keepdims=False)

        # for head, (r, c) in enumerate(zip(row_max_idx[layer_idx], col_max_idx[layer_idx])):
        #     full_seqlen = batch["input_ids"].size()[-1]
        #     r_id = seq_ids[batch_idx][r+full_seqlen-actual_input_len]
        #     c_id = seq_ids[batch_idx][c+full_seqlen-actual_input_len]
        #     r_word = tokenizer.decode(r_id)
        #     c_word = tokenizer.decode(c_id)
        #     logger.info(f"head {head} focusing {r_word} on {c_word}, attn val: {sampled_attn[layer_idx][head][r][c]:.4f}")

        generated_string = tokenizer.batch_decode(seq_ids, skip_special_tokens=True)
        model_res, ref, model_res_avaliable_ans_only, ref_ans_only = \
            eval_method(model_res, ref, model_res_avaliable_ans_only, ref_ans_only, 
                        {
                            "generated_string": generated_string, 
                            "answers": batch["answer"],
                            "out_strs": out_strs,
                            "transition_probs": transition_probs,
                            "yes_ids": yes_ids,
                            "no_ids": no_ids,
                            }
                        )
        
        num_examples += batch_size
    
    avg_sparsity = np.mean(attn_sparsities)
    logger.info(f"avg sparsity: {avg_sparsity:.4f}")
    metric_all = evaluate.load("exact_match")
    res_em = metric_all.compute(predictions=model_res, 
                                references=ref)
    logger.info(f"exact match: {res_em}")
    print("exact match: ", res_em)

    for i in range(len(model_res)):
        model_res[i] = True if model_res[i] == "True" else False
        ref[i] = True if ref[i] == "True" else False

    metric_all = evaluate.load("f1")
    res_f1_all = metric_all.compute(predictions=model_res, 
                                references=ref)
    logger.info(f"f1 for all: {res_f1_all}")
    print("f1 for all: ", res_f1_all)

    if (len(model_res_avaliable_ans_only) > 0):
        f1_metric = evaluate.load("f1")
        res_f1 = f1_metric.compute(predictions=model_res_avaliable_ans_only, 
                                    references=ref_ans_only)
        acc_metric = evaluate.load("accuracy")
        res_acc = acc_metric.compute(predictions=model_res_avaliable_ans_only,
                                    references=ref_ans_only)

        logger.info(f"accuracy: {res_acc}")
        print("accuracy: ", res_acc)
        logger.info(f"f1: {res_f1}")
        print("f1: ", res_f1)

        print(f"Freq of True: {num_true/num_examples}, False: {num_false/num_examples}")

def get_yes_no_ids(tokenizer):
    vocab = tokenizer.get_vocab()
    re_yes_code = r"[_|\W]*(yes)[_|\W]*"
    re_no_code = r"[_|\W]*(no|not)[_|\W]*"

    yes_idx, no_idx = [], []
    
    for key, val in vocab.items():
        lower_key = key.lower()
        yes_match_res = re.fullmatch(re_yes_code, lower_key)
        no_match_res = re.fullmatch(re_no_code, lower_key)

        if yes_match_res:
            print(key, val)
            yes_idx.append(val)
        
        if no_match_res:
            print(key, val)
            no_idx.append(val)

    return yes_idx, no_idx
        
def main():
    num_examples = -1

    # Load the tokenizer. All OPT models with different sizes share the same tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b", padding_side="left")
    tokenizer.add_bos_token = False

    test_data = load_and_prepare_dataset(tokenizer, batch_size=1, pad_on_right=False, num_examples=num_examples)
    yes_ids, no_ids = get_yes_no_ids(tokenizer)
    # higher_freq_true = run_prob_eval(test_data, tokenizer, num_examples)
    run_eval_with_constraints(test_data, tokenizer, eval_method=infer_by_logits, yes_ids=yes_ids, no_ids=no_ids, is_forcing_words=False)

    
if __name__ == "__main__":
    main()
