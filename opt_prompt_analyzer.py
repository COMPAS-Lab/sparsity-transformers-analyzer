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
# from llm_serving.model.wrapper import get_model
import random, logging, sys, re
from tqdm import tqdm
from sparse_tensor_analyzer import get_mat_sparsity

MODEL_NAME = "facebook/opt-iml-max-1.3b"
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

def run_eval_with_constraints(test_data, tokenizer, num_examples, device="cuda:0", is_forcing_words=False):
    # Load the model. Alpa automatically downloads the weights to the specificed path
    model = OPTForCausalLM.from_pretrained(MODEL_NAME)
    model = model.to(device)
    # model = get_model(model_name="alpa/opt-30b", path=OPT_CACHE)

    # Generate
    model_res, ref = [], []
    num_true, num_false = 0, 0
    model_res_avaliable_ans_only, ref_ans_only = [], []
    num_beams = 3
    
    force_words_ids = None
    if is_forcing_words:
        single_word_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b", add_prefix_space=True)
        force_words = ["Yes", "No"]
        force_words_ids = single_word_tokenizer(force_words, add_special_tokens=False).input_ids
        print(force_words_ids)

    for step, batch in tqdm(enumerate(test_data)):
        batch = move_to(batch, device)
        gen_params = {
            "inputs": batch["input_ids"], 
            "attention_mask": batch["attention_mask"],
            "max_new_tokens": 1,
            "num_beams": num_beams,
            "output_attentions": True,
            "return_dict_in_generate": True,
            }
        if is_forcing_words:
            gen_params.update({
                "force_words_ids": force_words_ids
            })
        
        output = model.generate(**gen_params)
        seq_ids = output.sequences
        # expected atten size: layer_size, num_beamsxbatch_size, len, len
        attens = output.attentions

        actual_input_len = torch.count_nonzero(batch["attention_mask"], dim=-1)

        generated_string = tokenizer.batch_decode(seq_ids, skip_special_tokens=True)

        for i, answer in zip(generated_string, batch["answer"]):
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
     
    em_metric = evaluate.load("exact_match")
    res_em = em_metric.compute(predictions=model_res, 
                                references=ref)
    logger.info(f"exact match: {res_em}")
    print("exact match: ", res_em)

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

def eval_by_logits(tokenizer, eval_data, higher_freq_true, yes_ids, no_ids):
    # Load the model. Alpa automatically downloads the weights to the specificed path
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # model = get_model(model_name="alpa/opt-30b", path=OPT_CACHE)

    # Generate
    model_res, ref = [], []
    num_neither_ans, num_examples = 0, 0
    for step, batch in tqdm(enumerate(eval_data)):
        output = model.generate(
            batch["input_ids"], 
            attention_mask=batch["attention_mask"],
            max_new_tokens=1,
            output_scores=True, 
            return_dict_in_generate=True
        )

        transition_probs = nn.functional.softmax(output.scores[0], dim=-1)

        for generated_ids, probability, answer in zip(output.sequences, transition_probs, batch["answer"]):
            num_examples += 1
            generated_str = tokenizer.decode(generated_ids[-1])
            generated_str = generated_str.lower()
            
            re_yes_match = re.fullmatch(r"[_|\W]*(yes)[_|\W]*", generated_str)
            re_no_match = re.fullmatch(r"[_|\W]*(no|not)[_|\W]*", generated_str)
            
            if re_yes_match:
                model_res.append(True)
            elif re_no_match:
                model_res.append(False)
            else:
                num_neither_ans += 1
                yes_prob_sum = sum([probability[idx] for idx in yes_ids])
                no_prob_sum = sum([probability[idx] for idx in no_ids])

                if higher_freq_true:
                    model_res.append((yes_prob_sum > no_prob_sum).item())
                else:
                    model_res.append(not (yes_prob_sum > no_prob_sum).item())

            ref.append(answer.item())

    logger.info(model_res)
    logger.info(ref)
    f1_metric = evaluate.load("f1")
    res_f1 = f1_metric.compute(predictions=model_res, 
                                references=ref)
    acc_metric = evaluate.load("accuracy")
    res_acc = acc_metric.compute(predictions=model_res,
                                references=ref)

    logger.info(f"number of neither-yes-nor-no answers: {num_neither_ans}/{num_examples}")
    logger.info(f"accuracy: {res_acc}")
    print("accuracy: ", res_acc)
    logger.info(f"accuracy: {res_f1}")
    print("accuracy: ", res_f1)

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
    num_examples = 10

    # Load the tokenizer. All OPT models with different sizes share the same tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b", padding_side="left")
    tokenizer.add_bos_token = False

    test_data = load_and_prepare_dataset(tokenizer, batch_size=2, pad_on_right=False, num_examples=num_examples)
    # yes_ids, no_ids = get_yes_no_ids(tokenizer)
    # higher_freq_true = run_prob_eval(test_data, tokenizer, num_examples)
    # eval_by_logits(tokenizer, test_data, True, yes_ids, no_ids)
    run_eval_with_constraints(test_data, tokenizer, num_examples)

    
if __name__ == "__main__":
    main()
