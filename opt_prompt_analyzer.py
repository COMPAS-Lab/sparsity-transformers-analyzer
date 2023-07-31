from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM, 
    LlamaTokenizer,
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

# MODEL_NAME = "facebook/opt-iml-max-1.3b"
# MODEL_NAME = "facebook/opt-13b"
# MODEL_NAME = "facebook/opt-350m"
# TOKENIZER_NAME = "facebook/opt-13b"
TOKENIZER_NAME = "decapoda-research/llama-7b-hf"
MODEL_NAME = "decapoda-research/llama-7b-hf"
# MODEL_NAME = "tiiuae/falcon-7b-instruct"
# TOKENIZER_NAME = "tiiuae/falcon-7b-instruct"
NUM_LAYERS = 32
OPT_CACHE = "/chronos_data/tji/.huggingface_cache/"

# torch.cuda.set_device(torch.device("cuda:3"))

# init log
log_fhandler = logging.FileHandler("opt_prompt_analyzer.log", mode="w", encoding="utf-8")
log_stdout_handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger("opt_prompt_analyzer")
logger.setLevel(logging.INFO)
logger.addHandler(log_fhandler)
logger.propagate = False
hf_logging.disable_default_handler()
hf_logging.add_handler(log_stdout_handler)

def load_and_prepare_boolq(tokenizer, num_examples=-1, split="validation", pad_on_right=True, batch_size=3, max_seq_len=1024, stride=32):
    raw_data = load_dataset("boolq", split=split)

    column_names = raw_data.column_names
    question_column_name = "question"
    context_column_name = "passage"
    answer_column_name = "answer"

    def tokenize_data(examples):
        # examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        tokenized_examples = tokenizer(
            # examples[question_column_name if pad_on_right else context_column_name],
            # examples[context_column_name if pad_on_right else question_column_name],
            # truncation="only_second" if pad_on_right else "only_first",
            examples["full"],
            truncation="only_first",
            # max_length=max_seq_len,
            stride=stride,
            # padding="max_length",
        )

        return tokenized_examples

    if num_examples > -1:
        selected_idx = random.sample(range(len(raw_data)), num_examples)
        raw_data = raw_data.select(selected_idx)
        # raw_data = raw_data.select(range(num_examples))

    # append sep token to question and contexts
    def prompt_ques(a):
        # gpt-3 style
        # ques_part = "\nQuestion: " + a["question"]
        # ans_part = "\nAnswer: "
        # yes_no_question
        # ques_part = "\n\nAnswer the following yes/no question:"
        # ans_part = a["question"] + "? Yes or no? \nAnswer: "
        # exam
        # ques_part = "EXAM\nAnswer by yes or no.\n\nDocument: "
        # ans_part = "\nQuestion: " + a["question"] + "?\nAnswer: "
        # based on the following passage
        # ques_part = "Based on the following passage, " + a["question"] + "? Answer by yes or no.\n"
        # ans_part = "\nAnswer: "
        # could you tell me
        # ques_part = "\n\nHaving read that, could you tell me " + a["question"] + "? Answer by yes or no.\n"
        # ans_part = "\nAnswer: "
        ref_int = torch.tensor([1]) if a["answer"] else torch.tensor([0])
        # return {"full": ques_part + a["passage"] + ans_part, "ans": ref_int}
    
        # use templates
        boolq_prompts = DatasetTemplates("super_glue/boolq")
        # templates: ['GPT-3 Style', 'I wonder…', 'after_reading', 'based on the following passage', 'based on the previous passage', 'could you tell me…', 'exam', 'exercise', 'valid_binary', 'yes_no_question'] 
        prompts = boolq_prompts["exam"]
        constructed = prompts.apply(a)
        # print(constructed)
        return {"full": constructed[0] + "\nAnswer:", "ans": ref_int}


    raw_data = raw_data.map(
        prompt_ques
    )
    
    tokenized_data = raw_data.map(
        tokenize_data, 
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True
    )
    tokenized_data.set_format("torch")
    print(tokenized_data)

    num_true = torch.sum(torch.tensor([i["ans"] for i in tokenized_data]))
    num_false = len(tokenized_data) - num_true

    logger.info(f"dataset: {num_true} true ans, {num_false} false ans.")

    tokenized_dataloader = DataLoader(
        tokenized_data, collate_fn=default_data_collator, batch_size=batch_size, shuffle=True
    )

    random.seed = 1722163
    torch.manual_seed = 1722163
    torch.cuda.manual_seed = 1722163
    torch.backends.cudnn.deterministic = True

    return tokenized_dataloader

def load_and_prepare_stance_det(tokenizer, topic, num_examples=-1, split="test", pad_on_right=True, batch_size=1, max_seq_len=2048, use_promptsource=False):
    dataset_path = "/chronos_data/tji/.huggingface_cache/datasets/twitter_stance/"
    test_dat = dataset_path + f"stance_{topic}_test_with_history_v2.csv"
    train_dat = dataset_path + f"stance_{topic}_train_with_history_v2.csv"
    raw_data = load_dataset("csv", data_files={"test": test_dat}, split=split)
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
        st_str_list = {-1: "against", 1: "favor", 0:"none"}
        st_list = {-1: 1, 1: 2, 0: 0}

        if not use_promptsource:
            # text = f"Task: Based on the tweets below, is the person's stance favor, none or against the topic {topic}? " + \
            #         f"Answer \"favor\", \"none\" or \"against\". \n" + \
            #         f"Tweet: {text_his} \n" + \
            #         f"{text_anchor} \n" + \
            #         f"Answer: "
            # text = f"Tweet history: {text_his} \n" + \
            #         f"Anchor Tweet: {text_anchor} \n" + \
            #         f"Question: Based on the history of tweets from a person, are their stance pro or against {topic}? Answer pro or against. \n" + \
            #         "Answer:\n\n"
            text = f"Tweets: {text}\n" + \
                    f"Question: In the tweets above, what is the author's stance on the {topic}, neutral, against or in favor?\n" + \
                    "Answer:\n\n"
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
            return {"text": text, "ans": inst["stance"]}

    column_names = insts.column_names
    insts = insts.map(replace_tweet, remove_columns=column_names)

    # use promptsource:
    if use_promptsource:
        datTemplate = DatasetTemplates("tweet_eval/stance_abortion")
        # available templates: 
        # ['abortion', 'abortion_guess_passive', 'abortion_guess_passive_author', 'abortion_how_describe', 'abortion_option', 'abortion_predict_stance']
        prompt = datTemplate["abortion_guess_passive_author"]
        insts = insts.map(lambda example: {
            "text": "Question: " + prompt.apply(example, truncate=False)[0] + " \nAnswer: "})
        column_names = insts.column_names

    tokenized_data = insts.map(
        lambda example: tokenizer(example["text"]), 
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=True
    )
    tokenized_data.set_format("torch")

    tokenized_data = tokenized_data.filter(lambda example: 1024 < example["input_ids"].size(0) < max_seq_len)
    if num_examples > -1 and num_examples < len(tokenized_data):
        selected_idx = random.sample(range(len(tokenized_data)), num_examples)
        raw_data = raw_data.select(selected_idx)

    tokenized_dataloader = DataLoader(
        tokenized_data, collate_fn=default_data_collator, batch_size=batch_size, shuffle=True
    )
    random.seed = 1722163
    torch.manual_seed = 1722163
    torch.cuda.manual_seed = 1722163
    torch.backends.cudnn.deterministic = True

    return tokenized_dataloader

def load_and_prepare_hotpotqa(tokenizer, num_examples=-1, split="test", batch_size=1, max_seq_len=2048, use_promptsource=False):
    raw_data = load_dataset("hotpot_qa", "fullwiki", split=split)
    insts = raw_data.filter(lambda example: len(example["context"]["sentences"]) > 0)

    def prepare_context(inst):
        text = inst["context"]["sentences"]
        text = reduce(lambda x,y: x+y, text)
        text = " ".join(text)
        ques = inst["question"]

        # context = f"Text: {text}\n" + \
        #             f"Based on the text above, answer the question: {ques} \n" + \
        #             f"Answer: \n\n"
        context = f"Task: \nAnswer the question according to the given text. \n" + \
                    f"Text: \n{text}\n" + \
                    f"Question: \n{ques}\n" + \
                    f"Answer: \n"

        return {"text": context, "ans": [ord(c) for c in inst["answer"]]}

    column_names = insts.column_names
    insts = insts.map(prepare_context, remove_columns=column_names)

    # # use promptsource:
    # if use_promptsource:
    #     datTemplate = DatasetTemplates("tweet_eval/stance_abortion")
    #     # available templates: 
    #     # ['abortion', 'abortion_guess_passive', 'abortion_guess_passive_author', 'abortion_how_describe', 'abortion_option', 'abortion_predict_stance']
    #     prompt = datTemplate["abortion_guess_passive_author"]
    #     insts = insts.map(lambda example: {
    #         "text": "Question: " + prompt.apply(example, truncate=False)[0] + " \nAnswer: "})
    #     column_names = insts.column_names

    tokenized_data = insts.map(
        lambda example: tokenizer(example["text"]), 
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=True
    )
    tokenized_data.set_format("torch")

    tokenized_data = tokenized_data.filter(lambda example: 1024 < example["input_ids"].size(0) < max_seq_len)
    if num_examples > -1 and num_examples < len(tokenized_data):
        selected_idx = random.sample(range(len(tokenized_data)), num_examples)
        raw_data = raw_data.select(selected_idx)

    tokenized_dataloader = DataLoader(
        tokenized_data, collate_fn=default_data_collator, batch_size=batch_size, shuffle=True
    )
    random.seed = 1722163
    torch.manual_seed = 1722163
    torch.cuda.manual_seed = 1722163
    torch.backends.cudnn.deterministic = True

    return tokenized_dataloader

def load_and_prepare_rte(tokenizer, num_examples=-1, split="validation", pad_on_right=True, batch_size=3, max_seq_len=1024, stride=32):
    raw_data = load_dataset("SetFit/rte", split=split)
    column_names = raw_data.column_names

    def tokenize_data(examples):
        # examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        tokenized_examples = tokenizer(
            examples["full"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_len,
            stride=stride,
            padding="max_length",
        )

        return tokenized_examples

    if num_examples > -1:
        selected_idx = random.sample(range(len(raw_data)), num_examples)
        raw_data = raw_data.select(selected_idx)
        # raw_data = raw_data.select(range(num_examples))

    logger.info(f"dataset size: {len(raw_data)}")
    
    # append sep token to question and contexts
    def prompt_ques(a):
        s1 = "Statement 1: " + a["text1"] + "\n"
        s2 = "Statement 2: " + a["text2"] + "\n"
        ques_part = "Question: Is it true or false that statement 1 entails statement 2? \n"
        ans_str = "Answer:\n\n"
        label_in_bool = False if a["label"] == 1 else True
        return {"full": s1 + s2 + ques_part + ans_str, "answer": [label_in_bool]}

    raw_data = raw_data.map(
        prompt_ques
    )

    # explore the dataset:
    data_with_true = raw_data.filter(lambda a: a["answer"] == [True])
    data_with_false = raw_data.filter(lambda a: a["answer"] == [False])
    logger.info(f"true in dataset: {len(data_with_true)}, false: {len(data_with_false)}")
    
    tokenized_data = raw_data.map(
        tokenize_data, 
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True
    )
    tokenized_data.set_format("torch")
    print(tokenized_data[3])

    tokenized_dataloader = DataLoader(
        tokenized_data, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size
    )

    return tokenized_dataloader

def load_and_prepare_winogrande(tokenizer, num_examples=-1, split="validation", pad_on_right=True, batch_size=3, max_seq_len=1024, stride=32):
    raw_data = load_dataset("winogrande", "winogrande_l", split=split)

    def tokenize_data(examples):
        # examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        tokenized_examples = tokenizer(
            examples["full"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_len,
            stride=stride,
            padding="max_length",
        )

        return tokenized_examples

    if num_examples > -1:
        selected_idx = random.sample(range(len(raw_data)), num_examples)
        raw_data = raw_data.select(selected_idx)
        # raw_data = raw_data.select(range(num_examples))

    logger.info(f"dataset size: {len(raw_data)}")
    
    # append sep token to question and contexts
    def prompt_ques(a):
        statement = " In the previous sentence, does _ refer to "
        op1 = a["option1"]
        op2 = a["option2"]
        return {"full": a["sentence"] + statement + op1 + " or " + op2 + "?\n", "answer": int(a["answer"])}

    raw_data = raw_data.map(
        prompt_ques
    )

    # explore the dataset:
    data_with_1 = raw_data.filter(lambda a: a["answer"] == 1)
    data_with_2 = raw_data.filter(lambda a: a["answer"] == 2)
    logger.info(f"1 in dataset: {len(data_with_1)}, 2: {len(data_with_2)}")
    
    tokenized_data = raw_data.map(
        tokenize_data, 
        batched=True,
        remove_columns=["sentence", "option1", "option2"],
        load_from_cache_file=False
    )
    tokenized_data.set_format("torch")
    print(tokenized_data[0])

    tokenized_dataloader = DataLoader(
        tokenized_data, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size
    )

    return tokenized_dataloader

def load_and_prepare_c4(tokenizer, num_examples=-1, split="validation", pad_on_right=False, batch_size=3, max_seq_len=1024, stride=32):
    # raw_data = load_dataset("c4", "en", split=split)
    raw_data = load_dataset("ola13/small-c4", split=split)

    def tokenize_data(examples):
        # examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        tokenized_examples = tokenizer(
            examples["text"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_len,
            stride=stride,
            padding="do_not_pad",
        )

        return tokenized_examples

    if num_examples > -1:
        selected_idx = random.sample(range(len(raw_data)), num_examples)
        raw_data = raw_data.select(selected_idx)
        # raw_data = raw_data.select(range(num_examples))

    logger.info(f"dataset size: {len(raw_data)}")
    
    tokenized_data = raw_data.map(
        tokenize_data, 
        batched=True,
        remove_columns=["url", "text", "timestamp"]
    )
    tokenized_data.set_format("torch")

    tokenized_dataloader = DataLoader(
        tokenized_data, collate_fn=default_data_collator, batch_size=batch_size, shuffle=True
    )

    return tokenized_dataloader

def infer_ans_directly(prompt_ans_dict: dict, device, true_str="yes", false_str="no"):
    model_res, ref = \
        torch.tensor([]).to(device), torch.tensor([]).to(device)
    
    generated_string = prompt_ans_dict.get("generated_string", None)
    answers = prompt_ans_dict.get("answers", None)
    # print(f"generated: {generated_string}")

    tc_true = torch.tensor([1])
    tc_false = torch.tensor([0])
    
    num_avaliable_ans = 0
    if generated_string is None or answers is None:
        raise TypeError("Invalid answers to infer ans directly!")
    else:
        for i, answer in zip(generated_string, answers):
            ans_str = i.split("\nAnswer:")[-1].lower()
            re_code = r"[_|\W]*(" + true_str + r"|" + false_str + r")[_|\W]*"
            re_code = r"[_|\W]*(yes|true|no|false)[_|\W]*"
            stripped_ans = "".join(re.findall(re_code, ans_str))
            print(f"output: {ans_str} extracted: {stripped_ans}")

            # if true_str == stripped_ans:
            if stripped_ans in ["yes", "true"]:
                model_res = torch.cat((model_res, tc_true.to(device)))
                num_avaliable_ans += 1
            # elif false_str == stripped_ans:
            elif stripped_ans in ["no", "false"]:
                model_res = torch.cat((model_res, tc_false.to(device)))
                num_avaliable_ans += 1
            else:
                tc_inverted_ref = 0 if answer.item() else 1
                tc_inverted_ref = torch.tensor([tc_inverted_ref]).to(device)
                model_res = torch.cat((model_res, tc_inverted_ref))

        ref = torch.cat((ref, answer.to(device)))

    return model_res, ref, torch.tensor([num_avaliable_ans]).to(device)

def infer_by_logits_boolq(prompt_ans_dict: dict, device):

    model_res, ref = \
        torch.tensor([]).to(device), torch.tensor([]).to(device)
    # model_res_avaliable_ans_only, ref_ans_only = \
    #     torch.tensor([]).to(device), torch.tensor([]).to(device)
        
    out_strs = prompt_ans_dict.get("out_strs", None)
    transition_probs = prompt_ans_dict.get("transition_probs", None)
    answers = prompt_ans_dict.get("answers", None)
    yes_ids = prompt_ans_dict.get("yes_ids", None)
    no_ids = prompt_ans_dict.get("no_ids", None)

    num_avaliable_ans = 0

    for generated_str, probability, answer in zip(out_strs, transition_probs, answers):
        ans_str = generated_str.split("\nAnswer:")[-1].lower()
        ref_ans = torch.tensor([answer]).to(ref.device)
        
        re_code = r"[_|\W]*(yes|true|no|false)[_|\W]*"
        stripped_ans = "".join(re.findall(re_code, ans_str))
        print(f"output: {generated_str} extracted: {stripped_ans}")

        tc_true = torch.tensor([True])
        tc_false = torch.tensor([False])
        
        if stripped_ans in ["yes", "true"]:
            model_res = torch.cat((model_res, tc_true.to(model_res.device)))
            # model_res_avaliable_ans_only = \
            #     torch.cat((model_res_avaliable_ans_only, tc_true.to(model_res_avaliable_ans_only.device)))
            # ref_ans_only = torch.cat((ref_ans_only, ref_ans))
            num_avaliable_ans += 1
        elif stripped_ans in ["no", "false"]:
            model_res = torch.cat((model_res, tc_false.to(model_res.device)))
            # model_res_avaliable_ans_only = \
            #     torch.cat((model_res_avaliable_ans_only, tc_false.to(model_res_avaliable_ans_only.device)))
            # ref_ans_only = torch.cat((ref_ans_only, ref_ans))
            num_avaliable_ans += 1
        else:
            yes_prob_sum = sum([probability[idx] for idx in yes_ids])
            no_prob_sum = sum([probability[idx] for idx in no_ids])
            prob_res = tc_true if yes_prob_sum > no_prob_sum else tc_false
            print(f"no valid answer, logit {prob_res.item()} wins")
            model_res = torch.cat((model_res, prob_res.to(model_res.device)))

        ref = torch.cat((ref, ref_ans))

    return model_res, ref, torch.tensor([num_avaliable_ans]).to(device)

def infer_by_logits_stance_det(prompt_ans_dict: dict, device):

    model_res, ref = \
        torch.tensor([]).to(device), torch.tensor([]).to(device)
    # model_res_avaliable_ans_only, ref_ans_only = \
    #     torch.tensor([]).to(device), torch.tensor([]).to(device)
        
    out_strs = prompt_ans_dict.get("out_strs", None)
    answers = prompt_ans_dict.get("answers", None)
    num_avaliable_ans = 0

    for generated_str, answer in zip(out_strs, answers):
        ans_str = generated_str.split("Answer:")[-1].lower()
        ref_ans = torch.tensor([answer]).to(ref.device)
        
        re_code = r"[_|\W]*(against|none|neutral|favor|pro)[_|\W]*"
        stripped_ans = "".join(re.findall(re_code, ans_str))
        print(f"output: {ans_str} extracted: {stripped_ans}")

        tc_against = torch.tensor([-1])
        tc_none = torch.tensor([0])
        tc_favor = torch.tensor([1])
        tc_not_included = torch.tensor([20])
        
        if stripped_ans in ["against"]:
            model_res = torch.cat((model_res, tc_against.to(model_res.device)))
            num_avaliable_ans += 1
        elif stripped_ans in ["none", "neutral"]:
            model_res = torch.cat((model_res, tc_none.to(model_res.device)))
            num_avaliable_ans += 1
        elif stripped_ans in ["favor", "pro"]:
            model_res = torch.cat((model_res, tc_favor.to(model_res.device)))
            num_avaliable_ans += 1
        else:
            model_res = torch.cat((model_res, tc_not_included.to(model_res.device)))

        ref = torch.cat((ref, ref_ans))

    return model_res, ref, torch.tensor([num_avaliable_ans]).to(device)

def infer_directly_hotpot_qa(prompt_ans_dict: dict, device):

    model_res, ref = \
        torch.tensor([]).to(device), torch.tensor([]).to(device)
    # model_res_avaliable_ans_only, ref_ans_only = \
    #     torch.tensor([]).to(device), torch.tensor([]).to(device)
        
    out_strs = prompt_ans_dict.get("out_strs", None)
    answers = prompt_ans_dict.get("answers", None).tolist()
    num_avaliable_ans = 0

    for generated_str, answer in zip(out_strs, answers):
        ans_str = generated_str.split("Answer:")[-1].lower()
        ref_ans = "".join([chr(i) for i in answer])
        print(f"generated: {generated_str}")
        print(f"output: {ans_str}, ref: {ref_ans}")

        tc_correct = torch.tensor([0])
        tc_incorret = torch.tensor([1])
        
        if ref_ans in ans_str:
            model_res = torch.cat((model_res, tc_correct.to(model_res.device)))
        else:
            model_res = torch.cat((model_res, tc_incorret.to(model_res.device)))

    return model_res, ref, torch.tensor([num_avaliable_ans]).to(device)

def run_prob_eval(test_data, tokenizer, num_examples):
    # Load the model. Alpa automatically downloads the weights to the specificed path
    model = OPTForCausalLM.from_pretrained(MODEL_NAME)

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
        load_path=None,
        device="cuda:0", 
        is_forcing_words=False,
        ATTN_SAMPLE_PATH="/chronos_data/tji/.huggingface_cache/transformers/"
        ):
    '''
    Examine only yes or no answers
    '''
    accelerator = Accelerator(mixed_precision="fp16")

    # Load the model.
    if load_path:
        print("loading local model ", load_path)
        if "llama" in MODEL_NAME:
            model = LlamaForCausalLM.from_pretrained(load_path)
        elif "opt" in MODEL_NAME:
            model = OPTForCausalLM.from_pretrained(load_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(load_path)

        extract_param_names(load_path)
        # examine if weights have sparsity:
        wq_layer1_sparsity = get_mat_sparsity(model.model.layers[0].self_attn.q_proj.weight.data)
        print("wq layer 0 sparsity: ", wq_layer1_sparsity)
    else:
        print("loading online model")
        if "llama" in MODEL_NAME:
            model = LlamaForCausalLM.from_pretrained(MODEL_NAME)
        elif "opt" in MODEL_NAME:
            model = OPTForCausalLM.from_pretrained(MODEL_NAME)
        else:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code = True)
    
    model = model.to(accelerator.device)
    model, test_data = accelerator.prepare(model, test_data)

    # Generate
    force_words_ids = None
    if is_forcing_words:
        single_word_tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_NAME, add_prefix_space=True)
        force_words = ["yes", "no"]
        force_words_ids = single_word_tokenizer(force_words, add_special_tokens=False).input_ids
        force_words_ids = [[force_words_ids[0][0], force_words_ids[1][0]]]
        print(force_words_ids)

    num_examples = 0

    model_res_all, ref_all = [], []
    attn_sparsities_all, attn_sparsities_layer_all = [], []
    num_valid_ans_all = []

    metric_all_acc = evaluate.load("accuracy")
    metric_all_f1 = evaluate.load("f1")
    metric_all_rocauc = evaluate.load("roc_auc")

    model.eval()
    for step, batch in tqdm(enumerate(test_data), total=len(test_data)):
        batch = move_to(batch, accelerator.device)
        num_beams = 2
        batch_size = len(batch["input_ids"])
        gen_params = {
            "inputs": batch["input_ids"], 
            "attention_mask": batch["attention_mask"],
            "max_new_tokens": 8,
            # "output_attentions": True,
            "output_scores": True,
            "return_dict_in_generate": True,
            "num_beams": num_beams,
            "num_beam_groups": 1,
            }
        if is_forcing_words:
            gen_params.update({
                "force_words_ids": [force_words_ids],
                "do_sample": False
            })
        
        output = model.generate(**gen_params)
        seq_ids = output.sequences
        transition_probs = nn.functional.softmax(output.scores[0], dim=-1)

        ori_prompt = tokenizer.batch_decode(seq_ids)
        generated_string = tokenizer.batch_decode(seq_ids, skip_special_tokens=True)
        out_strs = []
        for out_inst in ori_prompt:
            out_strs.append(out_inst.split("Answer:\n")[-1])

        # all_attens = output.attentions[0]
        
        # check attention
        # expected atten size: layer_size, num_beamsxbatch_sizexhead_size, len, len
        # attens = [i.to("cpu") for i in all_attens]
        # attens = torch.stack(attens)
        # print(attens.size())
        # if "llama" in MODEL_NAME:
        #     layer_size, _, head_size, seq_len, _ = attens.size()
        #     attens = attens.view(layer_size, head_size, 
        #                         num_beams*batch_size, seq_len, seq_len)
        # elif "opt" in MODEL_NAME:
        #     layer_size, head_size, seq_len, _ = attens.size()
        #     head_size = head_size // (num_beams*batch_size)
        #     attens = attens.view(layer_size, head_size, 
        #                         num_beams*batch_size, seq_len, seq_len)
                        
        # attn_sparsities = torch.tensor([]).to(accelerator.device)
        # attn_sparsities_layer = torch.tensor([]).to(accelerator.device)
        # for i in range(num_beams*batch_size):
        #     actual_input_len = torch.count_nonzero(batch["attention_mask"][i//num_beams], dim=-1).item()
        #     curr_attens = torch.squeeze(attens[:,:,i,-actual_input_len:,-actual_input_len:])
        #     # model_name = MODEL_NAME.split("/")[1]
        #     # attn_path = f"{ATTN_SAMPLE_PATH}/{model_name}-attsample/attn_s{step}b{i}.pt"
        #     # print(f"saving attn to {attn_path}...")
        #     # torch.save(curr_attens, attn_path)

        #     curr_sparsity = torch.tensor([get_mat_sparsity(curr_attens, causal_mask=True)])
        #     attn_sparsities = \
        #         torch.cat((attn_sparsities, curr_sparsity.to(accelerator.device)), dim=-1)
        #     curr_spar_layer = get_mat_sparsity(curr_attens, causal_mask=True, per_layer=True)
        #     attn_sparsities_layer = \
        #         torch.cat((attn_sparsities_layer, curr_spar_layer.to(accelerator.device)), dim=-1)
        
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

        model_res, ref, num_valid_ans = eval_method({
                                            "generated_string": generated_string, 
                                            "answers": batch["ans"],
                                            "out_strs": out_strs,
                                            "transition_probs": transition_probs,
                                            "yes_ids": yes_ids,
                                            "no_ids": no_ids,
                                            },
                                        device=accelerator.device)
        
        num_examples += batch_size

        # attn_sparsities = accelerator.pad_across_processes(attn_sparsities, dim=1, pad_index=-100)
        # attn_sparsities_gathered = accelerator.gather_for_metrics(attn_sparsities).cpu().numpy()

        model_res_all.append(accelerator.gather(model_res).cpu().numpy())
        ref_all.append(accelerator.gather(ref).cpu().numpy())
        # attn_sparsities_all.append(accelerator.gather(attn_sparsities).cpu().numpy())
        # attn_sparsities_layer_all.append(accelerator.gather(attn_sparsities_layer).cpu().numpy())
        num_valid_ans_all.append(accelerator.gather(num_valid_ans).cpu().numpy())

    print("gathering finished")
    
    # logger.info(f"attn_sparsities, {attn_sparsities}")
    # avg_sparsity = np.mean(attn_sparsities)
    # logger.info(f"avg sparsity: {avg_sparsity}")

    model_res_all = np.concatenate(model_res_all)
    ref_all = np.concatenate(ref_all)
    # attn_sparsities_all = np.concatenate(attn_sparsities_all)
    # attn_sparsities_layer_all = np.concatenate(attn_sparsities_layer_all)
    # attn_sparsities_layer_all = attn_sparsities_layer_all.reshape(NUM_LAYERS, -1)
    num_valid_ans_all = np.concatenate(num_valid_ans_all)

    if accelerator.is_main_process:
        logger.info(f"res: {model_res_all}\nref: {ref_all}")
        
        avg_sparsity = np.mean(attn_sparsities_all)
        logger.info(f"avg sparsity: {avg_sparsity}")

        # print("layer sparsities shape: ", attn_sparsities_layer_all.shape)
        # avg_layer_spars = np.mean(attn_sparsities_layer_all, axis=-1)
        # for l in range(NUM_LAYERS):
        #     logger.info(f"{avg_layer_spars[l]}")

        res_em = metric_all_acc.compute(predictions=model_res_all, references=ref_all)
        logger.info(f"acc for all: {res_em}")

        res_f1_all = metric_all_f1.compute(predictions=model_res_all, references=ref_all, average="micro")
        logger.info(f"f1 for all: {res_f1_all}")

        # res_rocauc_all = metric_all_rocauc.compute(prediction_scores=model_res_all, references=ref_all)
        # logger.info(f"roc auc: {res_rocauc_all}")

        logger.info(f"number of valid ans: {np.sum(num_valid_ans_all)}")

def finetune(
        train_data, 
        eval_data,
        tokenizer,
        device="cuda:0",
        ):
    '''
    Examine only yes or no answers
    '''
    accelerator = Accelerator(fp16=True)

    # Generate
    force_words_ids = None
    model_res_all, ref_all = [], []
    attn_sparsities_all = []
    num_valid_ans_all = []

    metric_all_acc = evaluate.load("accuracy")
    metric_all_f1 = evaluate.load("f1")
    metric_all_rocauc = evaluate.load("roc_auc")

    def inner_training_loop():
        nonlocal accelerator, train_data, eval_data
        accelerator.free_memory()

        # Load the model.
        model = LlamaForCausalLM.from_pretrained(MODEL_NAME)
        optimizer = AdamW(model.parameters(), lr=2e-5)

        model, optimizer, train_data, eval_data = accelerator.prepare(
            model, optimizer, train_data, eval_data
        )

        num_train_epochs = 3
        num_update_steps_per_epoch = len(train_data)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=num_training_steps,
        )

        progress_bar = tqdm(range(num_training_steps))
        TRUE_ID, FALSE_ID = 1565, 2089

        for epoch in range(num_train_epochs):
            # Training
            model.train()
            for step, batch in enumerate(train_data):
                batch = move_to(batch, accelerator.device)
                num_beams = 1
                batch_size = len(batch["input_ids"])
                print("input size: ", batch["input_ids"].size())
                model_params = {
                    "input_ids": batch["input_ids"], 
                    "attention_mask": batch["attention_mask"],
                    "labels": batch["input_ids"],
                    "output_attentions": False,
                    "output_hidden_states": True,
                    "return_dict": True,
                    }
                outputs = model(**model_params)
                transition_probs = nn.functional.softmax(outputs.logits[:,-1,:], dim=-1)
                # form target from ans
                vocab_size = tokenizer.vocab_size
                assert vocab_size == transition_probs.size()[1], f"vocab size = {transition_probs.size()} incorrect"

                formed_target = []
                for i, i_ans in enumerate(batch["ans"]):
                    if i_ans == 1:
                        formed_target.append(TRUE_ID)
                    else:
                        formed_target.append(FALSE_ID)
                formed_target = torch.tensor(formed_target, device = transition_probs.device, dtype=torch.long)
                # compute loss
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(transition_probs, formed_target)
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            model.eval()
            if eval_data is not None:
                model.eval()
                model_poss_res = []
                for batch in tqdm(eval_data):
                    model_params = {
                        "input_ids": batch["input_ids"], 
                        "attention_mask": batch["attention_mask"],
                        "output_attentions": False,
                        "output_hidden_states": True,
                        "return_dict": True,
                        }
                    with torch.no_grad():
                        outputs = model(**model_params)

                    transition_probs = nn.functional.softmax(outputs.logits[:,-1,:], dim=-1)
                    transition_probs = transition_probs.view(-1, tokenizer.vocab_size)

                    for inst in transition_probs:
                        if inst[TRUE_ID] > inst[FALSE_ID]:
                            model_poss_res.append(1)
                        else:
                            model_poss_res.append(0)

            print(f"epoch {epoch}")
            accelerator.wait_for_everyone()

            # Save and upload
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                "/chronos_data/tji/.huggingface_cache/transformers/llama-7b-hf-qkv-bfp12/",
                save_function = accelerator.save
                )
            
    inner_training_loop()

def get_str_ids(tokenizer, word: str):
    vocab = tokenizer.get_vocab()
    re_code = r"[_|\W]*(" + word + r")[_|\W]*"
    word_idx = []
    
    for key, val in vocab.items():
        lower_key = key.lower()
        yes_match_res = re.fullmatch(re_code, lower_key)

        if yes_match_res:
            word_idx.append(val)

    return word_idx

def get_yes_no_ids(tokenizer):
    vocab = tokenizer.get_vocab()
    re_yes_code = r"[_|\W]*(yes|true)[_|\W]*"
    re_no_code = r"[_|\W]*(no|not|false)[_|\W]*"

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
    ## examine tokenization
    # tokenizer_7b = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    # tokenizer_30b = LlamaTokenizer.from_pretrained("decapoda-research/llama-30b-hf")

    # assert(len(tokenizer_7b.get_vocab()), len(tokenizer_30b.get_vocab()))
    # for a, b in zip(tokenizer_7b.get_vocab(), tokenizer_30b.get_vocab()):
    #     tok_7b = tokenizer_7b.get_vocab()[a]
    #     tok_30b = tokenizer_30b.get_vocab()[b]
    #     if tok_7b != tok_30b:
    #         print(f"7b tok: {a} - {tok_7b}, 30b tok: {b} - {tok_30b}")

    # finetuning and eval
    num_examples = 100    # Load the tokenizer. All OPT models with different sizes share the same tokenizer
    if "llama" in MODEL_NAME:
        tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_NAME)
    elif "opt" in MODEL_NAME:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    else:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    tokenizer.add_bos_token = False

    # finetuning
    # train_data = load_and_prepare_c4(tokenizer, batch_size=1, pad_on_right=False, num_examples=num_examples, split="train")
    # finetune(train_data, None, tokenizer)

    # eval
    test_data = load_and_prepare_hotpotqa(tokenizer, split="validation", batch_size=1, num_examples=num_examples)
    # test_data = load_and_prepare_stance_det(tokenizer, "abortion",  batch_size=1, num_examples=num_examples)
    # test_data = load_and_prepare_boolq(tokenizer, batch_size=1, pad_on_right=False, num_examples=num_examples)
    # test_data = load_and_prepare_rte(tokenizer, batch_size=1, pad_on_right=False, num_examples=num_examples, split="validation")
    # test_data = load_and_prepare_winogrande(tokenizer, batch_size=1, pad_on_right=True, num_examples=num_examples, split="validation")

    yes_ids, no_ids = get_yes_no_ids(tokenizer)
    # higher_freq_true = run_prob_eval(test_data, tokenizer, num_examples)
    run_eval_with_constraints(
        test_data, 
        tokenizer, 
        eval_method=infer_directly_hotpot_qa, 
        yes_ids=yes_ids, no_ids=no_ids, 
        is_forcing_words=False,
        # load_path = "/chronos_data/tji/.huggingface_cache/transformers/llama-7b-hf-sparsegpt-bfp",
        # load_path = "/chronos_data/tji/.huggingface_cache/transformers/llama-7b-hf-selfattnonly-sparsegpt-bfp",
        # load_path = "/chronos_data/tji/.huggingface_cache/transformers/llama-13b-hf-sparsegpt",
        # load_path = "/chronos_data/tji/.huggingface_cache/transformers/llama-13b-hf-sparsegpt-bfp",
    )

    
if __name__ == "__main__":
    main()
