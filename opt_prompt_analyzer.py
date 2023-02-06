from transformers import (
    AutoTokenizer,
    OPTForCausalLM,
    default_data_collator
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import evaluate
import torch

import random

OPT_CACHE = "/chronos_data/tji/opt_model_cache"

def load_and_prepare_dataset(tokenizer, num_examples=-1, split="validation", pad_on_right=True, batch_size=3, max_seq_len=1024, stride=32):
    raw_data = load_dataset("boolq", cache_dir=OPT_CACHE, split=split)

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
            max_length=max_seq_len,
            stride=stride,
            padding="max_length",
        )

        return tokenized_examples

    if num_examples > -1:
        # selected_idx = random.sample(range(len(raw_data)), num_examples)
        # raw_data = raw_data.select(selected_idx)
        raw_data = raw_data.select(range(num_examples))

    answers = raw_data[answer_column_name]

    # append sep token to question and contexts
    raw_data = raw_data.map( \
        # lambda a: {"question_wq": "\nQuestion: " + a["question"] + "? Choose between \"Yes\" and \"No\".\nAnswer: \n\n"})
        lambda a: {"question_wq": "\nBased on the above paragraph, answer the following question with \"Yes\" or \"No\": " + a["question"] + "? \nAnswer:\n\n"})

    
    tokenized_data = raw_data.map(
        tokenize_data, 
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True
    )
    tokenized_data.set_format("torch")

    tokenized_dataloader = DataLoader(
        tokenized_data, shuffle=False, collate_fn=default_data_collator, batch_size=batch_size
    )

    return tokenized_dataloader, answers


def main():
    # Load the tokenizer. All OPT models with different sizes share the same tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b", padding_side="left", cache_dir = OPT_CACHE)
    tokenizer.add_bos_token = False

    test_data, answers = load_and_prepare_dataset(tokenizer, batch_size=5, num_examples=10, pad_on_right=False)
    print(test_data)
    answers_str = [str(i) for i in answers]

    # Load the model. Alpa automatically downloads the weights to the specificed path
    model = OPTForCausalLM.from_pretrained("facebook/opt-13b", cache_dir=OPT_CACHE)

    # Generate
    model_res = []
    for step, batch in enumerate(test_data):
        output = model.generate(**batch, max_new_tokens=1, do_sample=True)

        actual_input_len = torch.count_nonzero(batch["attention_mask"], dim=-1)
        print(actual_input_len)

        generated_string = tokenizer.batch_decode(output, skip_special_tokens=True)
        for i in generated_string:
            ans_str = i.split("Answer: ")[-1]
            stripped_ans = ans_str.strip().lower()
            if "yes" == stripped_ans:
                model_res.append("True")
            elif "no" == stripped_ans:
                model_res.append("False")
            else:
                model_res.append("Neither")
            print("output: ", i)
            print("ans: ", stripped_ans)

    # f1_metric = evaluate.load("f1")
    # res_f1 = f1_metric.compute(predictions=model_res, 
    #                             references=answers)
     
    acc_metric = evaluate.load("exact_match")
    res_accu = acc_metric.compute(predictions=model_res, 
                                references=answers_str)
    
    print("exact match: ", res_accu)
    
if __name__ == "__main__":
    main()