from transformers import AutoTokenizer, GPT2Tokenizer, BertTokenizer, OPTForQuestionAnswering
from transformers import get_scheduler
from datasets import load_dataset
from transformers import default_data_collator
import torch 
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
import evaluate, collections
from accelerate import Accelerator, find_executable_batch_size
import numpy as np
import pandas as pd
from math import isnan
from tqdm.auto import tqdm
from pprint import pprint

max_length = 1024
stride = 128

MODEL_NAME = "facebook/opt-13b"
OPT_CACHE = "/chronos_data/tji/opt_model_cache"

def load_squad_dataset():
    raw_datasets = load_dataset("squad", cache_dir=OPT_CACHE+"/squad_dataset")
    return raw_datasets

def preprocess_training_examples(examples, tokenizer):
    '''
    from huggingface course
    '''
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation_examples(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

def compute_metrics(start_logits, end_logits, features, examples, n_best=20, max_answer_length=30):
    '''
    from huggingface course
    '''
    metric = evaluate.load("squad")
    
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

def finetune_opt_squad():
    # data load and preprocessing
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=OPT_CACHE, use_fast=True)
    raw_datasets = load_squad_dataset()

    train_dataset = raw_datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )
    len(raw_datasets["train"]), len(train_dataset)

    validation_dataset = raw_datasets["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )
    len(raw_datasets["validation"]), len(validation_dataset)

    train_dataset.set_format("torch")
    validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    validation_set.set_format("torch")
    
    accelerator = Accelerator(fp16=True)

    @find_executable_batch_size(starting_batch_size=4)
    def inner_training_loop(batch_size):
        nonlocal accelerator
        accelerator.free_memory()

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=batch_size,
        )
        eval_dataloader = DataLoader(
            validation_set, collate_fn=default_data_collator, batch_size=batch_size
        )

        model = OPTForQuestionAnswering.from_pretrained(MODEL_NAME, cache_dir=OPT_CACHE)
        optimizer = AdamW(model.parameters(), lr=2e-5)

        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        num_train_epochs = 3
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=num_training_steps,
        )

        progress_bar = tqdm(range(num_training_steps))

        for epoch in range(num_train_epochs):
            # Training
            model.train()
            for step, batch in enumerate(train_dataloader):
                batch.to(accelerator.device)
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            # Evaluation
            model.eval()
            start_logits = []
            end_logits = []
            accelerator.print("Evaluation!")
            for batch in tqdm(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
                end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

            start_logits = np.concatenate(start_logits)
            end_logits = np.concatenate(end_logits)
            start_logits = start_logits[: len(validation_dataset)]
            end_logits = end_logits[: len(validation_dataset)]

            metrics = compute_metrics(
                start_logits, end_logits, validation_dataset, raw_datasets["validation"]
            )
            print(f"epoch {epoch}:", metrics)

            # Save and upload
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                OPT_CACHE+"/opt-13b-finetuned-squad", 
                save_function=accelerator.save
            )

    
    inner_training_loop()

def eval_opt_squad():
    # data load and preprocessing
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=OPT_CACHE, use_fast=True)
    raw_datasets = load_squad_dataset()

    validation_dataset = raw_datasets["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )
    len(raw_datasets["validation"]), len(validation_dataset)

    validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    validation_set.set_format("torch")
    
    accelerator = Accelerator(fp16=True)

    eval_dataloader = DataLoader(
        validation_set, collate_fn=default_data_collator, batch_size=1
    )

    model = OPTForQuestionAnswering.from_pretrained(MODEL_NAME, cache_dir=OPT_CACHE)

    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )

    # Evaluation
    model.eval()
    start_logits = []
    end_logits = []
    accelerator.print("Evaluation!")
    for batch in tqdm(eval_dataloader[0]):
        with torch.no_grad():
            outputs = model(**batch)

        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(validation_dataset)]
    end_logits = end_logits[: len(validation_dataset)]

    metrics = compute_metrics(
        start_logits, end_logits, validation_dataset, raw_datasets["validation"]
    )
    print(metrics)

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=OPT_CACHE, use_fast=True)
    raw_datasets = load_squad_dataset()
    model = OPTForQuestionAnswering.from_pretrained(MODEL_NAME, cache_dir=OPT_CACHE)
    model.to("cuda:0")
    import time
    print("start sleeping...")
    time.sleep(10)
    print(model.named_parameters())


def main():
    # load_model()
    # finetune_opt_squad()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=OPT_CACHE, use_fast=True)
    print(tokenizer.cls_token_id)

    # eval_opt_squad()

if __name__ == "__main__":
    main()