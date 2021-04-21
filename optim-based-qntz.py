####################################################################
####################################################################
import os
import sys
import urllib
from pprint import pprint
from itertools import compress
from tqdm import tqdm

import random

from dataclasses import dataclass, field
from typing import Optional, Union, List
import numpy as np
import torch

from transformers import pipeline
from transformers.data.metrics.squad_metrics import *
from transformers import set_seed, HfArgumentParser

####################################################################
####################################################################
DATA_PATH = "./data/"
MAX_SEQ_LEN = 320
ATT_SIZE = [12, 12, MAX_SEQ_LEN, MAX_SEQ_LEN]
HS_SIZE = [ATT_SIZE[0]+1, 1, MAX_SEQ_LEN, 64*ATT_SIZE[1]]
DEVICE="cuda:1"

####################################################################
####################################################################

@dataclass
class arguments:
    samples: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Number of samples for distribution"
        }
    )

def get_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2  
    
    return answers

def parse_squad_json(squad_ver='v1.1', dev=True):
    dataset = "dev-" if dev else "train-"
    FILE_PATH = DATA_PATH+dataset+squad_ver+".json"
    if not os.path.isfile(FILE_PATH):
        # download json file from web
        print("SQuAD {} file not found, try to download it...".format(squad_ver))
        url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/{}.json".format(dataset+squad_ver)
        data = (urllib.request.urlopen(url)).read()
        with open(FILE_PATH, "wb+") as out_file:
            out_file.write(data)

    data = {}
    with open(FILE_PATH, "r", encoding="utf-8") as data_file:
        squad_raw_data = json.load(data_file)["data"]

        for topic in squad_raw_data:
            for pgraph in topic["paragraphs"]:
                ques_per_paragraph = []
                for qa in pgraph["qas"]:
                    if (squad_ver == 'v1.1') or (squad_ver == "v2.0" and not qa["is_impossible"]):
                        answers = get_end_idx(qa['answers'], [pgraph["context"],]*len(qa["answers"]))
                        gold_ans, ans_start, ans_end = [], [], []
                        for answer in answers:
                            gold_ans.append(answer["text"])
                            ans_start.append(answer["answer_start"])
                            ans_end.append(answer["answer_end"])
                        ques_per_paragraph.append({"question": qa["question"], "answers": gold_ans, "start_positions":ans_start, "end_positions":ans_end})
                data[pgraph["context"]] = ques_per_paragraph

    return data

def get_qa_trainset(filter_inputs=True, single_input=True, sample_inputs=-1, dev=True):
    
    data = parse_squad_json(dev=dev)
    associated_data = []
    for context in data.keys():
        context_ques_pair = []
        for ques in data[context]:
            context_ques_pair.append(
                {'context': context, 'question': ques['question'], 'answers': ques['answers'], "start_positions": ques["start_positions"], "end_positions": ques["end_positions"]})
        associated_data.append(context_ques_pair)

    associated_data = sum(associated_data, [])
    # use latter 90% of the data for evaluation
    #associated_data = associated_data[:int(len(associated_data)*0.1)]
    input_lens = [len(i['context']+i['question']) for i in associated_data]
    print("QA string pair length: [{}, {}]".format(min(input_lens), max(input_lens)))

    # sample several instances from all data for short test
    if sample_inputs > 0:
        random.seed(123)
        associated_data = random.sample(associated_data, sample_inputs)

    fed_data = associated_data
    # construct and apply length filter to inputs
    # TODO: parameterize the length selector
    if filter_inputs:
        len_filter = [1 if 600 <= i < 700 else 0 for i in input_lens]
        filtered_associated_data = list(compress(associated_data, len_filter))
        fed_data = filtered_associated_data
    if single_input:
        single_associated_data = [random.choice(associated_data)]
        fed_data = single_associated_data  

    return fed_data, associated_data  

def run_qa_pipeline(qa_pipeline, filter_inputs=True, single_input=True, sample_inputs=-1, att_threshold=0.0, hs_threshold=0.0, att_quant_bits=0.0, hstate_quant_bits=0.0):
    '''
    run question answering pipeline. 
    filter inputs: filter out the question-context pairs that have lengths out of 
    600-700 chars.
    sample inputs: randomly sample some question-context pairs to get all 
    raw attentions from each of them, instead of aggregrating the values
    the sample_inputs should be less than 100 to control the RAM usage under 5.89GB
    '''

    print("Running pipeline...")
    fed_data, associated_data = get_qa_trainset(filter_inputs, single_input, sample_inputs)

    # MARK: define head mask here
    head_mask = np.ones(ATT_SIZE[:2])
    head_mask[0][9], head_mask[0][11], head_mask[1][2], head_mask[7][8] = 0, 0, 0, 0
    head_mask = None
    
    res, pipeline_running_counter, fed_data_len = None, 0, len(fed_data)
    total_elem_count = 0
    print("Among all inputs {}/{} are selected.".format(fed_data_len, len(associated_data)))
    # run the prediction
    for qa_pair in fed_data:
        print("running pipeline iter {}/{}...".format(pipeline_running_counter, fed_data_len))
        prediction, loss = qa_pipeline(
            {'context': qa_pair['context'], 'question': qa_pair['question']}, max_seq_len=MAX_SEQ_LEN, att_threshold=att_threshold, hs_threshold=hs_threshold, head_mask=head_mask, quantize_att_bits=att_quant_bits, quantize_hstate_bits=hstate_quant_bits)
        #pprint ({'context': qa_pair['context'], 'question': qa_pair['question'], "answer": qa_pair['answers']})
        em_score = max(compute_exact(gold_ans, prediction['answer']) for gold_ans in qa_pair['answers'])
        att_array = prediction['attentions']
        q_prbs, k_prbs, v_prbs, scrs_prbs, att_out_prbs = prediction['pipeline_prbs']

        # aggregrate attention and hidden states
        # MARK: I am only getting values that are zero for the sparsity here. No specific sparsity bar.
        def get_spars(x, axis): 
            return x.shape[-1] ** 2 - np.count_nonzero(x[:, :, :x.shape[-1], :], axis=axis)
        def agg_func(f): return np.stack([f(i, axis=(-2, -1)) for i in att_array], axis=0)
        def add_func(f): return np.sum([f(i, axis=(-2, -1)) for i in att_array], axis=0)
        if res is None:
            res = {'score': em_score, 'hidden_states': np.zeros(HS_SIZE),
                   'max': agg_func(np.amax), 'min': agg_func(np.amin), 'mean': agg_func(np.mean),
                   'std': agg_func(np.std), 'sparsity': add_func(get_spars), 
                   'q': q_prbs, 'k': k_prbs, 'v': v_prbs, 'scrs': scrs_prbs, 'att_out': att_out_prbs}
            res['attentions'] = [] if sample_inputs > 0 else np.zeros(ATT_SIZE)
        else:
            res['score'] = (res['score'] + em_score)
            res['max'] = np.concatenate((res['max'], agg_func(np.amax)), axis=0)
            res['min'] = np.concatenate((res['min'], agg_func(np.amin)), axis=0)
            res['mean'] = np.concatenate((res['mean'], agg_func(np.mean)), axis=0)
            res['std'] = np.concatenate((res['std'], agg_func(np.std)), axis=0)
            res['sparsity'] = np.add(res['sparsity'], add_func(get_spars))
            if sample_inputs > 0:
                res['q'] += q_prbs
                res['k'] += k_prbs
                res['v'] += v_prbs
                res['scrs'] += scrs_prbs
                res['att_out'] += att_out_prbs

        # collect attentions
        if sample_inputs > 0:
            res['attentions'] += att_array
            if np.count_nonzero(res['hidden_states']) == 0: res['hidden_states'] = prediction['hidden_states']
            else: res['hidden_states'] = np.concatenate((res['hidden_states'], prediction['hidden_states']), axis=1)
        else:
            for layer_idx, (res_layer, pred_layer) in enumerate(zip(res['hidden_states'], prediction['hidden_states'])):
                res['hidden_states'][layer_idx][0] = np.add(res_layer[0], pred_layer[0])
            for att in att_array:
                padded_att = np.zeros(ATT_SIZE)
                padded_att[:, :, :att.shape[2], :att.shape[3]] = att
                # aggregrate all the results
                # unfold the tensor to 2-D array to walk around buggy numpy sum
                for layer_idx, (res_layer, pred_layer) in enumerate(zip(res['attentions'], padded_att)):
                    for head_idx, (res_head, pred_head) in enumerate(zip(res_layer, pred_layer)):
                        res['attentions'][layer_idx][head_idx] = np.add(res_head, pred_head)

        pipeline_running_counter += 1
        total_elem_count += sum([att.shape[-1] * att.shape[-1] for att in att_array])

        # if (sample_inputs > 0): 
        #     for i in res['attentions']: 
        #         if (i > len(res['attentions'])).any():
        #             idx0, idx1, idx2, idx3, idx4 = np.where(i > len(res['attentions']))
        #             print("iter {} has attention larger than 1 ({}), exist..."
        #                 .format(len(res['attentions']), (idx0[0], idx1[0], idx2[0], idx3[0], idx4[0])))
        #             exit()

        print(prediction['answer'], em_score, res['score'] / pipeline_running_counter)

    res['sparsity'] = res['sparsity'].astype(float) / total_elem_count
    res['qa_pair_len'] = fed_data_len
    return res

def set_quantize(layers, pipeline):

    #TODO: set model name based on the model you are using. 
    pipeline.model.qa_outputs.requires_grad=False
    for layer_idx in range(13):
        layer = pipeline.model.roberta.embeddings.parameters() if layer_idx == 0 else pipeline.model.roberta.encoder.layer[layer_idx-1].parameters()
        for params in layer:
            params.requires_grad=False

    for layer_idx in layers:
        pipeline.model.roberta.encoder.layer[layer_idx].attention.self.quantize = True
        for params in pipeline.model.roberta.encoder.layer[layer_idx].attention.self.quantizer.parameters():
            params.requires_grad=True
            print (params.data)
    
    return pipeline

def get_batch(fed_data, start_idx, end_idx, all_ans=False):

    batch_data = {"context": [], "question": [], "start_position_character": [], "answer_text": []}
    #print (start_idx, end_idx, len(fed_data))
    #pprint (fed_data[start_idx: min(end_idx, len(fed_data))])
    for qa_pair in fed_data[start_idx: min(end_idx, len(fed_data))]:

        batch_data["context"].append(qa_pair["context"])
        batch_data["question"].append(qa_pair["question"])
        if not all_ans:
            batch_data["start_position_character"].append(qa_pair["start_positions"][0])
        #batch_data["end_positions"].append(qa_pair["end_positions"][0])
            batch_data["answer_text"].append(qa_pair["answers"][0])
        else:
            batch_data["start_position_character"].append(qa_pair["start_positions"])
        #batch_data["end_positions"].append(qa_pair["end_positions"][0])
            batch_data["answer_text"].append(qa_pair["answers"])
        #for item in batch_data:
            #batch_data[item].append(qa_pair[item])
    
    #batch_data["start_positions"] = torch.tensor(batch_data["start_positions"], dtype=torch.long).to(DEVICE)
    #batch_data["end_positions"] = torch.tensor(batch_data["end_positions"], dtype=torch.long).to(DEVICE)
    return batch_data

def extract_ans(pred, gold_ans):

    em_score = compute_exact(gold_ans, pred['answer'])
    f1 = compute_f1(gold_ans, pred['answer'])
    return pred["answer"], em_score, f1


def train(qa_pipeline, fed_data):
    optim = torch.optim.AdamW(qa_pipeline.model.parameters(), lr=3e-5)
    batch_size = 8
    losses = []
    res = {"score":0, "num_instances":0, "preds": [], "f1":0}

    qa_pipeline.model.train()
    qa_pipeline.model.to(DEVICE)
    with tqdm(total = len(fed_data)//batch_size) as trainer:
        for start in tqdm(range(0, len(fed_data), batch_size), desc="training"):
            batch_data = get_batch(fed_data, start, start+batch_size)
            batch_data["is_training"] = True
            batch_data["max_seq_len"]=MAX_SEQ_LEN
            #batch_data["att_quant_bits"]=3.0
            predictions, loss_batch = qa_pipeline(**batch_data)
            if isinstance(predictions, List):
                for i, pred in enumerate(predictions):
                    #print (f"St: {pred['start']}, En: {pred['end']}, pr: {pred['answer']}, An: {batch_data['answer_text'][i]}")
                    answer, em_score, f1 = extract_ans(pred, batch_data["answer_text"][i])
                    res["score"] += em_score
                    res["f1"] += f1
                    res["num_instances"] += 1
                    res["preds"].append(answer)
            else:
                #print (f"St: {predictions['start']}, En: {predictions['end']}, pr: {predictions['answer']}, An: {batch_data['answer_text'][0]}")
                answer, em_score, f1 = extract_ans(predictions, batch_data["answer_text"][0])
                res["score"] += em_score
                res["f1"] += f1
                res["num_instances"] += 1
                res["preds"].append(answer)
            
            optim.zero_grad()
            loss_batch.backward()
            optim.step()
            losses.append(loss_batch.detach().cpu().item())
            trainer.set_description("Loss:%.3f, EM score:%.3f, F1: %.3f"%(np.mean(losses), res["score"]/res["num_instances"], res["f1"]/res["num_instances"]))
            #process results/metrics
            #clear cache once in 10 steps
            if len(losses)%10 == 0: torch.cuda.empty_cache()
            #print (f"loss = {loss_batch}")
    
    for layer_idx in [0,]:
        for params in qa_pipeline.model.roberta.encoder.layer[layer_idx].attention.self.quantizer.parameters():
            print (params.data)

    return (res, losses)

def evaluate(qa_pipeline, fed_data):
    batch_size = 16
    losses = []
    res = {"score":0, "num_instances":0, "preds": [], "f1":0}

    for layer_idx in [0,]:
        for params in qa_pipeline.model.roberta.encoder.layer[layer_idx].attention.self.quantizer.parameters():
            print (params.data)

    qa_pipeline.model.eval()
    with torch.no_grad():
        with tqdm(total = len(fed_data)//batch_size) as trainer:
            for start in tqdm(range(0, len(fed_data), batch_size), desc="Eval"):
                batch_data = get_batch(fed_data, start, start+batch_size, True)
                predictions, loss_batch = qa_pipeline(context=batch_data["context"], question=batch_data["question"], max_seq_len=MAX_SEQ_LEN)#, att_quant_bits=3.0)
                if isinstance(predictions, List):
                    for i, pred in enumerate(predictions):
                        #print (f"St: {pred['start']}, En: {pred['end']}, pr: {pred['answer']}, An: {batch_data['answer_text'][i]}")
                        answer, em_score, f1 = pred["answer"], max([extract_ans(pred, gold_ans)[1] for gold_ans in batch_data["answer_text"][i]]), max([extract_ans(pred, gold_ans)[2] for gold_ans in batch_data["answer_text"][i]])
                        res["score"] += em_score
                        res["f1"] += f1
                        res["num_instances"] += 1
                        res["preds"].append(answer)
                else:
                    #print (f"St: {predictions['start']}, En: {predictions['end']}, pr: {predictions['answer']}, An: {batch_data['answer_text'][0]}")
                    answer, em_score, f1 = predictions["answer"], max([extract_ans(predictions, gold_ans)[1] for gold_ans in batch_data["answer_text"][0]]), max([extract_ans(predictions, gold_ans)[2] for gold_ans in batch_data["answer_text"][0]])
                    res["score"] += em_score
                    res["f1"] += f1
                    res["num_instances"] += 1
                    res["preds"].append(answer)
                
                losses.append(loss_batch.detach().cpu().item())
                trainer.set_description("Loss:%.3f, EM score:%.3f, F1: %.3f"%(np.mean(losses), res["score"]/res["num_instances"], res["f1"]/res["num_instances"]))
                #process results/metrics
                #clear cache once in 10 steps
                if len(losses)%10 == 0: torch.cuda.empty_cache()
                #print (f"loss = {loss_batch}")
        
    return (res, losses)


def train_quantizer(qa_pipline, num_epochs, filter_inputs=True, single_input=True, sample_inputs=-1):

    tr_fed_data, _ = get_qa_trainset(filter_inputs, single_input, sample_inputs, dev=False)
    print (f"Length of train data: {len(tr_fed_data)}")
    te_fed_data, _ = get_qa_trainset(filter_inputs, single_input, 1000, dev=True)
    print (f"Length of test data: {len(te_fed_data)}")
    losses = []
    for epoch in range(num_epochs):
        res, loss_epoch = evaluate(qa_pipeline, te_fed_data)
        print (f"Epoch {epoch} Eval EM: {res['score']/res['num_instances']}")
        print (f"Epoch {epoch} Eval F1: {res['f1']/res['num_instances']}")
        res, loss_epoch = train(qa_pipeline, tr_fed_data)
        losses.append(np.array(loss_epoch).mean().item())
        print (f"Epoch {epoch} Train Loss: {losses[-1]}")
        print (f"Epoch {epoch} Train EM: {res['score']/res['num_instances']}")
        print (f"Epoch {epoch} Train F1: {res['f1']/res['num_instances']}")

    return (losses, res, qa_pipeline)


if __name__ == "__main__":

    parser = HfArgumentParser(arguments)
    args = parser.parse_args_into_dataclasses()[0]
    pprint (args.__dict__)
    ####################################################################
    
    set_seed(42)
    ####################################################################

    #train_data = get_qa_trainset(False, False, -1, False)
    #val_data = get_qa_trainset(False, False, -1, True)


    qa_pipeline = pipeline(
        "question-answering",
        model="csarron/roberta-base-squad-v1",
        tokenizer="csarron/roberta-base-squad-v1",
        device=-1
        #device=int(DEVICE[-1]),
        )

    predictions = run_qa_pipeline(
            qa_pipeline, filter_inputs=False, single_input=False, \
            sample_inputs=3, att_threshold=0.0, hs_threshold=0.0, \
            att_quant_bits=3.0, hstate_quant_bits=0.0)
    pipeline = set_quantize(layers=[0,], pipeline=qa_pipeline)
    print (qa_pipeline.model.roberta.encoder.layer[0].attention.self.quantizer.lower_bounds)
    print (qa_pipeline.model.roberta.encoder.layer[0].attention.self.quantizer.upper_bounds)
    sys.exit(0)
    train_quantizer(qa_pipeline, 10, filter_inputs=False, single_input=False, sample_inputs=args.samples)
    #test_quantizer(qa_pipeline)

    #TODO: Batching for training through pipeline
    #TODO: Support for multiple optim based methods in Quantizer