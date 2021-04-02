####################################################################
####################################################################
import os
import sys
import urllib
from pprint import pprint
from itertools import compress

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

def parse_squad_json(squad_ver='v1.1'):
    FILE_PATH = DATA_PATH+"dev-"+squad_ver+".json"
    if not os.path.isfile(FILE_PATH):
        # download json file from web
        print("SQuAD {} file not found, try to download it...".format(squad_ver))
        url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-{}.json".format(squad_ver)
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
                        for answer in answers:
                            ques_per_paragraph.append({"question": qa["question"], "answers": answer["text"], "start_positions":answer["answer_start"], "end_positions":answer["answer_end"]})
                data[pgraph["context"]] = ques_per_paragraph

    return data

def get_qa_trainset(filter_inputs=True, single_input=True, sample_inputs=-1,):
    
    data = parse_squad_json()
    associated_data = []
    for context in data.keys():
        context_ques_pair = []
        for ques in data[context]:
            context_ques_pair.append(
                {'context': context, 'question': ques['question'], 'answers': ques['answers'], "start_positions": ques["start_positions"], "end_positions": ques["end_positions"]})
        associated_data.append(context_ques_pair)

    associated_data = sum(associated_data, [])
    # use latter 90% of the data for evaluation
    associated_data = associated_data[:int(len(associated_data)*0.1)]
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
        prediction = qa_pipeline(
            {'context': qa_pair['context'], 'question': qa_pair['question']}, max_seq_len=MAX_SEQ_LEN, att_threshold=att_threshold, hs_threshold=hs_threshold, head_mask=head_mask, quantize_att_bits=att_quant_bits, quantize_hstate_bits=hstate_quant_bits)
        em_score = max(compute_exact(prediction['answer'], gold_ans)
                       for gold_ans in qa_pair['answers'])
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
    for layer_idx in layers:
        pipeline.model.roberta.encoder.layer[layer_idx].attention.self.quantize = True
    
    return pipeline

def get_batch(fed_data, start_idx, end_idx):

    batch_data = {"context": [], "question": [], "start_positions": [], "end_positions":[], "answers": []}
    print (start_idx, end_idx, len(fed_data))
    pprint (fed_data[start_idx: min(end_idx, len(fed_data))])
    for qa_pair in fed_data[start_idx: min(end_idx, len(fed_data))]:
        
        for item in batch_data:
            batch_data[item].append(qa_pair[item])
    
    batch_data["start_positions"] = torch.tensor(batch_data["start_positions"], dtype=torch.long).to("cuda:0")
    batch_data["end_positions"] = torch.tensor(batch_data["end_positions"], dtype=torch.long).to("cuda:0")
    return batch_data


def train(qa_pipeline, fed_data):
    optim = torch.optim.AdamW(qa_pipeline.model.parameters(), lr=0.001, weight_decay=0)
    batch_size = 12
    losses = []
    res = {}

    qa_pipeline.model.to("cuda:0")

    for start in range(0, len(fed_data), batch_size):
        batch_data = get_batch(fed_data, start, start+batch_size)
        
        predictions, loss_batch = qa_pipeline(**batch_data)
        print (f"loss = {loss_batch}")
        sys.exit(0)
        #process results/metrics
        optim.zero_grad()
        loss_batch.backward()
        optim.step()
        losses.append(loss_batch.item())
        
    
    return (res, losses)

def train_quantizer(qa_pipline, num_epochs, filter_inputs=True, single_input=True, sample_inputs=-1):

    fed_data, associated_data = get_qa_trainset(filter_inputs, single_input, sample_inputs)

    losses = []
    for epoch in range(num_epochs):
        res, loss_epoch = train(qa_pipeline, fed_data)
        losses.append(loss_epoch.mean().item())
        #metrics
    
    return (losses, res, qa_pipeline)


if __name__ == "__main__":

    parser = HfArgumentParser(arguments)
    args = parser.parse_args_into_dataclasses()[0]
    pprint (args.__dict__)
    ####################################################################
    
    set_seed(42)
    ####################################################################

    data = parse_squad_json()
    qa_pipeline = pipeline(
        "question-answering",
        model="roberta-base",
        tokenizer="roberta-base",
        device=0)

    predictions = run_qa_pipeline(
            qa_pipeline, filter_inputs=False, single_input=False, \
            sample_inputs=args.samples, att_threshold=0.0, hs_threshold=0.0, \
            att_quant_bits=4.0, hstate_quant_bits=0.0)
        
    pipeline = set_quantize(layers=[11,], pipeline=qa_pipeline)
    train_quantizer(qa_pipeline, 1)
    #test_quantizer(qa_pipeline)
    

