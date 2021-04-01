####################################################################
####################################################################
import os
import sys
import urllib
from pprint import pprint

from dataclasses import dataclass, field
from typing import Optional, Union, List
import numpy as np
import torch

from transformers import AutoModel, AutoTokenizer, AutoModelForQuestionAnswering
from transformers.data.metrics.squad_metrics import *
from transformers import set_seed, HfArgumentParser

####################################################################
####################################################################
DATA_PATH = "./data/"


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

def parse_squad_json(squad_ver='v1.1'):
    FILE_PATH = DATA_PATH+"dev-"+squad_ver+".json"
    if not os.path.isfile(FILE_PATH):
        # download json file from web
        print("SQuAD {} file not found, try to download it...".format(squad_ver))
        url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-{}.json".format(
            squad_ver)
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
                        gold_ans = [answer['text'] for answer in qa['answers']
                                    if normalize_answer(answer['text'])]
                        if not gold_ans:
                            gold_ans = [""]
                        ques_per_paragraph.append(
                            {"question": qa["question"], "answers": gold_ans})

                data[pgraph["context"]] = ques_per_paragraph

    return data

if __name__ == "__main__":

    parser = HfArgumentParser(arguments)
    args = parser.parse_args_into_dataclasses()
    pprint (args[0].__dict__)
    ####################################################################
    
    set_seed(42)
    ####################################################################

    #data = parse_squad_json()
    

