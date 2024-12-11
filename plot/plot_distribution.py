import argparse
import datetime
import os
import json
import dataset_loader
from methods.utils import load_base_model, load_base_model_and_tokenizer
import numpy as np
from methods.supervised import run_supervised_experiment
from methods.detectgpt import run_detectgpt_experiments
from methods.gptzero import run_gptzero_experiment
from methods.radar import run_radar
from methods.sentinel import run_sentinel
from methods.metric_based import get_ll, get_rank, get_entropy, get_rank_GLTR, run_threshold_experiment, run_GLTR_experiment
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pdb
import torch
import random

# python length.py --finetune --domain CS --train_sentence_length 75 --lower 50 --upper 100
# python length.py --test_only --domain CS --train_sentence_length 75 --lower 100 --upper 150


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
parser.add_argument('--test_only', action='store_true', default=False)
parser.add_argument('--train_size', type=int, default=1000)
parser.add_argument('--test_size', type=int, default=1500)
parser.add_argument('--train_sentence_length', type=int, default=75)
parser.add_argument('--test_sentence_length', type=int, default=100)
parser.add_argument('--finetune', action="store_true", default=True)
parser.add_argument('--domain', type=str, default="CS", choices=["CS", "HSS", "PHX", "All"])
parser.add_argument('--transfer', action="store_true", default=False)
parser.add_argument('--lower', type=int, default=50)
parser.add_argument('--upper', type=int, default=100)

args = parser.parse_args()


section = [[0,120],[120,300],[300,1023]]

for domain in ["CS", "HSS", "PHX"]:
    for (lower, upper) in section:
        count = 0
        for root, dirs, files in os.walk("/scratch/jh7956/mixset_dataset"):
            for file in files:
                if domain in file and ("task" in file or "ground" in file):
                    # print("processing file: ", file)
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        json_data = json.load(f)
                        for key, value in json_data.items():
                            text = value['abstract']
                            token_length = len(base_tokenizer.tokenize(text))
                            if lower <= token_length < upper:
                                count+=1
                    
        print(f"number of data in domain {domain} the period {lower}-{upper}: ", count)
                    
