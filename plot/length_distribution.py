import argparse
import datetime
import os
import json
import dataset_loader
from methods.utils import load_base_model, load_base_model_and_tokenizer
import numpy as np
import pandas as pd
from methods.supervised import run_supervised_experiment
from methods.detectgpt import run_detectgpt_experiments
from methods.gptzero import run_gptzero_experiment
from methods.radar import run_radar
from methods.sentinel import run_sentinel
from methods.metric_based import get_ll, get_rank, get_entropy, get_rank_GLTR, run_threshold_experiment, run_GLTR_experiment
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
import pdb
import torch
import random





upper = args.upper  # 用户指定的最大token数
lower = args.lower  # 用户指定的最小token数
test_data = {'text': [], 'label': []}
train_data = {'text': [], 'label': []}

data = []

for root, dirs, files in os.walk("/scratch/jh7956/mixset_dataset"):
    for file in files:
        if args.domain in file and ("task" in file or "ground" in file):
            print("processing file: ", file)
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                for entry in json_data.values():
                    text = entry['abstract']
                    if "task1" in file:
                        label = 2
                    elif "task3" in file:
                        label = 1
                    else:
                        label = 0
                    data.append((text, label))

# shuffle the data
data = shuffle(data, random_state=42)
# 假设 data 是一个列表，包含了 (text, label) 对
texts, labels = zip(*data)  # 解压数据为两个列表

max_length = 450
interval = 50  # 定义区间长度
bins = range(50, max_length + interval, interval)  # 创建区间的边界

# 创建一个Pandas DataFrame来存储结果，索引为长度区间，列为标签
unique_labels = sorted(set(labels))  # 获取所有唯一的标签，并排序
index = pd.IntervalIndex.from_tuples([(bins[i], bins[i+1] - 1) for i in range(len(bins)-1)], closed='left')
count_matrix = pd.DataFrame(0, index=index, columns=unique_labels)

# 遍历数据，增加计数
for text, label in zip(texts, labels):
    length = len(base_tokenizer.tokenize(text))
    # 找到长度对应的区间
    for interval in count_matrix.index:
        if length in interval:
            count_matrix.at[interval, label] += 1
            break

# 输出矩阵
print(count_matrix)