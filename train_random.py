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

# python train_random.py --finetune --domain CS --lower 100 --upper 150
# python length.py --test_only --domain CS --lower 100 --upper 150


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
parser.add_argument('--test_only', action='store_true', default=False)
parser.add_argument('--train_size', type=int, default=1000)
parser.add_argument('--test_size', type=int, default=3000)
parser.add_argument('--finetune', action="store_true", default=True)
parser.add_argument('--domain', type=str, default="CS", choices=["CS", "HSS", "PHX", "All"])
parser.add_argument('--transfer', action="store_true", default=False)
parser.add_argument('--lower', type=int, default=50)
parser.add_argument('--upper', type=int, default=100)

args = parser.parse_args()


print(f"Start running experiment for test length range: {args.lower} - {args.upper}")

print("Arguments:")
for k, v in vars(args).items():
    print(f"{k}: {v}")
print("---------")

ckpt_dir = "./ckpt"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

cache_dir = ".cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

batch_size = args.batch_size

base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_name, cache_dir)
base_model.to(DEVICE)
base_model.eval()


total_train_size = 0
total_test_size = 0

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

# 假设 data 是一个列表，包含了 (text, label) 对
texts, labels = zip(*data)  # 解压数据为两个列表

# max_length = 400
# interval = 50  # 定义区间长度
# bins = list(range(50, max_length + interval, interval))
# bins.append(np.inf)  # 添加无穷大，表示大于 max_length 的区间

# # 创建一个Pandas DataFrame来存储结果，索引为长度区间，列为标签
# unique_labels = sorted(set(labels))  # 获取所有唯一的标签，并排序
# interval_labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-2)] + [f">{max_length}"]
# index = pd.IntervalIndex.from_tuples([(bins[i], bins[i+1]) for i in range(len(bins)-1)], closed='right')
# count_matrix = pd.DataFrame(0, index=interval_labels, columns=unique_labels)

# # 遍历数据，增加计数
# for text, label in zip(texts, labels):
#     length = len(base_tokenizer.tokenize(text))  # 计算 token 长度
#     # 找到长度对应的区间
#     for interval in index:
#         if length <= interval.right:
#             count_matrix.at[interval_labels[index.get_loc(interval)], label] += 1
#             break

# # 输出矩阵
# print(count_matrix)


# 初始化训练和测试数据的结构
train_data = {'text': [], 'label': []}
test_data = {'text': [], 'label': []}

# 转换 labels 为 numpy 数组，用于 stratify
labels = np.array(labels)

# 第一次遍历：筛选符合训练数据 token 长度要求的索引
train_candidate_indices = [i for i, text in enumerate(texts)]
train_candidate_texts = [texts[i] for i in train_candidate_indices]
train_candidate_labels = [labels[i] for i in train_candidate_indices]

# StratifiedShuffleSplit 用于选择 1000 条训练数据
sss_train = StratifiedShuffleSplit(n_splits=1, train_size=1000, random_state=42)
train_indices, _ = next(sss_train.split(train_candidate_texts, train_candidate_labels))

# 记录用作训练数据的原始索引
used_train_indices = set(train_candidate_indices[i] for i in train_indices)

# 提取训练数据
for idx in train_indices:
    text = train_candidate_texts[idx]
    label = train_candidate_labels[idx]
    tokens = base_tokenizer(text, max_length=1023, truncation=True, return_tensors="pt")
    truncated_text = base_tokenizer.decode(tokens['input_ids'][0])
    train_data['text'].append(truncated_text)
    train_data['label'].append(label)

# 第二次遍历：从未使用的数据中选择测试数据
test_candidate_indices = [i for i in range(len(texts)) if i not in used_train_indices and lower <= len(base_tokenizer.tokenize(texts[i])) <= upper]
test_candidate_texts = [texts[i] for i in test_candidate_indices]
test_candidate_labels = [labels[i] for i in test_candidate_indices]

print("Number of testing candidate samples: ", len(test_candidate_texts))

# StratifiedShuffleSplit 用于选择 3000 条测试数据
sss_test = StratifiedShuffleSplit(n_splits=1, test_size=3000, random_state=42)
_, test_indices = next(sss_test.split(test_candidate_texts, test_candidate_labels))

# 提取测试数据
for idx in test_indices:
    text = test_candidate_texts[idx]
    label = test_candidate_labels[idx]
    tokens = base_tokenizer(text, max_length=1023, truncation=True, return_tensors="pt")
    truncated_text = base_tokenizer.decode(tokens['input_ids'][0])
    test_data['text'].append(truncated_text)
    test_data['label'].append(label)

data = {"train": train_data, "test": test_data}

# examine the length of the data
print("Number of training samples: ", len(data["train"]["text"]))
print("Number of testing samples: ", len(data["test"]["text"]))


# shuffle the data
data["train"]["text"], data["train"]["label"] = shuffle(data["train"]["text"], data["train"]["label"], random_state=42)
data["test"]["text"], data["test"]["label"] = shuffle(data["test"]["text"], data["test"]["label"], random_state=42)


def ll_criterion(text): return get_ll(
    text, base_model, base_tokenizer, DEVICE)

def rank_criterion(text): return -get_rank(text,
                                            base_model, base_tokenizer, DEVICE, log=False)

def logrank_criterion(text): return -get_rank(text,
                                                base_model, base_tokenizer, DEVICE, log=True)

def entropy_criterion(text): return get_entropy(
    text, base_model, base_tokenizer, DEVICE)

def GLTR_criterion(text): return get_rank_GLTR(
    text, base_model, base_tokenizer, DEVICE)



# print("Start metrics-based methods")
# print(run_threshold_experiment(data, ll_criterion, "likelihood", test_only = args.test_only, no_auc=True, ckpt_dir=ckpt_dir))
# print(run_threshold_experiment(data, rank_criterion, "rank", test_only = args.test_only, no_auc=True, ckpt_dir=ckpt_dir))
# print(run_threshold_experiment(
#     data, logrank_criterion, "log_rank", test_only = args.test_only, no_auc=True, ckpt_dir=ckpt_dir))
# print(run_threshold_experiment(
#     data, entropy_criterion, "entropy", test_only = args.test_only, no_auc=True, ckpt_dir=ckpt_dir))
# print(run_GLTR_experiment(data, GLTR_criterion, "rank_GLTR", test_only = args.test_only, no_auc=True, ckpt_dir=ckpt_dir))
print("Running distilbert") 
print(run_supervised_experiment(data, model='distilbert-base-uncased',
            cache_dir=cache_dir, batch_size=batch_size, DEVICE=DEVICE, pos_bit=1, num_labels=3, finetune=True, test_only = args.test_only, no_auc=True, ckpt_dir=ckpt_dir, domain="CS_length75"))
# print("Running roberta")
# print(run_supervised_experiment(data, model='Hello-SimpleAI/chatgpt-detector-roberta',
#             cache_dir=cache_dir, batch_size=batch_size, DEVICE=DEVICE, pos_bit=1, num_labels=3, test_only = args.test_only, no_auc=True, ckpt_dir=ckpt_dir, finetune=True))
# print("Running radar")
# print(run_radar(data, DEVICE=DEVICE, finetune=args.finetune, no_auc=True, ckpt_dir=ckpt_dir, test_only=args.test_only, three_classes=True, domain="transfer"))


    

# save results with time stamp
log = "logs/results-{START_DATE}-{START_TIME}.json"
if not os.path.exists(os.path.dirname(log)):
    os.makedirs(os.path.dirname(log))
with open(log, "w") as wf:
    for row in outputs:
        json.dump(row, wf)
        wf.write("\n")

print(f"finished running experiment for test length range: {lower} - {upper}")
print("__________________________________")

