import argparse
import datetime
import os
import json
import dataset_loader
from methods.utils import load_base_model, load_base_model_and_tokenizer
import numpy as np
import pandas as pd
from methods.supervised import run_supervised_experiment
from methods.metric_based import get_ll, get_rank, get_entropy, get_rank_GLTR, run_threshold_experiment, run_GLTR_experiment
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
import pdb
import torch
import random
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
parser.add_argument('--test_only', action='store_true', default=False)
parser.add_argument('--train_size', type=int, default=1000)
parser.add_argument('--test_size', type=int, default=3000)
parser.add_argument('--finetune', action="store_true", default=True)
parser.add_argument('--domain', type=str, default="CS", choices=["CS", "HSS", "PHX", "All"])
parser.add_argument('--transfer', action="store_true", default=False)
parser.add_argument('--small_percentage', type=float, default=0.5)
parser.add_argument('--medium_percentage', type=float, default=0.4)
parser.add_argument('--large_percentage', type=float, default=0.1)

args = parser.parse_args()

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
thresholds = {
    'small': (0, 150),
    'medium': (150, 300),
    'large': (300, 400)
}

# if there is no data, create the data
if not os.path.exists("/scratch/jh7956/mixset_dataset/1122_data.json"):
    base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_name, cache_dir)
    base_model.to(DEVICE)
    base_model.eval()
    data = []
    for root, dirs, files in os.walk("/scratch/jh7956/mixset_dataset"):
        for file in files:
            if args.domain in file and ("task" in file or "ground" in file):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                    for entry in json_data.values():
                        text = entry["abstract"]
                        label = 2 if "task1" in file else 1 if "task3" in file else 0
                        data.append((text, label))
    data = shuffle(data, random_state=42)

    length_data = defaultdict(list)
    for text, label in data:
        length = len(base_tokenizer(text)['input_ids'])
        for key, value in thresholds.items():
            if value[0] <= length < value[1]:
                length_data[key].append((text, label))
                break

    data = {
        "train_small": {"text": [], "label": []},
        "train_medium": {"text": [], "label": []},
        "train_large": {"text": [], "label": []},
        "test": {"text": [], "label": []}
    }

    unused_data = []
    for percentage_str in ["small", "medium", "large"]:
        for _ in range(3):
            sample_data = length_data[percentage_str]
            sample_data = shuffle(sample_data, random_state=42)
            temp = sample_data[:111]
            rest_data = sample_data[111:]
            unused_data.extend(rest_data)
            data[f"train_{percentage_str}"]["text"] += [sample[0] for sample in temp]
            data[f"train_{percentage_str}"]["label"] += [sample[1] for sample in temp]

    sample_data = shuffle(unused_data, random_state=42)
    sample_data = sample_data[:3000]
    data["test"]["text"] = [sample[0] for sample in sample_data]
    data["test"]["label"] = [sample[1] for sample in sample_data]

    for key, value in data.items():
        data[key]["text"], data[key]["label"] = shuffle(data[key]["text"], data[key]["label"], random_state=42)

    with open("/scratch/jh7956/mixset_dataset/1122_data.json", "w") as f:
        json.dump(data, f)
    print("Data saved to 1122_data.json")

record = []
trial = 0
for small_per in np.arange(0, 1.1, 0.2):
    for medium_per in np.arange(0, 1.1, 0.2):
        for large_per in np.arange(0, 1.1, 0.2):
            if small_per + medium_per + large_per <=0:
                continue
            trial += 1
            trial_str = f"number_{trial} small_{small_per} medium_{medium_per} large_{large_per}"
            print("----------------------------------")
            print(f"Trial: {trial}")
            print(f"Small: {small_per}, Medium: {medium_per}, Large: {large_per}")
            # load the data
            with open("/scratch/jh7956/mixset_dataset/1122_data.json", "r") as f:
                data = json.load(f)
            
            data["train"] = {"text": [], "label": []}
            # sample the data according to percentage
            for key, value in data.items():
                if "train_small" in key:
                    small_sample_size = int(len(value["text"]) * small_per)
                    data["train"]["text"] += value["text"][:small_sample_size]
                    data["train"]["label"] += value["label"][:small_sample_size]
                elif "train_medium" in key:
                    medium_sample_size = int(len(value["text"]) * medium_per)
                    data["train"]["text"] += value["text"][:medium_sample_size]
                    data["train"]["label"] += value["label"][:medium_sample_size]
                elif "train_large" in key:
                    large_sample_size = int(len(value["text"]) * large_per)
                    data["train"]["text"] += value["text"][:large_sample_size]
                    data["train"]["label"] += value["label"][:large_sample_size]

            # 抽取validation set
            train_text = data["train"]["text"]
            train_label = data["train"]["label"]
            total_train = len(train_text)
            
            if total_train > 1000:
                indices = np.arange(total_train)
                np.random.seed(42)
                np.random.shuffle(indices)
                val_size = 1000
                val_indices = indices[:val_size]
                train_indices = indices[val_size:]

                val_text = [train_text[i] for i in val_indices]
                val_label = [train_label[i] for i in val_indices]

                train_text = [train_text[i] for i in train_indices]
                train_label = [train_label[i] for i in train_indices]

                data["val"] = {"text": val_text, "label": val_label}
                data["train"] = {"text": train_text, "label": train_label}
            else:
                data["val"] = {"text": [], "label": []}
            
            for key, value in data.items():
                print(f"{key}: {len(value['text'])}")
            
            data["train"]["text"], data["train"]["label"] = shuffle(data["train"]["text"], data["train"]["label"], random_state=42)
            
            result = run_supervised_experiment(
                data, 
                model='distilbert-base-uncased', 
                cache_dir=".cache", 
                batch_size=batch_size, 
                DEVICE=DEVICE, 
                pos_bit=1, 
                num_labels=3, 
                finetune=True, 
                test_only=args.test_only, 
                no_auc=True, 
                ckpt_dir="./ckpt", 
                domain=trial_str
            )
            
            f1_test = result['general']['f1_test']
            record.append({"trial": trial, "percentage": trial_str, "f1_test": f1_test})
            print("----------------------------------")

record = pd.DataFrame(record)
record.to_csv("1122.csv")
print("Record saved to 1122.csv")

top5 = record.sort_values(by="f1_test", ascending=False).head(5)
print("Top 5 trials")
print(top5)