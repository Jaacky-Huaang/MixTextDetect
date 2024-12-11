import argparse
import datetime
import os
import json
import dataset_loader
from methods.utils import load_base_model, load_base_model_and_tokenizer
import numpy as np
import pandas as pd
from methods.supervised import run_supervised_experiment, my_get_supervised_model_prediction
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


batch_size = args.batch_size
# Define thresholds for text length
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

    # Split the data according to length
    length_data = defaultdict(list)
    for text, label in data:
        # get the number of tokens in the text
        length = len(base_tokenizer(text)['input_ids'])
        for key, value in thresholds.items():
            if value[0] <= length < value[1]:
                length_data[key].append((text, label))
                break

    # for the three labels, choose respectively 111 small, 111 medium, 111 large samples as training data;in total 333 *3 samples
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

    # sample the test data
    sample_data = shuffle(unused_data, random_state=42)
    sample_data = sample_data[:3000]
    data["test"]["text"] = [sample[0] for sample in sample_data]
    data["test"]["label"] = [sample[1] for sample in sample_data]

    # Output dataset sizes
    for key, value in data.items():
        print(f"{key}: {len(value['text'])}")

    # shuffle the data
    for key, value in data.items():
        data[key]["text"], data[key]["label"] = shuffle(data[key]["text"], data[key]["label"], random_state=42)

    # store the data
    with open("/scratch/jh7956/mixset_dataset/1122_data.json", "w") as f:
        json.dump(data, f)
    print("Data saved to 1122_data.json")
else:
    with open("/scratch/jh7956/mixset_dataset/1122_data.json", "r") as f:
        data = json.load(f)

if not os.path.exists("/scratch/jh7956/mixset_dataset/1124_data.json"):
    data["train"] = {"text": [], "label": []}
    # sample the data according to percentage
    for key, value in data.items():
        if "train" in key and key != "train":
            data["train"]["text"] += value["text"]
            data["train"]["label"] += value["label"]
    #remove train_small, train_medium, train_large
    data.pop("train_small")
    data.pop("train_medium")
    data.pop("train_large")
    training_data = data["train"]
    pred, loss = my_get_supervised_model_prediction (model_name = "distilbert-base-uncased", data = training_data, device = DEVICE, num_labels = 3,cache_dir = cache_dir)
    data["train"]["loss"] = loss
    assert len(data["train"]["text"]) == len(data["train"]["loss"])
    # store the data
    with open("/scratch/jh7956/mixset_dataset/1124_data.json", "w") as f:
        json.dump(data, f)
    print("Data saved to 1124_data.json")
else:
    with open("/scratch/jh7956/mixset_dataset/1124_data.json", "r") as f:
        data = json.load(f)

# get the list of loss values
losses = []
for key, value in data.items():
    if "train" in key:
        losses += value["loss"]

# examine the data: check the ones with the highest loss
for key, value in data.items():
    if "train" in key:
        loss = value["loss"]
        loss = np.array(loss)
        loss = np.sort(loss)
        print(f"{key}: {loss[:5]}")


# run different percentages of the data
record = []
trial = 0
all_loss = []
for threshold in np.arange(0.05, 0.1, 0.01):
    trial += 1
    trial_str = f"{threshold}"
    print("----------------------------------")
    print(f"Trial: {trial}")
    print(f"Percentage: {trial_str}")

    #select training data with the highest loss
    training_data = {"text": [], "label": []}
    used_data = {}
    avg_loss = 0
    for key, value in data.items():
        if "train" in key:
            loss = value["loss"]
            loss = np.array(loss)
            loss = np.sort(loss)
            threshold = loss[int(len(loss) * threshold)]
            for i in range(len(value["text"])):
                if value["loss"][i] > threshold:
                    avg_loss += value["loss"][i]
                    training_data["text"].append(value["text"][i])
                    training_data["label"].append(value["label"][i])
    avg_loss = avg_loss / len(training_data["text"])
    print(f"Average loss: {avg_loss}")

    used_data["train"] = training_data
    used_data["test"] = data["test"]
    for key, value in used_data.items():
        print(f"{key}: {len(value['text'])}")
    # shuffle the data
    used_data["train"]["text"], used_data["train"]["label"] = shuffle(used_data["train"]["text"], used_data["train"]["label"], random_state=42)
    result = run_supervised_experiment(used_data, model='distilbert-base-uncased', cache_dir=cache_dir, batch_size=batch_size, DEVICE=DEVICE, pos_bit=1, num_labels=3, finetune=True, test_only = args.test_only, no_auc=True, ckpt_dir=ckpt_dir, domain="CS_length75")
    f1_test = result['general']['f1_test']
    record.append({"trial": trial, "percentage": trial_str, "f1_test": f1_test})
    print("----------------------------------")

# save the record
record = pd.DataFrame(record)
record.to_csv("1124.csv")
print("Record saved to 1124.csv")

# get the top 5 trials
top5 = record.sort_values(by="f1_test", ascending=False).head(5)
print("Top 5 trials")
print(top5)
