import argparse
import datetime
import os
import json
import dataset_loader
from methods.utils import load_base_model, load_base_model_and_tokenizer
import numpy as np
from methods.supervised import run_supervised_experiment
from methods.metric_based import get_ll, get_rank, get_entropy, get_rank_GLTR, run_threshold_experiment, run_GLTR_experiment
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pdb

# python benchmark.py --finetune --three_classes --domain HSS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="TruthfulQA")
    parser.add_argument('--detectLLM', type=str, default="ChatGPT")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--mask_filling_model_name',
                        type=str, default="t5-large")
    parser.add_argument('--cache_dir', type=str, default=".cache")
    parser.add_argument('--DEVICE', type=str, default="cuda")

    # params for DetectGPT
    parser.add_argument('--pct_words_masked', type=float, default=0.3)
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_perturbation_list', type=str, default="10")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')
    
    # params for Mixcase
    parser.add_argument('--MGT_only_GPT', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--train_threshold', type=int, default=10000)
    parser.add_argument('--test_threshold', type=int, default=2000)
    parser.add_argument('--no_auc', action='store_true')
    parser.add_argument('--only_supervised', action='store_true')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--ckpt_dir',type=str, default="./ckpt")
    parser.add_argument('--log_name', type=str, default='Log')
    parser.add_argument('--transfer_filename', type=str, default=None)
    parser.add_argument('--three_classes', action='store_true')
    parser.add_argument('--finetune', action="store_true")
    parser.add_argument('--mixcase_as_mgt', action="store_true")
    parser.add_argument('--domain', type=str, default="CS", choices=["CS", "HSS", "PHX", "All"])

    
    args = parser.parse_args()

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
        print(f"'{args.ckpt_dir}' are created.")
    else:
        print(f"'{args.ckpt_dir}' already exist.")
    DEVICE = args.DEVICE
    print(f"Using device {DEVICE}")

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    mask_filling_model_name = args.mask_filling_model_name
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples

    cache_dir = args.cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    # get generative model
    base_model, base_tokenizer = load_base_model_and_tokenizer(
        args.base_model_name, cache_dir)
    base_model.to(DEVICE)
    print(f"Using base model {args.base_model_name}")

    raw_data = {"text": [], "label": []}

    dataset_name =  args.domain + "_" + "processed_data.json"
    dataset_path = "/scratch/jh7956/mixset_dataset/" + dataset_name
    print("Dataset path: ", dataset_path)

    for root, dirs, files in os.walk("/scratch/jh7956/mixset_dataset"):
        if dataset_name in files:
            print("Using old dataset file")
            break
        else:
            if args.domain == "All":
                for file in files:
                    print("processing file: ", file)
                    with open(os.path.join(root, file), 'r') as f:
                        json_data = json.load(f)
                        for key, value in json_data.items():
                            text = value["abstract"]
                            # Encode and truncate the text
                            tokens = base_tokenizer(data[dataset]['text'][i], max_length=1023, truncation=True, return_tensors="pt")
                            # Decode tokens to string and update
                            truncated_text = base_tokenizer.decode(tokens['input_ids'][0])
                            raw_data["text"].append(truncated_text)
                            raw_data["label"].append(int(value["type"]))                  
            else:
                for file in files:
                    if args.domain in file:
                        print("processing file: ", file)
                        with open(os.path.join(root, file), 'r') as f:
                            json_data = json.load(f)
                            for key, value in json_data.items():
                                try:
                                    text = value["abstract"]
                                except:
                                    print("error in ", file)
                                # Encode and truncate the text
                                tokens = base_tokenizer(text, max_length=1023, truncation=True, return_tensors="pt")
                                # Decode tokens to string and update
                                truncated_text = base_tokenizer.decode(tokens['input_ids'][0])
                                raw_data["text"].append(truncated_text)
                                raw_data["label"].append(int(value["type"])) 

            # train test split
            train_text, test_text = train_test_split(raw_data["text"], test_size=0.2, random_state=42)
            train_label, test_label = train_test_split(raw_data["label"], test_size=0.2, random_state=42)
            print("train size: ", len(train_text), "test size: ", len(test_text))

            # shuffle the data
            train_text, train_label = shuffle(train_text, train_label)
            print("Shuffle done")

            # save the data
            processed_data = {
                "train": {
                    "text": train_text,
                    "label": train_label
                },
                "test": {
                    "text": test_text,
                    "label": test_label
                }
            }

            with open(dataset_path, "w") as f:
                json.dump(processed_data, f)
            print("Data saved to ", dataset_path)
    
    # load the data
    print("Loading data")
    with open(dataset_path, "r") as f:
        data = json.load(f)
    print("Data loaded from ", dataset_path)
    
    # check if there are actually three classes
    if len(set(data["train"]["label"])) != 3:
        raise ValueError("There are not three classes in the dataset")
    print("There are three classes in the dataset")

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
    
    outputs = []
    
    print("Start three classes experiments")
    outputs.append(run_threshold_experiment(data, ll_criterion, "likelihood", test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
    outputs.append(run_threshold_experiment(data, rank_criterion, "rank", test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
    outputs.append(run_threshold_experiment(
        data, logrank_criterion, "log_rank", test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
    outputs.append(run_threshold_experiment(
        data, entropy_criterion, "entropy", test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
    outputs.append(run_GLTR_experiment(data, GLTR_criterion, "rank_GLTR", test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
    outputs.append(run_supervised_experiment(data, model='distilbert-base-uncased',
               cache_dir=cache_dir, batch_size=batch_size, DEVICE=DEVICE, pos_bit=1, num_labels=3, finetune=True, test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir))
    outputs.append(run_supervised_experiment(data, model='Hello-SimpleAI/chatgpt-detector-roberta',
                cache_dir=cache_dir, batch_size=batch_size, DEVICE=DEVICE, pos_bit=1, num_labels=3, test_only = args.test_only, no_auc=args.no_auc, ckpt_dir=args.ckpt_dir, finetune=True))        

    # save results with time stamp
    log = "logs/results-{START_DATE}-{START_TIME}.json"
    if not os.path.exists(os.path.dirname(log)):
        os.makedirs(os.path.dirname(log))
    with open(log, "w") as wf:
        for row in outputs:
            json.dump(row, wf)
            wf.write("\n")

    print("Finish")
