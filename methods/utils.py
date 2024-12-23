import transformers
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import time
from functools import wraps
import random
import pickle
import os
from joblib import dump, load 

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds\n\n')
        return result
    return timeit_wrapper


# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")


def select_train_data(data, select_num=-1):
    new_train = {
        'text': [],
        'label': [],
    }
    total_num = len(data['train']['text'])
    index_list = list(range(total_num))
    random.seed(0)
    random.shuffle(index_list)
    if select_num == -1:
        return data
    else:
        for i in range(select_num):
            text = data['train']['text'][index_list[i]]
            label = data['train']['label'][index_list[i]]
            new_train['text'].append(text)
            new_train['label'].append(label)
        data['train'] = new_train

    return data


def filter_test_data(data, max_length=25):
    new_test = {
        'text': [],
        'label': [],
    }
    for i in range(len(data['test']['text'])):
        text = data['test']['text'][i]
        label = data['test']['label'][i]
        if len(text.split()) <= 25:
            new_test['text'].append(text)
            new_test['label'].append(label)
    data['test'] = new_test
    return data


def load_base_model_and_tokenizer(name, cache_dir):

    print(f'Loading BASE model {name}...')
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        name, cache_dir=cache_dir)
    print(f'Loading BASE tokenizer {name}...')
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(
        name, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer


def load_base_model(base_model, DEVICE):
    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    start = time.time()
    base_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')


def cal_metrics(label, pred_label, pred_posteriors, no_auc):
    flag = True
    for i in set(label):
        if i >= 2:
            flag = False
    if flag:
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label, average='weighted')
        recall = recall_score(label, pred_label, average='weighted')
        f1 = f1_score(label, pred_label, average='weighted')
        if no_auc: 
            auc = -1.0
        else:
            auc = roc_auc_score(label, pred_posteriors)
    else:
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label, average='weighted')
        recall = recall_score(label, pred_label, average='weighted')
        f1 = f1_score(label, pred_label, average='weighted')
        auc = -1.0
        conf_m = confusion_matrix(label, pred_label)
        print(conf_m)
    return acc, precision, recall, f1, auc


def get_clf_results(x_train, y_train, x_test, y_test, name, test_only, no_auc, ckpt_dir):
    model_path = f"{ckpt_dir}/{name}.joblib"  
    if test_only:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        clf = load(model_path)  
    else:
        clf = LogisticRegression(random_state=0).fit(x_train, y_train)
        dump(clf, model_path)  
    if test_only:
        train_res = 0, 0, 0, 0, -1.0
    else:
        y_train_pred = clf.predict(x_train)
        y_train_pred_prob = clf.predict_proba(x_train)
        y_train_pred_prob = [_[1] for _ in y_train_pred_prob]
        acc_train, precision_train, recall_train, f1_train, auc_train = cal_metrics(
            y_train, y_train_pred, y_train_pred_prob, no_auc)
        train_res = acc_train, precision_train, recall_train, f1_train, auc_train

    y_test_pred = clf.predict(x_test)
    y_test_pred_prob = clf.predict_proba(x_test)
    y_test_pred_prob = [_[1] for _ in y_test_pred_prob]
    acc_test, precision_test, recall_test, f1_test, auc_test = cal_metrics(
        y_test, y_test_pred, y_test_pred_prob, no_auc)
    test_res = acc_test, precision_test, recall_test, f1_test, auc_test

    return train_res, test_res
