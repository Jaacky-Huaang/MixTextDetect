import numpy as np
import transformers
import torch
from tqdm import tqdm
from methods.utils import timeit, cal_metrics
from torch.utils.data import DataLoader
from transformers import AdamW
import os
from torch.nn import CrossEntropyLoss

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def cal_metrics(y_true, y_pred, y_prob, no_auc):
    # 此处根据需求编写metrics计算逻辑，以下为示例占位
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc = -1.0
    if not no_auc and len(set(y_true)) == 2:
        # 二分类才计算AUC
        auc = roc_auc_score(y_true, y_prob)
    return acc, prec, rec, f1, auc


def get_supervised_model_prediction(model, tokenizer, data, batch_size, DEVICE, pos_bit=0):
    with torch.no_grad():
        preds = []
        for start in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            batch_data = tokenizer(batch_data, padding=True, truncation=True,
                                   max_length=512, return_tensors="pt").to(DEVICE)
            preds.extend(model(**batch_data).logits.softmax(-1)
                         [:, pos_bit].tolist())
    return preds

def get_supervised_model_prediction_multi_classes(model, tokenizer, data, batch_size, DEVICE, pos_bit=0):
    with torch.no_grad():
        preds = []
        for start in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            batch_data = tokenizer(batch_data, padding=True, truncation=True,
                                   max_length=512, return_tensors="pt").to(DEVICE)
            preds.extend(torch.argmax(
                model(**batch_data).logits, dim=1).tolist())
    return preds


def fine_tune_model(model, tokenizer, data, batch_size, DEVICE, pos_bit=1, num_labels=2, epochs=3, ckpt_dir='./ckpt/', domain='CS', has_val=False, no_auc=False):
    train_text = data['train']['text']
    train_label = data['train']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']

    if pos_bit == 0 and num_labels == 2:
        train_label = [1 if _ == 0 else 0 for _ in train_label]
        test_label = [1 if _ == 0 else 0 for _ in test_label]

    train_encodings = tokenizer(train_text, truncation=True, padding=True)
    train_dataset = CustomDataset(train_encodings, train_label)

    model.train()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    # 若有val集则处理val数据
    val_loader = None
    if has_val and 'val' in data and len(data['val']['text'])>0:
        val_text = data['val']['text']
        val_label = data['val']['label']
        if pos_bit == 0 and num_labels == 2:
            val_label = [1 if _ == 0 else 0 for _ in val_label]
        val_encodings = tokenizer(val_text, truncation=True, padding=True)
        val_dataset = CustomDataset(val_encodings, val_label)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6)

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # 每轮结束如果有val集则评估val
        if val_loader is not None:
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    labels = batch['labels'].cpu().tolist()
                    logits = model(input_ids, attention_mask=attention_mask).logits
                    if num_labels == 2:
                        # 二分类使用pos_bit计算概率
                        probs = torch.softmax(logits, dim=-1)[:, pos_bit].cpu().tolist()
                        preds = [round(p) for p in probs]
                        all_preds.extend(probs)
                    else:
                        # 多分类直接argmax
                        preds = torch.argmax(logits, dim=-1).cpu().tolist()
                        # 无实际概率分布需求的话，可用占位概率
                        all_preds.extend([1.0]*len(preds))
                    all_labels.extend(labels)
            val_pred_class = [round(_) for _ in all_preds]
            val_res = cal_metrics(all_labels, val_pred_class, all_preds, no_auc)
            acc_val, precision_val, recall_val, f1_val, auc_val = val_res
            print(f"Epoch {epoch}: Val Acc: {acc_val}, Val Precision: {precision_val}, Val Recall: {recall_val}, Val F1: {f1_val}, Val AUC: {auc_val}")

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_path = os.path.join(ckpt_dir, f"{model._get_name()}_CS10k.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def run_supervised_experiment(data, model, cache_dir, batch_size, DEVICE, pos_bit=0, finetune=False, num_labels=2, epochs=3, test_only=False, no_auc=False, ckpt_dir='./ckpt/', domain='transfer'):
    print(f'Beginning supervised evaluation with {model}...')
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(
        model, num_labels=num_labels, cache_dir=cache_dir, ignore_mismatched_sizes=True).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model, cache_dir=cache_dir)

    # 判断是否有val数据
    has_val = 'val' in data and len(data['val']['text']) > 0

    if finetune and not test_only:
        print(f"Fine-tuning {model}...")
        fine_tune_model(detector, tokenizer, data, batch_size, DEVICE, pos_bit, num_labels, epochs=epochs, ckpt_dir=ckpt_dir, domain=domain, has_val=has_val, no_auc=no_auc)
    elif finetune and test_only:
        model_path = os.path.join(ckpt_dir, f"{detector._get_name()}_CS10k.pth")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")
        detector.load_state_dict(torch.load(model_path))

    if test_only:
        # 仅测试阶段
        test_text = data['test']['text']
        test_label = data['test']['label']

        if num_labels == 2:
            test_preds = get_supervised_model_prediction(
                detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)
        else:
            test_preds = get_supervised_model_prediction_multi_classes(
                detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)

        y_test_pred_prob = test_preds
        y_test_pred = [round(_) for _ in y_test_pred_prob]
        y_test = test_label

        train_res = (0, 0, 0, 0, -1.0)
        test_res = cal_metrics(y_test, y_test_pred, y_test_pred_prob, no_auc)
        acc_train, precision_train, recall_train, f1_train, auc_train = train_res
        acc_test, precision_test, recall_test, f1_test, auc_test = test_res
        print(f"{model} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
        print(f"{model} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")

        del detector
        torch.cuda.empty_cache()

        return {
            'name': model,
            'general': {
                'acc_train': acc_train,
                'precision_train': precision_train,
                'recall_train': recall_train,
                'f1_train': f1_train,
                'auc_train': auc_train,
                'acc_test': acc_test,
                'precision_test': precision_test,
                'recall_test': recall_test,
                'f1_test': f1_test,
                'auc_test': auc_test,
            }
        }
    else:
        # 训练+测试阶段
        train_text = data['train']['text']
        train_label = data['train']['label']
        test_text = data['test']['text']
        test_label = data['test']['label']

        # 如果有val集，进行val预测与指标计算
        if has_val:
            val_text = data['val']['text']
            val_label = data['val']['label']
            if num_labels == 2:
                val_preds = get_supervised_model_prediction(
                    detector, tokenizer, val_text, batch_size, DEVICE, pos_bit)
            else:
                val_preds = get_supervised_model_prediction_multi_classes(
                    detector, tokenizer, val_text, batch_size, DEVICE, pos_bit)
            y_val_pred_prob = val_preds
            y_val_pred = [round(_) for _ in y_val_pred_prob]
            y_val = val_label
            val_res = cal_metrics(y_val, y_val_pred, y_val_pred_prob, no_auc)
        else:
            val_res = (0,0,0,0,-1.0)

        # train和test预测
        if num_labels == 2:
            train_preds = get_supervised_model_prediction(
                detector, tokenizer, train_text, batch_size, DEVICE, pos_bit)
            test_preds = get_supervised_model_prediction(
                detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)
        else:
            train_preds = get_supervised_model_prediction_multi_classes(
                detector, tokenizer, train_text, batch_size, DEVICE, pos_bit)
            test_preds = get_supervised_model_prediction_multi_classes(
                detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)

        y_train_pred_prob = train_preds
        y_train_pred = [round(_) for _ in y_train_pred_prob]
        y_train = train_label

        y_test_pred_prob = test_preds
        y_test_pred = [round(_) for _ in y_test_pred_prob]
        y_test = test_label

        train_res = cal_metrics(y_train, y_train_pred, y_train_pred_prob, no_auc)
        test_res = cal_metrics(y_test, y_test_pred, y_test_pred_prob, no_auc)
        acc_train, precision_train, recall_train, f1_train, auc_train = train_res
        acc_test, precision_test, recall_test, f1_test, auc_test = test_res

        if has_val:
            acc_val, precision_val, recall_val, f1_val, auc_val = val_res
            print(f"{model} val_acc: {acc_val}, val_precision: {precision_val}, val_recall: {recall_val}, val_f1: {f1_val}, val_auc: {auc_val}")
        else:
            acc_val, precision_val, recall_val, f1_val, auc_val = (-1,-1,-1,-1,-1)

        print(f"{model} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
        print(f"{model} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")

        del detector
        torch.cuda.empty_cache()

        return {
            'name': model,
            'general': {
                'acc_train': acc_train,
                'precision_train': precision_train,
                'recall_train': recall_train,
                'f1_train': f1_train,
                'auc_train': auc_train,
                'acc_val': acc_val,
                'precision_val': precision_val,
                'recall_val': recall_val,
                'f1_val': f1_val,
                'auc_val': auc_val,
                'acc_test': acc_test,
                'precision_test': precision_test,
                'recall_test': recall_test,
                'f1_test': f1_test,
                'auc_test': auc_test,
            }
        }


def my_get_supervised_model_prediction(model_name, data,  device, num_labels, cache_dir):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, cache_dir=cache_dir, ignore_mismatched_sizes=True).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir)
    model.eval()
    preds = []
    individual_losses = []
    loss_fn = CrossEntropyLoss()

    texts = data['text']
    labels = data['label']

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), 1), desc="Evaluating"):
            text = texts[start]
            label = labels[start]
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs, labels=torch.tensor([label]).to(device))
            loss = outputs.loss
            logits = outputs.logits
            preds.append(logits.softmax(-1).tolist()[0])
            individual_losses.append(loss.item())

    return preds, individual_losses