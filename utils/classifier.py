import random
import numpy as np
import pandas as pd
import torch
import csv
import sys, os, json
from transformers import DebertaTokenizer, DebertaConfig, DebertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


class tokendataset(Dataset):
    def __init__(self, args, data_path, tokenizer):
        self.file_path = data_path
        dataset = []
        file_row = 0
        with open(self.file_path, 'r') as f:
            for line in f.readlines():
                dic = json.loads(line)
                dic['text'] = tokenizer.encode(
                    dic['text'], max_length=300, truncation=True, add_special_tokens=True
                )
        self.file_row = file_row
        self.tokenizer = tokenizer
        self.dataset = dataset

    def __len__(self):
        return self.file_row

    def __getitem__(self, idx):
        return self.dataset[idx]


def padding_fuse_fn(data_list):
    input_ids = []
    attention_masks = []
    label = []
    texts = []
    for item in data_list:
        texts.append(len(item['text']))
        label.append([item['label']])

    max_text_len = max(texts)
    for i, item in enumerate(data_list):
        text_pad_len = max_text_len - texts[i]
        attention_mask = [1] * texts[i] + [0] * text_pad_len
        text = item["text"] + [0] * text_pad_len

        input_ids.append(text)
        attention_masks.append(attention_mask)
    
    batch = {}
    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_masks
    batch["label"] = label
    return batch


class Classifier:
    def __init__(
            self, model_name_or_path, save_path, train_dataset, val_dataset,
            tokenizer, n_class, epochs, batch_size, max_len, args, device
        ):
        self.args = args
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = DebertaForSequenceClassification.from_pretrained(model_name_or_path, num_labels=n_class)
        self.device = device

    def accuracy(self, preds, labels):
        # preds_flat = np.argmax(preds, axis=1).flatten()
        # labels_flat = labels.flatten()
        return accuracy_score(labels, preds)

    def f1(self, preds, labels):
        return f1_score(labels, preds)

    def train_model(self):
    	self.model.to(self.device)
        self.model.train()
        self.model.zero_grad()

        train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
        # set sampler --> shuffle must be False
        train_dataloader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, drop_last=False, 
            collate_fn=padding_fuse_fn, sampler=train_sampler)
        val_sampler = torch.utils.data.RandomSampler(self.val_dataset)
        # set sampler --> shuffle must be False
        val_dataloader = DataLoader(
            self.val_dataset, batch_size=args.batch_size, drop_last=False, 
            collate_fn=padding_fuse_fn, sampler=val_sampler)

        global_step = 0
        current_epoch = 0
        tr_loss, logging_loss = 0.0, 0.0
        optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
        num_training_steps = len(train_dataloader) * args.num_train_epochs
        num_warmup_steps = math.floor(num_train_steps * args.warmup_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        train_loss = []

        for epoch in trange(int(args.num_train_epochs), desc='Epoch'):
            total_loss, total_val_loss = 0, 0
            total_eval_accuracy = 0
            total_eval_f1 = 0
            current_epoch += 1

            for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
                input_ids, attention_mask, label = batch['input_ids'], batch['attention_mask'], batch['label']

                # output: transformers.modeling_outputs.SequenceClassifierOutput
                output = self.model(input_ids=text, attention_mask=attention_mask, labels=label)
                loss = output.loss
                logits = output.logits
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)
            train_loss.append(avg_train_loss)

            print(f'Train loss: {avg_train_loss}')
            self.save_model(args.save_path + '-' + str(epoch))
            
    def save_model(self, path):
        self.model.save_pretrained(path)

    def eval_model(self, Tokenizer, model, text_list, y_true):
        preds = self.predict_batch(Tokenizer, model, text_list)
        print(classification_report(y_true, preds))

    def predict_batch(self, text_list):
        self.model.to(self.device)
        self.model.eval()

        tokens = self.tokenizer(
            text_list,
            padding = True,
            truncation = True,
            max_length = args.max_len,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        pred_data = TensorDataset(input_ids, attention_mask)
        pred_dataloader = DataLoader(pred_data, batch_size=args.batch_size, shuffle=False)
        preds = []
        pred_logits = []
        for i, batch in enumerate(tqdm(pred_dataloader, desc='Test')):
            with torch.no_grad():
                output = self.model(input_ids=batch[0].to(self.device),
                                    attention_mask=batch[1].to(self.device)
                                )
                outputlogits = torch.softmax(output.logits, 1)
                logits = outputlogits.detach().cpu().numpy()
                pred = list(np.argmax(logits, axis=1))
                preds += pred
                # torch.gather 选取对应的logit
                pred = torch.tensor(pred).reshape((4, 1))
                logits = outputlogits.detach().cpu()
                pred_logit = torch.gather(logits, 1, pred).squeeze(-1)
                pred_logits += pred_logit.numpy().tolist()
        return preds, pred_logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="train_shuffled.jsonl", type=str, )
    parser.add_argument("--val_data_path", default="val.jsonl", type=str, )
    parser.add_argument("--test_data_path", default="test.jsonl", type=str, )
    parser.add_argument("--model_name_or_path", default="/home/chenyh/robot/model/chinese_roberta", type=str, )
    parser.add_argument("--save_path", default="/home/chenyh/robot/model/chinese_roberta", type=str, )
    parser.add_argument("--mode", default="train", type=str, )

    parser.add_argument("--batch_size", default=8, type=int, )
    parser.add_argument("--n_class", default=2, type=int, )
    parser.add_argument("--max_len", default=200, type=int, )
    parser.add_argument("--num_train_epochs", default=2, type=int, help="训练epochs次数", )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="学习率衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="梯度裁减值")
    parser.add_argument("--label_smooth", default=0.9, type=float, help="标签平滑系数")
    parser.add_argument("--warmup_rate", default=0.1, type=int, help="学习率线性预热步数")
    parser.add_argument("--logging_steps", type=int, default=1000, help="每多少步打印日志")
    parser.add_argument("--seed", type=int, default=42, help="初始化随机种子")
    parser.add_argument("--max_steps", default=1000000, type=int, help="训练的总步数", )
    parser.add_argument("--save_steps", default=200000, type=int, help="保存的间隔steps", )
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--nodes', default=1, type=int, metavar='N')
    parser.add_argument("--port", default='6666', type=str, help="端口号")

    args = parser.parse_args()
    set_seed(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    # tokenizer and dataloader
    tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
    train_dataset = tokendataset(args, args.data_path, tokenizer=tokenizer)
    train_dataset = tokendataset(args, args.val_data_path, tokenizer=tokenizer)

    model = Classifier(
        model_name_or_path=args.model_name_or_path, 
        train_dataset=train_dataset, val_dataset=val_dataset,
        tokenizer=tokenizer, device=args.device
    )
    model.train_model()

