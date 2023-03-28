import torch
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, RobertaTokenizer
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from datasets import load_dataset, Dataset
from torch.utils import data
from tqdm import tqdm
import numpy as np
import argparse
import mauve
import json
import nltk
import os
import re


##############################  dataset  #############################
class tokendataset(data.Dataset):
    def __init__(self, data_list, tokenizer):
        file_raws = len(data_list)
        self.dataset=data_list
        self.file_raws = file_raws
        self.tokenizer=tokenizer
        self.token_dataset=self.tokenize()

    def __len__(self):
        return self.file_raws

    def __getitem__(self, idx):
        return self.token_dataset[idx]

    def tokenize(self):
        new_all_data=[]
        for text in self.dataset:
            text = self.tokenizer.encode(text, add_special_tokens=True,
                                              max_length=100, truncation=True)
            new_data={}
            new_data["text"] = text
            new_all_data.append(new_data)
        return new_all_data


def padding_classifer_fn(datalist):
    input_ids = []
    attention_masks = []
    text_length = []
    for item in datalist:
        text_length.append(len(item["text"]))
    max_text_len = max(text_length)

    for i,item in enumerate(datalist):
        pad_len = max_text_len-text_length[i]
        text = item["text"]
        attention_mask = [1]*text_length[i]
        text = text + [0]*pad_len
        attention_mask.extend([0]*pad_len)

        input_ids.append(text)
        attention_masks.append(attention_mask)
    batch = {}
    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_masks
    return batch


############################  accuracy  #############################
def eval_classify_acc(eval_data, classifier, tokenizer, class_num, tar_att, device):
    eval_dataset = tokendataset(eval_data, tokenizer)
    eval_dataloader = data.DataLoader(eval_dataset, batch_size=300, shuffle=False,
                               drop_last=False, collate_fn=padding_classifer_fn)

    all_results = []
    data_num = len(eval_dataset)
    correct_num = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_dataloader)):
            input_ids, attention_mask = \
                batch["input_ids"], batch["attention_mask"]

            input_ids = torch.tensor(input_ids).to(device)
            attention_mask = torch.tensor(attention_mask).to(device)

            outputs = classifier(input_ids=input_ids, attention_mask=attention_mask,)

            logits = outputs[0]
            assert logits.shape[1] == class_num
            logits = logits.cpu()
            probs = softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=-1)

            batch_num = preds.shape[0]
            for n in range(batch_num):
                result = {}
                result["probs"] = probs[n].tolist()
                result["sentiment"] = preds[n].item()
                if result["sentiment"] == tar_att:
                    correct_num += 1
                all_results.append(result)
    return correct_num / data_num


def eval_latent_classify_acc():
    pass


##############################  perplexity  #############################
def perplexity(txts, device):
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=txts, model_id='gpt2-large', device=device)
    print('average perplexity:%.4f' % (results['mean_perplexity']))


def cal_ppl_use_loss(txts, device):
    model_id = "gpt2-large"
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    txts = {'text': txts}
    test = Dataset.from_dict(txts)
    encodings = tokenizer(test['text'], max_length=100, truncation=True, add_special_tokens=True)

    nlls = []
    ppls = []
    tokennum = 0
    for sent, attn_mask in zip(encodings.input_ids, encodings.attention_mask):
        # add bos_token (=eos_token)
        sent = tokenizer(tokenizer.bos_token).input_ids + sent
        attn_mask = [1] + attn_mask  # not indispensible
        tokennum += len(sent)
        sent = torch.tensor(sent).to(device)
        attn_mask = torch.tensor(attn_mask).to(device)
        with torch.no_grad():
            outputs = model(sent, attention_mask=attn_mask, labels=sent, use_cache=True, output_hidden_states=True)
            neg_log_likelihood = outputs.loss

        ppl = torch.exp(neg_log_likelihood.sum() / len(sent)-1)
        nlls.append(neg_log_likelihood)
        ppls.append(ppl)
    #perplexity_batch = torch.exp2(
    #    (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
    #    / shift_attention_mask_batch.sum(1)
    #)
    avgppl = torch.exp(torch.stack(nlls).sum() / len(encodings.input_ids))
    print('average perplexity:%.4f' % (avgppl.data))


def cal_ppl(txts, device):
    # another realization
    model_id = "gpt2-large"
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    txts = {'text': txts}
    test = Dataset.from_dict(txts)
    encodings = tokenizer(test['text'], max_length=100, truncation=True, add_special_tokens=True)

    ppls = []
    tokennum = 0
    for sent, attn_mask in tqdm(zip(encodings.input_ids, encodings.attention_mask)):
        # add bos_token (=eos_token)
        sent = tokenizer(tokenizer.bos_token).input_ids + sent
        attn_mask = [1] + attn_mask  # not indispensible
        tokennum += len(sent)
        sent = torch.tensor(sent).to(device)
        attn_mask = torch.tensor(attn_mask).to(device)
        with torch.no_grad():
            outputs = model(sent, attention_mask=attn_mask, labels=sent, use_cache=True, output_hidden_states=True)  
            logits = outputs.logits  # 输入的每个token经过gpt后的输出概率表

            shift_logits = logits[:-1, :].squeeze()  # seq_len, vocab_size
            shift_logits = torch.softmax(shift_logits, dim=-1)
            index = sent
            probs = []
            for i in range(shift_logits.shape[0]):
                # index[i+1]
                prob = torch.index_select(shift_logits[i], -1, index[i+1]).item()
                probs.append(prob)
            if 0 in probs:
                continue

            ppl2 = 1
            for prob in probs:
                ppl2 *= (1 / prob) ** (1 / shift_logits.shape[0])
            ppls.append(ppl2)
            '''  以下为等价写法
            ppl2 = 0
            for prob in probs:
                ppl2 += np.log(prob)
            ppl2 = np.exp((-1/shift_logits.shape[0]) * ppl2)
            ppls.append(ppl2)'''

    avgppl = sum(ppls) / len(ppls)
    print('average perplexity:%.4f' % (avgppl))
    return avgppl


def mauve():
    pass


##############################  diversity  ################################
def cal_dist(txts, n):
    dist_n = 0
    exceptions = 0
    for sent in txts:
        wordlist = sent.split(' ')
        tot_ngram = len(wordlist)-n+1
        if tot_ngram <= 0:
            exceptions += 1
            continue
        ngram_dic = {}
        for i in range(0, tot_ngram):
            ngram = ''
            for j in range(0, n):
                ngram += wordlist[i+j]
            if ngram in ngram_dic.keys():
                ngram_dic[ngram] = 0
            else:
                ngram_dic[ngram] = 1
        dist_n += sum(ngram_dic.values()) / tot_ngram
    return dist_n / (len(txts) - exceptions)


def calc_dist_n(texts, n):
    ngrams = []
    for text in tqdm(texts):
        # 将文本转换为n-gram序列
        words = re.findall(r'\w+', text.lower())  # 将文本中的单词转换为小写，并去除标点符号和空格
        ngrams.extend(zip(*[words[i:] for i in range(n)]))
    # 计算总n-gram数量和不同n-gram数量
    total_ngrams = len(ngrams)
    unique_ngrams = len(set(ngrams))
    # 计算dist-n
    dist_n = unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0
    return dist_n


def calc_self_bleu(test_corpus, max_ngram=4):
    bleu_scores = []
    for i in range(len(test_corpus)):
        test_corpus[i] = test_corpus[i].split()

    for i in tqdm(range(len(test_corpus))):
        reference = test_corpus[:i] + test_corpus[i+1:]
        hypothesis = test_corpus[i]

        # 计算BLEU分数
        bleu_score_i = sentence_bleu(reference, hypothesis)
        bleu_scores.append(bleu_score_i)

    # 计算平均self-BLEU分数
    self_bleu = np.mean(bleu_scores)
    return self_bleu


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='all')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--dir", type=str, default='../GPT2/results/')
    parser.add_argument("--file", type=str, default='len30_topk200_imdb_2.jsonl')
    parser.add_argument("--class_num", type=int, default=2)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    # 读取文件
    data_list = []
    file = args.file.split('.')
    file_path = os.path.join(args.dir, args.file)
    if file[1] == 'json':
        with open(file_path, 'r', encoding='utf8') as fin:
            for line in fin:
                # load: 针对文件 loads：针对字符串
                dic = json.loads(line)
                # data_list.extend(dic['pos'])
                for txt in dic['pos']:
                    # 去除起始符
                    data_list.append(txt[13:])

    elif file[1] == 'txt':
        with open(file_path, 'r', encoding='utf8') as fin:
            for line in fin:
                data_list.append(line)

    elif file[1] == 'jsonl':
        with open(file_path, 'r', encoding='utf8') as fin:
            for line in fin:
                dic = json.loads(line)
                data_list.append(dic['text'])

    # 评估模式
    if args.mode == 'ppl' or args.mode == 'all':
        # perplexity(data_list, args.device)
        cal_ppl(data_list, args.device)

    if args.mode == 'acc' or args.mode == 'all':
        # 暂时仅实现了情感评估
        class_num = args.class_num
        model_3classes = '../LatentCtrl/model/classifier/sentiment'
        model_2classes = "distilbert-base-uncased-finetuned-sst-2-english"

        # 2classes
        cls_tokenizer = AutoTokenizer.from_pretrained(model_2classes)
        classifier = AutoModelForSequenceClassification.from_pretrained(model_2classes)
        classifier.to(args.device)
        '''
        # 3 classes  0 -> Negative; 1 -> Neutral; 2 -> Positive
        classifier = AutoModelForSequenceClassification.from_pretrained(model)
        classifier.to(args.device)
        cls_tokenizer = RobertaTokenizer.from_pretrained(model)
        '''
        target = int(file[0][-1])
        result = eval_classify_acc(data_list, classifier, cls_tokenizer, class_num, target, args.device)
        print('sentiment: {} acc: {}'.format(target, result))

    if args.mode == 'dist' or args.mode == 'all':
        for n in range(1, 4):
            dist_n = dist(data_list, n)
            print('dist-{}: {}'.format(n, dist_n))

    if args.mode == 'sBL' or args.mode == 'all':
        print('self_BLEU: {}'.format(self_bleu(data_list)))
