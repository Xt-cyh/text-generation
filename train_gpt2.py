import torch
# from gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
# from prefix_tuning import PrefixGPT2
# from prompt_tuning import PromptTuning
from baseGPTmodel.prefix_tuning import PrefixGPT2
from baseGPTmodel.prompt_tuning import PromptTuning

from tqdm import tqdm, trange
import logging
import argparse
import random
import numpy as np
import pandas as pd
import json
import math
import os
import shutil


labels = {
    'positive': 1, 'negative': 0, 'world':1,
    'sport': 2, 'business':3, 'science':4
}
ids_to_labels = {
    1:'positive' , 0:'negative' , 1:'world',
    2:'sport' , 3:'business', 4:'science'
}

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def get_parameter_number(model):
    tot = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total:{}, Trainable:{}'.format(tot, trainable))

def transfer_label(label_id, tokenizer):
    # 应对混合label训练
    size = label_id.shape
    final_id = torch.zeros(size).reshape(-1)
    label_id = label_id.reshape(-1)
    for i in label_id.shape[0]:
        final_id[i] = tokenizer.encode(ids_to_labels[label_id[i]])
    return final_id.reshape(size)


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
                # 根据label进行筛选
                if dic['label'] == labels[args.label]:
                    dataset.append(dic)
                    file_row += 1
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


def evaluation(args, is_valid=True, model_name_or_path=None):
    if is_valid:
        data_path = args.dev_path
    else:
        data_path = args.test_path

    dataset = tokendataset(data_path=data_path, tokenizer=args.tokenizer)
    file_row = dataset.file_row
    sampler = torch.utils.data.RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=False, collate_fn=padding_fuse_fn, sampler=sampler)

    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    model.to(args.device)
    model.eval()

    tr_loss = 0.0
    global_step = 0
    logs = {}

    for step, batch in enumerate(tqdm(dataloader, desc='Evaluation')):
        global_step += 1
        input_ids, attention_mask, label = batch['input_ids'], batch['attention_mask'], batch['label']
        eos_token_ids = torch.tensor(args.tokenizer.encode(args.tokenizer.eos_token))
        eos_token_ids = eos_token_ids.expand(args.batch_size, eos_token_ids.shape[0])
        input_ids = torch.tensor(input_ids)
        input_ids = torch.cat([eos_token_ids, input_ids], dim=1).to(args.device)
        eos_token_mask = torch.tensor([1]).expand(args.batch_size, 1)
        prefix_mask = torch.tensor([1] * args.prefix_len).expand(args.batch_size, args.prefix_len)
        attention_mask = torch.tensor(attention_mask)
        attention_mask = torch.cat([eos_token_mask, attention_mask], dim=1)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1).to(args.device)
        label = torch.tensor(label).to(args.device)
        dic = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=True)  # , use_prefix=True
        logits = dic.logits
        shift_logits = logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        loss = args.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        tr_loss += loss.item()
        
    logs['TYPE'] = "Evaluation"
    logs['avg_loss'] = tr_loss / global_step
    print(logs)

    return tr_loss / global_step


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    args.tokenizer = tokenizer
    
    train_dataset = tokendataset(args, args.data_path, tokenizer=tokenizer)
    file_row = train_dataset.file_row
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # set sampler --> shuffle must be False
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False, collate_fn=padding_fuse_fn, sampler=train_sampler)
    # pdb.set_trace()
    model_config = GPT2Config.from_pretrained(args.model_name_or_path)
    model_config.prefix_len = args.prefix_len
    model_config.prefix_mid_size = args.prefix_mid_size

    decoder = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, config=model_config)
    if args.method == 'prefix_tuning':
        model = PrefixGPT2(decoder=decoder, decoder_tokenizer=tokenizer, args=args)
    elif args.method == 'prompt_tuning':
        model = PromptTuning(decoder=decoder, decoder_tokenizer=tokenizer, args=args)
    else:
        model = decoder

    if args.method != 'gpt':
        model.fix_decoder()
    
    # 模型参数量
    get_parameter_number(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    model.to(args.device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    num_train_steps = math.floor(len(train_dataset) / (
            args.batch_size * args.gradient_accumulation_steps)) * args.num_train_epochs
    num_warmup_steps = math.floor(num_train_steps * args.warmup_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.train()
    model.zero_grad()
    loss_fct = CrossEntropyLoss()
    args.loss_fct = loss_fct
    logger.info('start_training')
    
    current_epoch = 0
    dev_loss_list = []
    output_dir_list = []
    for epoch in trange(int(args.num_train_epochs), desc='Epoch'):
        current_epoch += 1
        for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
            input_ids, attention_mask, label = batch['input_ids'], batch['attention_mask'], batch['label']

            eos_token_ids = torch.tensor(tokenizer.encode(tokenizer.eos_token))
            eos_token_ids = eos_token_ids.expand(args.batch_size, eos_token_ids.shape[0])
            input_ids = torch.tensor(input_ids)
            # control_code = None

            input_ids = torch.cat([eos_token_ids, input_ids], dim=1).to(args.device)
            eos_token_mask = torch.tensor([1]).expand(args.batch_size, 1)
            attention_mask = torch.tensor(attention_mask)
            attention_mask = torch.cat([eos_token_mask, attention_mask], dim=1)
            if args.method == "prefix_tuning":
                prefix_mask = torch.tensor([1] * args.prefix_len).expand(args.batch_size, args.prefix_len)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            if args.method == "prompt_tuning":
                prompt_mask = torch.tensor([1] * args.prompt_len).expand(args.batch_size, args.prompt_len)
                attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
                # control_code = transfer_label(label, tokenizer)
                # control_code = ids_to_labels[label[0][0]]
            attention_mask = attention_mask.to(args.device)
            label = torch.tensor(label).to(args.device)
            
            # pdb.set_trace() , use_prefix=args.prefix_tuning
            # output 需要加上control code以初始化prompt
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)

            # calculate loss
            logits = output.logits
            if args.method == "prompt_tuning":
                shift_logits = logits[:, args.prompt_len:-1, :].contiguous()
            else:
                shift_logits = logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
            loss.requires_grad_(True)

            # loss = output.loss  # returned when labels(input itself) is provided
            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logs = {}
                loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                logs['epoch'] = current_epoch
                logs['step'] = global_step
                logs['loss'] = loss_scalar
                logging_loss = tr_loss
                print(logs)

            if global_step % args.save_steps == 0:
                if args.method != 'gpt':
                    output_dir = os.path.join(args.output_dir, '{}/{}-prefixlen-{}-bs-{}-steps-{}.pth'.format(
                        args.method, args.label, args.prefix_len, args.batch_size * args.gradient_accumulation_steps, global_step)
                    )
                    torch.save(model.state_dict(), output_dir)
                else:
                    output_dir = os.path.join(args.output_dir, '{}-bs-{}-steps-{}'.format(args.label, args.batch_size * args.gradient_accumulation_steps, global_step))
                    model_to_save = (model.module if hasattr(model, 'module') else model)
                    model_to_save.save_pretrained(output_dir)
                    model_config.save_pretrained(output_dir)
                # dev_loss = evaluation(args=args, is_valid=True, model_name_or_path=output_dir)
                # dev_loss_list.append(dev_loss)
                output_dir_list.append(output_dir)

        if current_epoch <= args.num_train_epochs:
            if args.method != 'gpt':
                output_dir = os.path.join(args.output_dir, '{}/{}-prefixlen-{}-bs-{}-epoch-{}.pth'.format(
                    args.method, args.label, args.prefix_len, args.batch_size * args.gradient_accumulation_steps, current_epoch)
                )
                torch.save(model.state_dict(), output_dir)
            else:
                output_dir = os.path.join(args.output_dir, '{}-bs-{}-epoch-{}'.format(args.label, args.batch_size * args.gradient_accumulation_steps, current_epoch))
                model_to_save = (model.module if hasattr(model, 'module') else model)
                model_to_save.save_pretrained(output_dir)
                model_config.save_pretrained(output_dir)
            # dev_loss = evaluation(args=args, is_valid=True, model_name_or_path=output_dir)
            # dev_loss_list.append(dev_loss)
            output_dir_list.append(output_dir)

    logger.info(' global_step = %s, average loss = %s', global_step, tr_loss / global_step)
    '''
    min_loss = min(dev_loss_list)
    min_index = dev_loss_list.index(min_loss)
    optim_output_dir = output_dir_list[min_index]

    for output_dir in output_dir_list:
        if output_dir == optim_output_dir:
            continue
        else:
            shutil.rmtree(output_dir)
    
    test_loss = evaluation(args=args, is_valid=False, model_name_or_path=optim_output_dir)
    print("the test less on the best model: {}".format(test_loss))
    '''


if __name__ == "__main__":
    print('start:{}'.format(torch.__version__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../datasets/IMDb/IMDb.jsonl", type=str)
    parser.add_argument("--dev_path", default="../datasets/AGNEWS/train.jsonl", type=str)
    parser.add_argument("--test_path", default="../datasets/IMDb/IMDb.jsonl", type=str)
    parser.add_argument("--label", default="pos", type=str)
    parser.add_argument("--model_name_or_path", default="gpt2-medium", type=str)
    parser.add_argument("--output_dir", default="./output/", type=str)
    parser.add_argument("--method", default="gpt", type=str, help="gpt, prefix_tuning, prompt_tuning")
    # prefix tuning
    parser.add_argument("--prefix_len", default=10, type=int)
    parser.add_argument("--prefix_mid_size", default=500, type=int)
    # prompt tuning
    parser.add_argument("--reparameterize", action="store_true")
    parser.add_argument("--prompt_len", default=10, type=int)
    parser.add_argument("--prompt_mid_size", default=500, type=int)
    
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=2, type=int)
    parser.add_argument("--warmup_rate", default=0.1, type=float)
    parser.add_argument("--logging_steps", default=200, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--save_steps", default=10000, type=int)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--device", default='cuda', type=str)
    
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--nodes', default=1, type=int, metavar='N')
    parser.add_argument("--port", default='6677', type=str, help="port")#''''''
    args = parser.parse_args()

    '''os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    '''
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = args.port  可能会导致无法init

    args.world_size = args.gpus * args.nodes
    torch.distributed.init_process_group('nccl', init_method='env://')
    device = torch.device("cuda:{}".format(args.local_rank))
    args.device = device

    set_seed(args)
    main(args)
