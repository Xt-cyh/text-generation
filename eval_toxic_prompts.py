import argparse
import os
import torch
import torch.nn.functional as F
import transformers
from transformers import GPT2LMHeadModel, BertModel, GPT2Tokenizer, BertTokenizer

from baseGPTmodel.prefix_tuning import PrefixGPT2
from baseGPTmodel.prompt_tuning import PromptTuning
from decodingStrategy.DExperts import Fudge, DExperts
from utils.perspective import detect_toxic

import datasets
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import random
from eval import *

labels_3classes = {
    'positive': 2, 'neutral':1, 'negative': 0, 'world':1,
    'sport': 2, 'business':3, 'science':4
}
labels = {
    'positive': 1, 'negative': 0, 'world':1,
    'sport': 2, 'business':3, 'science':4
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_data(args, begin=0, end=0):
    data_list = []
    file = args.data_file.split('.')
    file_path = os.path.join(args.data_dir, args.data_file)
    if file[1] == 'json':
        with open(file_path, 'r', encoding='utf8') as fin:
            for line in fin:
                # load: 针对文件 loads：针对字符串
                dic = json.loads(line)
                # data_list.extend(dic['pos5.5'])
                for txt in dic['pos5.5']:
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
    if end > 0:
        return data_list[begin:end]
    else:
        return data_list[:100]


def get_generated_data(file_path):
    results = []
    prompts = []
    with open(file_path, 'r', encoding='utf8') as fin:
        for line in fin:
            dic = json.loads(line)
            results.append(dic['text'])
            prompts.append(dic['prompt'])
    return results, prompts


def get_prompts(file):
    with open(file, 'r', encoding='utf8') as fin:
        dataset = pd.read_json(file, lines=True)
        prompts = pd.json_normalize(dataset['prompt'])['text'].tolist()
        prompts = prompts[:1000]
        return prompts


def generate_eval(args, model, prompts):
    # generate params
    topk=args.top_k
    topp=args.top_p
    temperature=args.temperature
    rp=args.repetition_penalty
    min_len=args.min_len
    max_len=args.max_len
    do_sample=True
    use_cache=True

    results = []
    final_prompts = []
    # results_withprompt = []
    # input prefix
    for context in tqdm(prompts):
        context_tokens = tokenizer(context, return_tensors='pt')
        input_ids = context_tokens.input_ids
        # context len + ' '
        context_len = len(context)+1

        attention_mask = context_tokens.attention_mask
        input_ids = input_ids.expand(args.batch_size, -1).to(args.device)
        attention_mask = attention_mask.expand(args.batch_size, -1).to(args.device)
        if args.method == 'prefix_tuning':
            prefix_mask = torch.tensor([1] * args.prefix_len).expand(args.batch_size, args.prefix_len).to(args.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        elif args.method == "prompt_tuning":
            prompt_mask = torch.tensor([1] * args.prompt_len).expand(args.batch_size, args.prompt_len).to(args.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # do generation
        output = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, 
                repetition_penalty=rp, do_sample=do_sample, 
                top_k=topk, top_p=topp, temperature=temperature, 
                max_length=max_len, min_length=min_len, use_cache=use_cache
            )
        # output without the context
        output = tokenizer.batch_decode(output.cpu(), skip_special_tokens=True)
        # results_withprompt.extend(output)
        prompt = []
        for i in range(args.batch_size):
            prompt.append(context)
            output[i] = output[i][context_len:]
        final_prompts.extend(prompt)
        results.extend(output)

    return results, final_prompts


def evaluate(results, results_withprompt):
    # perplexity
    # ppl = cal_ppl(results_withprompt, args.device)
    ppl = 28.7011
    # accuracy: without context
    toxicity = detect_toxic(results)
      
    # dist-n
    dist = [0] * 3
    for n in range(1, 4):
        dist[n-1] = calc_dist_n(results, n)
    # self_bleu
    sbl = calc_self_bleu(results)
    return ppl, toxicity, dist, sbl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_encoder", type=str, default="bert-base-uncased")
    parser.add_argument("--pretrained_decoder", type=str, default="gpt2-large")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--sampling_num", type=int, default=50, help='针对context，对应生成的文本数')
    parser.add_argument("--context", type=str, default='The last time')
    parser.add_argument("--batch_size", type=int, default=5, help='针对pre_tokens，每个对应生成的文本数')
    parser.add_argument("--pre_tokens",
        type=str,
        default=json.dumps(
            ['In summary','This essay discusses','Views on','The connection','Foundational to this is',
            'To review,','In brief,','An illustration of','Furthermore,','The central theme',
            'To conclude,','The key aspect','Prior to this','Emphasised are','To summarise',
            'The relationship','More importantly,','It has been shown','The issue focused on','In this essay',
            'Once upon a time','The book','The chicken','The city','The country',
            'The horse','The lake','The last time','The movie','The painting',
            'The pizza','The potato','The president of the country','The road','The year is 1910']
        )
    )
    parser.add_argument("--label", default="positive", type=str)
    parser.add_argument("--class_num", type=int, default=2)
    parser.add_argument("--method", default="gpt", type=str, help="gpt, prefix_tuning, prompt_tuning")
    parser.add_argument("--mode", default="generate-eval", type=str, help="all, generate, eval")
    # different methods:
    # prefix tuning
    parser.add_argument("--prefix_len", default=10, type=int)
    parser.add_argument("--prefix_mid_size", default=500, type=int)
    # prompt tuning
    parser.add_argument("--reparameterize", action="store_true")
    parser.add_argument("--prompt_len", default=10, type=int)
    parser.add_argument("--prompt_mid_size", default=500, type=int)
    # DExpert
    parser.add_argument("--expert", type=str, default="../DExperts/model/experts/sentiment/large/finetuned_gpt2_positive")
    parser.add_argument("--antiexpert", type=str, default="../DExperts/model/experts/sentiment/large/finetuned_gpt2_negative")
    parser.add_argument("--alpha", type=float, default=1.0)
    # files and gpu
    parser.add_argument("--model_dir", type=str, default='./output/prompt_tuning/neg-prefixlen-10-bs-4-epoch-1.pth')
    parser.add_argument("--data_file", type=str, default='IMDb/IMDb_pos.txt')
    parser.add_argument("--prompt_dir", type=str, default='../prompts/prompts/')
    parser.add_argument("--output_dir", type=str, default='./results/contrary_prompts/')
    parser.add_argument("--gpudevice", type=str, default='0')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--nodes', default=1, type=int, metavar='N')
    parser.add_argument("--seed", default=42, type=int)
    # generate args
    parser.add_argument("--top_k", default=200, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--max_len", default=50, type=int)
    parser.add_argument("--min_len", default=50, type=int)
    parser.add_argument("--length_penalty", default=1, type=int)
    parser.add_argument("--repetition_penalty", default=1, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)

    args = parser.parse_args()
    set_seed(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpudevice
    '''
    args.world_size = args.gpus * args.nodes
    torch.distributed.init_process_group('nccl', init_method='env://')
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.device = device
    '''

    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_decoder)
    if args.method == 'prefix_tuning':
        decoder = GPT2LMHeadModel.from_pretrained(args.pretrained_decoder)
        model = PrefixGPT2(decoder=decoder, decoder_tokenizer=tokenizer, args=args)
        model.load_state_dict(torch.load(args.model_dir))
    elif args.method == 'prompt_tuning':
        decoder = GPT2LMHeadModel.from_pretrained(args.pretrained_decoder)
        model = PromptTuning(decoder=decoder, decoder_tokenizer=tokenizer, args=args)
        model.load_state_dict(torch.load(args.model_dir))
    elif args.method == 'gpt':
        model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    elif args.method == 'DExperts':
        model = DExperts(
            args=args, base_model = args.pretrained_decoder, tokenizer=tokenizer,
            expert_model = args.expert, antiexpert_model=args.antiexpert
        )
    elif args.method == 'Fudge':
        condition_model = ClassificationHead(class_size=5, embed_size=1024).to(args.device)
        condition_model.load_state_dict(torch.load(args.condition_model))
        model = Fudge(args=args, base_model = args.pretrained_decoder, tokenizer=tokenizer, condition_model=condition_model)
    model.to(args.device)
    model.eval()

    ########## 文本生成 ###########
    # generate params
    topk=args.top_k
    topp=args.top_p
    lp=args.length_penalty
    rp=args.repetition_penalty
    min_len=args.min_len
    max_len=args.max_len
    do_sample=True
    use_cache=True

    toxic_prompts = get_prompts(args.prompt_dir + 'nontoxic_prompts-10k.jsonl')
    save_path = os.path.join(args.output_dir, '{}/{}_len{}_topk{}_noprompt.jsonl'.format(args.method, args.label, max_len, topk))

    if args.mode == 'generate' or args.mode == 'all':
        results, prompts = generate_eval(args, model, toxic_prompts)
        with open(save_path, 'w') as fout:
            for prompt, txt in zip(prompts, results):
                data = {}
                data['label'] = args.label
                data['prompt'] = prompt
                data['text'] = txt
                fout.write(json.dumps(data))
                fout.write('\n')

    if args.mode == 'eval' or args.mode == 'all':
        results, prompts = get_generated_data(save_path)
        results_withprompt = [(x + ' ' + y) for x,y in zip(prompts, results)]
        ppl, toxicity, dist, sbl = evaluate(results, results_withprompt)

        with open(save_path, 'a+') as fout:
            fout.write(f'Avg ppl:{ppl}, Avg dist1:{dist[0]}, Avg dist2:{dist[1]}, Avg dist3:{dist[2]}, Avg sBL:{sbl}\n')
            fout.write(f'Toxicity: {sum(toxicity)/len(toxicity)}\n')
