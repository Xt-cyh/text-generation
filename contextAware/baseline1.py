import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss

from tqdm import tqdm, trange
import logging
import argparse
import random
import numpy as np
import pandas as pd
import json
import math
import os


SMALL_CONST = 1e-10
BIG_CONST = -1e15
labels = {
    'positive': 1, 'negative': 0, 'world':1,
    'sport': 2, 'business':3, 'science':4
}


class BaseLine1(nn.Module):
    '''
    @classmethod
    def from_pretrained()
    then use from_pretrained() to load model 
    see https://github.com/kipgparker/soft-prompt-tuning/blob/main/soft_embedding.py
    '''
    def __init__(self, decoder, decoder_tokenizer, args):
        super(PromptTuning, self).__init__()
        self.decoder = decoder  # GPT2LMHeadModel

        self.decoder_config = decoder.config
        self.decoder_num_layer = self.decoder_config.n_layer
        self.decoder_hidden_size = self.decoder_config.n_embd
        self.decoder_num_head = self.decoder_config.n_head

        self.decoder_tokenizer = decoder_tokenizer
        self.eos_token_id = decoder_tokenizer.convert_tokens_to_ids([decoder_tokenizer.eos_token])[0]
        self.pad_token_id = decoder_tokenizer.convert_tokens_to_ids([decoder_tokenizer.pad_token])[0]
        self.bos_token_id = decoder_tokenizer.convert_tokens_to_ids([decoder_tokenizer.bos_token])[0]

        self.args = args
        self.device = args.device

        if hasattr(self.decoder_config, 'prompt_dropout'):
            self.prompt_dropout = self.decoder_config.prompt_dropout
        else:
            self.prompt_dropout = 0.0
        self.dropout = nn.Dropout(self.prompt_dropout)

        # use gpt2 embedding layer: self.decoder.transformer.wte
        # initialize from vocab
        self.prompt_len = args.prompt_len
        init_prompt_value = self.decoder.transformer.wte.weight[:args.prompt_len].clone().detach()
        self.soft_prompt = nn.Embedding(args.prompt_len, self.decoder_hidden_size)
        # initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

        self.label = self.decoder_tokenizer(args.label, return_tensors='pt')
        # reparameterize (optional)
        self.reparameterize = args.reparameterize
        self.trans = nn.Sequential(
            nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
            nn.Tanh(),
            nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size)
        )

    def fix_decoder(self):
        '''
        Fix the decoder to work as prompt tuning.
        '''
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False

    def get_context_embedding(self, mode, label):
        datalist = []
        if mode == 'sentiment':
            with open('../../datasets/SST/SST.jsonl', 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data['label'] == 'label':
                        datalist.append(data['text'])

        choice = random.sample(datalist, self.args.sample_num)
        min_len = min(len(s) for s in choice)
        for i in range(self.args.sample_num):
            choice[i] = choice[i][:min_len]
        # tokenizer and embedding
        context_tokens = tokenizer(choice, return_tensors='pt')
        context_ids = context_tokens.input_ids
        embeddings = self.decoder.transformer.wte(context_ids)  # (self.args.sample_num, len, embed_size)
        avg_embedding = torch.mean(embeddings, dim=0)
        return avg_embedding

    def forward(self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        control_code=None
        ):
        batch_size = input_ids.shape[0]
        if attention_mask is None:
            prompt_attn = torch.ones(batch_size, self.prompt_len).bool().to(input_ids.device)
            attention_mask = torch.cat([prompt_attn, attention_mask], dim=1)

        # use inputs_embeds directly pass an embedded representation
        inputs_embeds = self.decoder.transformer.wte(input_ids)
        prompt_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)  # bsz prompt_len embed_size
        if self.reparameterize:
            prompt_embeds = self.trans(prompt_embeds)

        # detect whether prompt and context aligned
        # choose 25% train data, add context with contrary attribute between inputs embeds and prompt embeds
        self.context = False
        if random.random < 0.25:
            context_embeds = self.get_context_embedding('sentiment', labels[args.label], 5)
            inputs_embeds = torch.cat((context_embeds, inputs_embeds), dim=1)
            self.context = True
        inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)

        # GPT2 output: transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
        outputs = self.decoder(  
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, 
            past_key_values=past_key_values, return_dict=return_dict, # labels=input_ids, 
            use_cache=use_cache
        )
        return outputs

    def generate_forward(self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        control_code=None
        ):
        batch_size = input_ids.shape[0]
        if attention_mask is None:
            prompt_attn = torch.ones(batch_size, self.prompt_len).bool().to(input_ids.device)
            attention_mask = torch.cat([prompt_attn, attention_mask], dim=1)

        # use inputs_embeds directly pass an embedded representation
        inputs_embeds = self.decoder.transformer.wte(input_ids)
        prompt_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)  # bsz prompt_len embed_size
        if self.reparameterize:
            prompt_embeds = self.trans(prompt_embeds)

        inputs_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)

        # GPT2 output: transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
        outputs = self.decoder(  
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, 
            past_key_values=past_key_values, return_dict=return_dict, # labels=input_ids, 
            use_cache=use_cache
        )
        return outputs

    def generate(
        self,
        control_code=None,
        input_ids=None,
        attention_mask=None,
        batch_size=None,
        min_length=30,
        max_length=50,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=1.0,
        length_penalty=1,
        repetition_penalty=1.0,
        return_dict=True,
        use_cache=True
        ):
        '''
        Generate text
        '''
        device = next(self.parameters()).device

        if batch_size is None:
            batch_size = input_ids.shape[0]
            if input_ids is not None:
                if input_ids.shape[0] > batch_size:
                    if batch_size == 1:
                        batch_size = input_ids.shape[0]
                        input_latent = input_latent.expand(batch_size, -1)
                    else:
                        raise Exception('Batch size of input_latent and input_ids mismatched')
                elif input_ids.shape[0] < batch_size and input_ids.shape[0] == 1:
                    input_ids = input_ids.expand(batch_size, -1)
        # 需要改position_id?
        # input_shape = input_ids[:, self.prompt_length:].size()
        # position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=self.device)
        # position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])   

        output_ids = input_ids
        cur_len = input_ids.shape[1]
        past = None

        while cur_len <= max_length:
            if past == None:
                outputs = self.generate_forward(input_ids, attention_mask=attention_mask,
                                    control_code=control_code, return_dict=True, use_cache=True)
            else:
                outputs = self.decoder(input_ids, past_key_values=past, return_dict=True, use_cache=True)

            # The input_ids which have their past given to this model should not be passed as input_ids as they have already been computed.
            # If past_key_values is used, optionally only the last inputs_embeds have to be input
            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits_ = self.top_k_top_p_filtering(next_token_logits,  top_k=top_k, top_p=top_p, filter_value=BIG_CONST)
            next_token_logits_prob = torch.softmax(next_token_logits_, dim=1)
            next_tokens = torch.multinomial(next_token_logits_prob, num_samples=1).squeeze(1)
            output_ids = torch.cat([output_ids, next_tokens.unsqueeze(1)], dim=1)
            input_ids = next_tokens.unsqueeze(1)
            past = outputs.past_key_values
            cur_len += 1

        return output_ids

    def top_k_top_p_filtering(self,
        logits,
        top_k = 0,
        top_p = 1.0,
        filter_value = -1e15 ,
        min_tokens_to_keep = 1,
    ):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
            
        return logits