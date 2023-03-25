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


class PrefixGPT2(nn.Module):

    def __init__(self, decoder, decoder_tokenizer, args):
        super(PrefixGPT2, self).__init__()
        self.decoder = decoder #GPT2LMHeadModel

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

        if hasattr(self.decoder_config, 'prefix_dropout'):
            self.prefix_dropout = self.decoder_config.prefix_dropout
        else:
            self.prefix_dropout = 0.0
        self.dropout = nn.Dropout(self.prefix_dropout)

        self.prefix_len = args.prefix_len
        self.prefix_mid_size = args.prefix_mid_size
        self.input_tokens = torch.arange(self.prefix_len).long()
        self.prefix_emb = nn.Embedding(self.prefix_len, self.decoder_config.n_embd)
        self.trans = nn.Sequential(
            nn.Linear(self.decoder_hidden_size, self.prefix_mid_size),
            nn.Tanh(),
            nn.Linear(self.prefix_mid_size, self.decoder_num_layer * 2 * self.decoder_hidden_size)
        )

    def fix_decoder(self):
        '''
        Fix the decoder to work as prefix tuning.
        '''
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False

    def get_prompt(self, control_code=None, batch_size=None):
        if control_code is None:
            # 默认control code
            input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        else:
            # 任务对应control code
            context_tokens = self.decoder_tokenizer(control_code, return_tensors='pt')
            input_tokens = context_tokens.input_ids.expand(batch_size, -1).to(self.device)

        temp_control = self.prefix_emb(input_tokens)
        past_key_values = self.trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.decoder_num_layer * 2, self.decoder_num_head, (self.decoder_hidden_size // self.decoder_num_head)
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self,
        input_ids,
        attention_mask=None,
        use_cache=True,
        return_dict=True,
        control_code=None
        ):

        batch_size = input_ids.shape[0]
        if attention_mask is None:
            prefix_attn = torch.ones(batch_size, self.prefix_len).bool().to(input_ids.device)
            attention_mask = torch.cat([prefix_attn, attention_mask], dim=1)
        past_key_values = self.get_prompt(control_code=control_code, batch_size=batch_size)

        # GPT2 output: transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
        outputs = self.decoder(
            input_ids=input_ids, attention_mask=attention_mask, 
            labels=input_ids, past_key_values=past_key_values, return_dict=return_dict,
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
        length_penalty=1,
        repetition_penalty=1.0,
        use_cache=True
        ):
        '''
        Generate text with given latent represention.
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

        past_key_values = self.get_prompt(control_code=control_code, batch_size=batch_size)

        if input_ids is None:
            input_ids = self.decoder.generate(input_ids=torch.LongTensor([[50256]]*batch_size).to(device), max_length=3, do_sample=True)[:,1:]
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, 2).bool()
                prefix_attn = torch.ones(batch_size, self.prefix_len).bool().to(device)
                attention_mask = torch.cat([prefix_attn, attention_mask.to(device)], dim=-1)
        else:
            input_ids = input_ids.to(device)
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, input_ids.shape[1]).bool()
                prefix_attn = torch.ones(batch_size, self.prefix_len).bool().to(device)
                attention_mask = torch.cat([prefix_attn, attention_mask.to(device)], dim=-1)

        cur_len = input_ids.shape[1]
        if cur_len < 1:
            raise Exception('input length error')
        if cur_len == 1:
            result = self.decoder.generate(
                input_ids=input_ids, past=past_key_values, attention_mask=attention_mask, repetition_penalty=repetition_penalty,\
                do_sample=do_sample, top_k=top_k, top_p=top_p, length_penalty=length_penalty, max_length=max_length, min_length=min_length, use_cache=use_cache
            )
        else:
            past_key_values = self.decoder(input_ids=input_ids[:,:-1], attention_mask=attention_mask[:,:-1], past_key_values=past_key_values, return_dict=True, use_cache=True).past_key_values
            result = self.decoder.generate(
                input_ids=input_ids, past=past_key_values, attention_mask=attention_mask, repetition_penalty=repetition_penalty,\
                do_sample=do_sample, top_k=top_k, top_p=top_p, length_penalty=length_penalty, max_length=max_length, min_length=min_length, use_cache=use_cache
            )

        return result


class PromptTuning(nn.Module):

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

        if hasattr(self.decoder_config, 'prefix_dropout'):
            self.prefix_dropout = self.decoder_config.prefix_dropout
        else:
            self.prefix_dropout = 0.0
        self.dropout = nn.Dropout(self.prefix_dropout)

        self.prefix_len = args.prefix_len
        self.prefix_mid_size = args.prefix_mid_size
        self.input_tokens = torch.arange(self.prefix_len).long()
        self.prefix_emb = nn.Embedding(self.prefix_len, self.decoder_config.n_embd)
        self.trans = nn.Sequential(
            nn.Linear(self.decoder_hidden_size, self.prefix_mid_size),
            nn.Tanh(),
            nn.Linear(self.prefix_mid_size, self.decoder_num_layer * 2 * self.decoder_hidden_size)
        )

    def fix_decoder(self):
        '''
        Fix the decoder to work as prefix tuning.
        '''
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False

    def get_prompt(self, control_code=None, batch_size=None):
        if control_code is None:
            # 默认control code
            input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        else:
            # 任务对应control code
            context_tokens = self.decoder_tokenizer(control_code, return_tensors='pt')
            input_tokens = context_tokens.input_ids.expand(batch_size, -1).to(self.device)

        temp_control = self.prefix_emb(input_tokens)
        past_key_values = self.trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.decoder_num_layer * 2, self.decoder_num_head, (self.decoder_hidden_size // self.decoder_num_head)
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self,
        input_ids,
        attention_mask=None,
        use_cache=True,
        return_dict=True,
        control_code=None
        ):

        batch_size = input_ids.shape[0]
        if attention_mask is None:
            prefix_attn = torch.ones(batch_size, self.prefix_len).bool().to(input_ids.device)
            attention_mask = torch.cat([prefix_attn, attention_mask], dim=1)
        past_key_values = self.get_prompt(control_code=control_code, batch_size=batch_size)

        # GPT2 output: transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
        outputs = self.decoder(
            input_ids=input_ids, attention_mask=attention_mask, 
            labels=input_ids, past_key_values=past_key_values, return_dict=return_dict,
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
        length_penalty=1,
        repetition_penalty=1.0,
        use_cache=True
        ):
        '''
        Generate text with given latent represention.
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

        past_key_values = self.get_prompt(control_code=control_code, batch_size=batch_size)

        if input_ids is None:
            input_ids = self.decoder.generate(input_ids=torch.LongTensor([[50256]]*batch_size).to(device), max_length=3, do_sample=True)[:,1:]
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, 2).bool()
                prefix_attn = torch.ones(batch_size, self.prefix_len).bool().to(device)
                attention_mask = torch.cat([prefix_attn, attention_mask.to(device)], dim=-1)
        else:
            input_ids = input_ids.to(device)
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, input_ids.shape[1]).bool()
                prefix_attn = torch.ones(batch_size, self.prefix_len).bool().to(device)
                attention_mask = torch.cat([prefix_attn, attention_mask.to(device)], dim=-1)

        cur_len = input_ids.shape[1]
        if cur_len < 1:
            raise Exception('input length error')
        if cur_len == 1:
            result = self.decoder.generate(
                input_ids=input_ids, past=past_key_values, attention_mask=attention_mask, repetition_penalty=repetition_penalty,\
                do_sample=do_sample, top_k=top_k, top_p=top_p, length_penalty=length_penalty, max_length=max_length, min_length=min_length, use_cache=use_cache
            )
        else:
            past_key_values = self.decoder(input_ids=input_ids[:,:-1], attention_mask=attention_mask[:,:-1], past_key_values=past_key_values, return_dict=True, use_cache=True).past_key_values
            result = self.decoder.generate(
                input_ids=input_ids, past=past_key_values, attention_mask=attention_mask, repetition_penalty=repetition_penalty,\
                do_sample=do_sample, top_k=top_k, top_p=top_p, length_penalty=length_penalty, max_length=max_length, min_length=min_length, use_cache=use_cache
            )

        return result