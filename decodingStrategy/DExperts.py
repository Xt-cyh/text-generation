import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, GPT2PreTrainedModel
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch import nn

from tqdm import tqdm, trange
from pathlib import Path
from typing import Union, List
import logging
import argparse
import random
import numpy as np
import pandas as pd
import json
import sys
import os

sys.path.append('..')
from utils.decode import *


STOP_TOKEN = "<|endoftext|>"


class DExperts(nn.Module): 
    def __init__(
        self,
        args,
        base_model: Union[str, Path, GPT2PreTrainedModel],
        antiexpert_model: Union[str, Path, GPT2PreTrainedModel] = None,
        expert_model: Union[str, Path, GPT2PreTrainedModel] = None,
        tokenizer: str = 'gpt2', 
    ):
        super(DExperts, self).__init__()
        # Set up device
        self.device = args.device
        self.base_model = GPT2LMHeadModel.from_pretrained(base_model).to(self.device)
        self.antiexpert = GPT2LMHeadModel.from_pretrained(antiexpert_model).to(self.device)
        self.expert = GPT2LMHeadModel.from_pretrained(expert_model).to(self.device)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = STOP_TOKEN
        assert self.tokenizer.eos_token_id == self.tokenizer.pad_token_id

    def __repr__(self):
        return f'<DExpertsGenerator model_name_or_path="{self.model}">'

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
        repetition_penalty=1.0,
        alpha=1,
        return_dict=True,
        use_cache=True
        ):
        device = next(self.parameters()).device
        batch_size, input_seq_len = input_ids.shape

        position_ids = attention_mask.cumsum(dim=1) - 1
        unfinished_sents = torch.ones(batch_size, dtype=torch.long, device=self.device)
        cur_len, step = input_ids.shape[1], 0
        context_len = input_ids.shape[1]

        self.base_model.eval()
        if self.expert:
            self.expert.eval()
        if self.antiexpert:
            self.antiexpert.eval()
        
        # generate length mode：1.fixed length with prompt 2.fixed length without prompt
        while cur_len <= max_length:
            step += 1
            # base model prediction
            output = self.base_model(
                input_ids, attention_mask=attention_mask, position_ids=position_ids, return_dict=True, use_cache=True)
            base_logits, base_past = output.logits, output.past_key_values

            # expert prediction
            if self.expert:
                expert_output = self.expert(
                input_ids, attention_mask=attention_mask, position_ids=position_ids, return_dict=True, use_cache=True)
                expert_logits, expert_past = expert_output.logits, expert_output.past_key_values
            else:
                expert_logits = base_logits

            # antiexpert prediction
            if self.antiexpert:
                antiexpert_output = self.antiexpert(
                input_ids, attention_mask=attention_mask, position_ids=position_ids, return_dict=True, use_cache=True)
                antiexpert_logits, antiexpert_past = antiexpert_output.logits, antiexpert_output.past_key_values
            else:
                antiexpert_logits = base_logits
            
            base_logits = top_k_top_p_filtering(base_logits, top_k=top_k, top_p=top_p)
            
            # DExperts
            alpha = torch.tensor(alpha).to(self.device)
            ensemble_logits = base_logits + alpha * (expert_logits - antiexpert_logits)

            # in the first decoding step, we want to use the 'real' last position for each sentence
            if step == 0:
                last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
                next_token_logits = ensemble_logits[range(batch_size), last_non_masked_idx, :]
            else:
                next_token_logits = ensemble_logits[:, -1, :]

            # repetition penalty
            if repetition_penalty != 1.0:
                enforce_repetition_penalty(
                    next_token_logits, batch_size, input_ids[:, -step:], repetition_penalty,
                )

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                if top_k > 0 or top_p < 1.0:
                    next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            '''
            # either append a padding token here if <EOS> has been seen or append next token
            tokens_to_add = next_tokens * unfinished_sents + self.tokenizer.pad_token_id * (1 - unfinished_sents)

            # this updates which sentences have not seen an EOS token so far
            # if one EOS token was seen the sentence is finished
            eos_in_sents = tokens_to_add == self.tokenizer.eos_token_id
            unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is an EOS in each sentence
            if unfinished_sents.max() == 0:
                break
            '''
            # Update input_ids, attention_mask and position_ids
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1)
            position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)
            cur_len += 1

        #decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        #                   for output in input_ids[:, input_seq_len:]]
        return input_ids
