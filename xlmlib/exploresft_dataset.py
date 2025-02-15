import os
import io
import sys
import threading
import time
import random
import multiprocessing
import logging
import json

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import IterableDataset as TorchIterableDataset


class SFTCorpus(object):
    def __init__(self, path):
        self.path = path
        self.num_words = 0        
        self.tokens = []
        with open(self.path, "r") as reader:
            for line in reader:
                items = json.loads(line.strip())
                book = items['book']
                tokens = items['tokens']
                num_words = items['num_words']

                self.num_words += num_words
                self.tokens.extend(tokens)

category = [0, 0, 0, 0, 0]
        for _b_idx, sft_d in enumerate(sft_dataloader):
            prompt = sft_d['messages'][0]['content'][0]
            completion = sft_d['messages'][1]['content'][0]

            completion_tokens = tokenizer([completion], add_special_tokens=False, max_length=32768, truncation=True)
            input_ids = completion_tokens['input_ids'][0] 
            
            if len(input_ids) < 4096:
                category[0] += 1
            elif len(input_ids) < 8192:
                category[1] += 1
            elif len(input_ids) < 16384:
                category[2] += 1
            elif len(input_ids) < 32768:
                category[3] += 1
        print('category', category, 'rank', rank)
        return
