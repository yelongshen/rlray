import inspect
import math
import warnings
import json
from typing import List, Optional, Tuple, Union
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from einops import rearrange, repeat
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.checkpoint import checkpoint
#import .checkpoint as user_checkpoint

from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers import AutoTokenizer 


# distributed inference engine.
# usage:
# on multinode:
# _engine = _inference_engine(model, rank, world_size)
# _engine.start() # start on thread. 
# on rank_0: 
#    load prompt set
#    for prompt 
#    _engine.pool.add(request)
#    
# on rank_0:
# for 
# processing _engine.results
class _inference_engine:
    def __init__(self, model, rank, world_size):
        self.model = model
        self.rank = rank
        self.world_size = world_size

    def start(self):

        
        
