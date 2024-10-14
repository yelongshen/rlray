import os
import io
import pickle
import traceback
import copy
import datetime
from typing import Any, Dict, Optional
from concurrent.futures import TimeoutError
from functools import partial
from contextlib import redirect_stdout
import sys

from datasets import load_dataset


def _test():
    dataset = load_dataset("deepmind/code_contests")
    train = dataset['train']

    idx = 0
    for i in range(0, len(train)):
        example = train[i]
        soluts = example['solutions']
        for (lang, code) in zip(soluts['language'], soluts['solution']):
            if lang == 3:
                print(example['description'], code)
                idx = 1
                break
        
        if idx == 1:
            break

if __name__ == '__main__':
    _test()