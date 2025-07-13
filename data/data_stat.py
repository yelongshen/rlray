import os
import io
import pickle
import traceback
import copy
import datetime
import sys
import threading
import time
import random
import argparse
import signal
import psutil  # To check process status before killing
import re
import multiprocessing
import logging
import json
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

import numpy as np
from queue import Queue
from typing import List, Optional, Tuple, Union, Any, Dict, Optional
import concurrent.futures
from concurrent.futures import TimeoutError
from functools import partial
from dataclasses import dataclass
from collections import deque

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import glob
from packed_dataset import PackedDataset

import argparse
import pyarrow as pa
import pyarrow.ipc as ipc
import textwrap
import pyarrow.dataset as ds          # high-level dataset API


def load_sharded_dir(path: str):
    """
    Load every *.arrow / *.feather file in `path`
    as a single pyarrow.dataset.Dataset.
    """
    # Arrow auto-discovers all IPC-formatted files in the tree
    dataset = ds.dataset(path, format="ipc")   # ipc == Arrow / Feather V2
    return dataset

def peek_dataset(dataset, n=10):
    table = dataset.head(n)          # grabs first n rows efficiently
    print("\nSchema\n------")
    print(dataset.schema)            # unified schema across shards
    print(f"\nPreview (first {n} rows)")
    print(table.to_pandas())         # convert only the head to pandas 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="none", help="path to arrow dataset.")
    args = parser.parse_args()
    
    dset = load_sharded_dir(args.data)
    peek_dataset(dset, 10)

    #peek(args.data, 10)