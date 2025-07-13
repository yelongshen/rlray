# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

import json
import glob
import os
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count
import random

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from packed_dataset import PackedDataset, PackedDatasetBuilder

#from lit_gpt.tokenizer import Tokenizer

from transformers import AutoTokenizer

import pandas as pd

import pyarrow as pa

import pyarrow.ipc as ipc

import argparse

def prepare_full(
    source_path: Path,
    destination_path: Path,
    chunk_size: int,
    filenames_subset: List[str] = None,
    process_id: int = 0
) -> None:


    phi35_tok = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct", trust_remote_code=True)    # custom tokenizer class)

    #tokenizer = Tokenizer(tokenizer_path)
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset 

    builder = PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"phi4_2b_middata_{process_id}",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=phi35_tok.eos_token_id,
        dtype="auto",
        vocab_size=phi35_tok.vocab_size,
    )

    #load_ckpt = builder._load_ckpt()
    #if load_ckpt:
    #    processed_samples = builder._total_samples - 1
    #else:
    processed_samples = -1

    success_count = 0
    num_samples = 0
    for filepath in filenames:
        print(f"Processing {filepath}")
        
        # Determine file type and process accordingly
        #file_ext = Path(filepath).suffix.lower()
        
        with open(filepath, 'rb') as f:
            reader = ipc.RecordBatchStreamReader(f)
            table = reader.read_all()

            try:
                for text_ids in table['input_ids']:
                    if num_samples < processed_samples:
                        num_samples += 1
                        continue
                    else:
                        # print("Start processing new samples")
                        num_samples = 0
                        processed_samples = -1
                    
                    #if text and isinstance(text, str) and text.strip():
                    #    text_ids = tokenizer.encode(text)
                    builder.add_array(np.array(text_ids.values.tolist(), dtype=builder.dtype))
                success_count += 1
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue

    print(f"Processed ID {process_id} Processed {success_count} rows from {len(filenames)} files.")
    # we throw away the final corpus to avoid meaningless corpus filled with bos_ids, see https://github.com/jzhang38/TinyLlama/issues/83 for more details
    # builder.write_reminder()


def prepare(
    source_path: Path = Path("../../m7_1/m7_1/"),
    destination_path: Path = None,
    chunk_size: int = 2049 * 8192,
    nproc: int = None,
    file_types: List[str] = None,
) -> None:
    import time


    if destination_path is None:
        destination_path = Path("../../m7_1/m7_1_2b_midtrain")
    else:
        destination_path = Path(destination_path)
    destination_path.mkdir(parents=True, exist_ok=True)

    # Default file types to process
    if file_types is None:
        file_types = ["*.parquet", "*.jsonl", "*.arrow"]
    
    # Find files of specified types
    filenames = []
    for file_type in file_types:
        pattern = os.path.join(source_path, "**", file_type)
        found_files = glob.glob(pattern, recursive=True)
        filenames.extend(found_files)
    
    print(filenames)
    filenames = sorted(filenames)
    print(f"Found {len(filenames)} files in {source_path}")
    print(f"File types: {file_types}")
    print("Sample files:", filenames[:10])
    
    random.seed(43)                           # ② set the seed
    random.shuffle(filenames)
    
    if nproc is not None:
        num_processes = nproc
    else:
        num_processes = cpu_count()
    chunked_filenames = np.array_split(filenames, num_processes)
    # print(chunked_filenames)
    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        if len(list(subset)) == 0:
            print(f"Skipping empty subset for process {i}")
            continue
        p = Process(target=prepare_full, args=(source_path, destination_path, chunk_size, list(subset), i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    


if __name__ == "__main__":
    #from jsonargparse import CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_data", type=str, default="none", help="path to arrow dataset.")
    parser.add_argument("--tgt_data", type=str, default="none", help="path to bin dataset.")
    args = parser.parse_args()

    prepare(args.src_data, args.tgt_data)
    #CLI(prepare)