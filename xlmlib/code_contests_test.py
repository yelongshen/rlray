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
    default_imports = """
import math
import os
import sys
"""

    old_stdin = sys.stdin

    dataset = load_dataset("deepmind/code_contests")
    train = dataset['train']

    for i in range(0, len(train)):
        example = train[i]
        soluts = example['solutions']
        pycode = ''
        status = 0
        for (lang, code) in zip(soluts['language'], soluts['solution']):
            if lang == 3:    
                pycode = default_imports + code
                status = 1
                break

        if status == 1:
            print('--------------------------------------------\n')
            print(pycode)

            tests = example['public_tests']
            for test_input, test_output in zip(tests['input'], tests['output']):
                print('------------test input-------------\n')
                print(test_input)

                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                
                sys.stdin = io.StringIO(test_input)
                exec(pycode, globals())
                output = sys.stdout.getvalue()
                sys.stdout = old_stdout
                
                print('--------------------------------------------\n')
                print("gold output:", test_output)
                print("exec output:", output)

        #if status == 1:
        #    break
if __name__ == '__main__':
    _test()