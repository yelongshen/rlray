
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

from collections import deque


def _test():
    batch_code = ["""
#print(input())
a, b = map(int, input().split())
print("Hello World!\n\n\n:", a, b, a+b)
"""
    ]

    input_data = '10 100'

    old_stdout = sys.stdout
    old_stdin = sys.stdin

    sys.stdout = io.StringIO()
    sys.stdin = io.StringIO(input_data)

    try:
        exec(batch_code[0])
        output = sys.stdout.getvalue()
    except EOFError:
        output = "Error: Unexpected end of input. Make sure to provide the necessary input."
    except Exception as e:
        output = f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout

    print("this is code output:", output)
    #executor = PythonExecutor(get_answer_from_stdout=True)
    #predictions = executor.apply(batch_code[0])
    #print(predictions)

if __name__ == '__main__':
    q = deque()

    q.append(99)
    q.append(999)
    print(q)

    _test()