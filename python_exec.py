
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


def _test():
    batch_code = [
        """
print("Hello World!")
        """
    ]

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    exec(batch_code[0])
    output = new_stdout.getvalue()
    
    print("this is code output:", output)
    #executor = PythonExecutor(get_answer_from_stdout=True)
    #predictions = executor.apply(batch_code[0])
    #print(predictions)

if __name__ == '__main__':
    _test()