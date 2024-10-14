
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



def _test():
    batch_code = [
        """
        print("Hello World!")
        """
    ]

    result = exec(batch_code[0])

    print(result)
    #executor = PythonExecutor(get_answer_from_stdout=True)
    #predictions = executor.apply(batch_code[0])
    #print(predictions)

if __name__ == '__main__':
    _test()