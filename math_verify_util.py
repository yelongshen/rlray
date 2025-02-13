import os
import sys

import subprocess
import pickle


def math_verify(answer, gold):
    escaped_answer = answer.replace('"', '\\"').replace("\\", "\\\\")
    escaped_gold = gold.replace('"', '\\"').replace("\\", "\\\\")
    command = ['python3.12', '-c', f'from math_verify import parse, verify; import pickle; import sys; pickle.dump(verify(parse("{escaped_answer}"), parse("{escaped_gold}")), sys.stdout.buffer)']
    o = False
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        o = pickle.load(proc.stdout)
    return o
    
