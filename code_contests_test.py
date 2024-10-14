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
from collections import deque
d = deque()
d.append(1)
d.append(2)

from collections import deque

def solve(adj, m, k, uv):
    n = len(adj)
    nn = [len(a) for a in adj]
    q = deque()
    for i in range(n):
        if nn[i] < k:
            q.append(i)
    while q:
        v = q.popleft()
        for u in adj[v]:
            nn[u] -= 1
            if nn[u] == k-1:
                q.append(u)
    res = [0]*m
    nk = len([1 for i in nn if i >= k])
    res[-1] = nk
    for i in range(m-1, 0, -1):
        u1, v1 = uv[i]

        if nn[u1] < k or nn[v1] < k:
            res[i - 1] = nk
            continue
        if nn[u1] == k:
            q.append(u1)
            nn[u1] -= 1
        if not q and nn[v1] == k:
            q.append(v1)
            nn[v1] -= 1

        if not q:
            nn[u1] -= 1
            nn[v1] -= 1
            adj[u1].remove(v1)
            adj[v1].remove(u1)

        while q:
            v = q.popleft()
            nk -= 1
            for u in adj[v]:
                nn[u] -= 1
                if nn[u] == k - 1:
                    q.append(u)
        res[i - 1] = nk
    return res

n, m, k = map(int, input().split())
a = [set() for i in range(n)]
uv = []
for i in range(m):
    u, v = map(int, input().split())
    a[u - 1].add(v - 1)
    a[v - 1].add(u - 1)
    uv.append((u-1, v-1))

res = solve(a, m, k, uv)
print(str(res)[1:-1].replace(' ', '').replace(',', '/\n'))
"""

    old_stdin = sys.stdin

    dataset = load_dataset("deepmind/code_contests")
    train = dataset['train']

    status = 0
    for i in range(0, len(train)):
        example = train[i]
        soluts = example['solutions']
        pycode = ''
        for (lang, code) in zip(soluts['language'], soluts['solution']):
            if lang == 3:    
                pycode = default_imports # + code
                status = 1
                break

        if status == 1:
            print('--------------------------------------------\n')
            print(pycode)

            tests = example['public_tests']
            for test_input, test_output in zip(tests['input'], tests['output']):
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                
                sys.stdin = io.StringIO(test_input)
                exec(pycode)
                output = sys.stdout.getvalue()
                sys.stdout = old_stdout
                
                print('--------------------------------------------\n')
                print("gold output:", test_output)
                print("exec output:", output)

        if status == 1:
            break
if __name__ == '__main__':
    _test()