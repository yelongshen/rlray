import math
import os
import sys
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
print(str(res)[1:-1].replace(' ', '').replace(',', '\n'))
