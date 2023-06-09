from queue import Queue
import numpy as np
import collections
import time

def bfs(rGraph, nNodes, source_idx, parent):
    q = Queue()
    visited = np.zeros(nNodes, dtype=bool)
    q.put(source_idx)
    visited[source_idx] = True
    parent[source_idx]  = -1

    while not q.empty():
        u = q.get()

        for child in rGraph[u].keys():
            if (not visited[child]) and rGraph[u][child] > 0:
                q.put(child)
                parent[child] = u
                visited[child] = True

    return visited[child]

def dfs(rGraph, nNodes, source_idx, visited):
    stack = [source_idx]
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            stack.extend([u for u in rGraph[v].keys() if rGraph[v][u]])

def edmondsKarp(graph, source_idx, sink_idx):
    print("Running augmenting path algorithm")
    rGraph = graph.copy()

    nNodes = len(rGraph)
    parent = np.zeros(nNodes, dtype='int32')

    start = time.time()
    while bfs(rGraph, nNodes, source_idx, parent):
        pathFlow = float("inf")

        # walk back to get min flow
        v = sink_idx
        while v != source_idx:
            u = parent[v]
            pathFlow = min(pathFlow, rGraph[u][v])
            v = u

        v = sink_idx
        while v != source_idx:
            u = parent[v]
            rGraph[u][v] -= pathFlow
            rGraph[v][u] += pathFlow
            v = u

    print("BFS time: {}".format(time.time() - start))

    start = time.time()
    visited = np.zeros(nNodes, dtype=bool)
    dfs(rGraph, nNodes, source_idx, visited)

    print("DFS time: {}".format(time.time() - start))

    start = time.time()
    cuts = np.where(visited == False)[0]
    print("Cuts computation time: {}".format(time.time() - start))
    return cuts
