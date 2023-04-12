from queue import Queue
import numpy as np
import time
import cv2

def bfs(rGraph, nNodes, source_idx, parent):
    q = Queue()
    visited = np.zeros(nNodes, dtype=bool)
    q.put(source_idx)
    visited[source_idx] = True
    parent[source_idx]  = -1

    while not q.empty():
        u = q.get()
        for v in range(nNodes):
            if (not visited[v]) and rGraph[u][v] > 0:
                q.put(v)
                parent[v] = u
                visited[v] = True

    return visited[v]

def dfs(rGraph, nNodes, source_idx, visited):
    stack = [source_idx]
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            stack.extend([u for u in range(nNodes) if rGraph[v][u]])

def augmentingPath(graph, source_idx, sink_idx):
    print("Running augmenting path algorithm")
    rGraph = graph.copy()
    nNodes = graph.shape[0]
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
    cuts = []
    for i in range(nNodes):
        for j in range(nNodes):
            if not visited[i] and not visited[j] and graph[i][j]:
                cuts.append((i, j))
    print("Cuts computation time: {}".format(time.time() - start))
    return cuts
