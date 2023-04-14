from queue import Queue
import numpy as np
import collections
import itertools
import time

NO_TREE = 0
S_TREE = 1    # source tree id
T_TREE = 2    # sink tree id

NO_PARENT = -1

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)   


def GetPath(parent: {}, root1: int, root2: int, source_idx: int):

    tree1 = [root1]
    p = parent[root1]
    while(p != NO_PARENT):
        tree1.append(p)
        p = parent[p]

    tree2 = [root2]
    p = parent[root2]
    while(p != NO_PARENT):
        tree2.append(p)
        p = parent[p]

    # make sure path is from source -> sink
    if tree1[-1] == source_idx:
        path = tree1[::-1] + tree2
    else:
        path = tree2[::-1] + tree1
    return path


def GrowthStage(rGraph: collections.defaultdict(dict), activeNodes: Queue, tree: dict, parent: dict, source_idx: int) -> []:

    while not activeNodes.empty():
        p = activeNodes.get()

        if tree[p] == NO_TREE:
            continue

        for q in rGraph[p].keys():
            if rGraph[p][q] > 0:
                if tree[q] == NO_TREE:
                    activeNodes.put(q)
                    tree[q] = tree[p]
                    parent[q] = p
                elif tree[q] != NO_TREE and tree[q] != tree[p]:
                    return GetPath(parent, q, p, source_idx)
    return []


def AugmentationStage(path: [], rGraph: collections.defaultdict(dict), tree: dict, parent: dict, orphans: Queue):

    minFlow = float("inf")
    for p, q in pairwise(path):
        minFlow = min(minFlow, rGraph[p][q])

    if minFlow <= 0:
        return

    for p, q in pairwise(path):
        rGraph[p][q] -= minFlow
        rGraph[q][p] += minFlow

        if rGraph[q][p] >= 100:  # edge is saturated, label q as an orphan
            if tree[q] == S_TREE and tree[p] == S_TREE:
                parent[q] = NO_PARENT
                orphans.put(q)
            if tree[q] == T_TREE and tree[p] == T_TREE:
                parent[p] = NO_PARENT
                orphans.put(p)


def GetOrigin(parent: dict, node: int) -> bool:
    while parent[node] != NO_PARENT:
        node = parent[node]
    return node


def AdoptionStage(rGraph: collections.defaultdict(dict), tree: dict, parent: dict, orphans: Queue, activeNodes: Queue, source_idx: int, sink_idx: int):

    while not orphans.empty():
        p = orphans.get()

        # search for a valid new parent among neighbors that links back to source or sink
        foundParent = False
        for q in rGraph[p].keys():
            origin = GetOrigin(parent, q)
            if tree[q] == tree[p] and rGraph[q][p] > 0 and (origin == source_idx or origin == sink_idx):
                parent[p] = q
                foundParent = True
                break

        # otherwise mark it as not active and search for more orphans nearby
        if not foundParent:
            for q in rGraph[p].keys():
                if tree[q] == tree[p]:
                    if rGraph[q][p] > 0:
                        activeNodes.put(q)
                    if parent[q] == p:
                        orphans.put(q)
                        parent[q] = NO_PARENT
            tree[p] = NO_TREE  # this will mark p as no longer active in activeNodes


def DFS(rGraph, source_idx, visited):
    stack = [source_idx]
    while stack:
        p = stack.pop()
        if not visited[p]:
            visited[p] = True
            stack.extend([q for q in rGraph[p].keys() if rGraph[p][q] == 100])


def boykovKolmogorov(graph, source_idx, sink_idx):
    print("Running Boykov-Kolmogorov algorithm")

    # growth -> augmentation -> adoption

    rGraph = graph.copy()

    nNodes = len(graph)

    # set of active nodes
    activeNodes = Queue()
    activeNodes.put(source_idx)
    activeNodes.put(sink_idx)

    # which tree each node belongs to
    tree = {}
    for i in range(nNodes):
        tree[i] = NO_TREE
    tree[source_idx] = S_TREE
    tree[sink_idx] = T_TREE

    # parent of each node
    parent = {}
    for i in range(nNodes):
        parent[i] = NO_PARENT

    while(True):
        #print('grow')
        path = GrowthStage(rGraph, activeNodes, tree, parent, source_idx)
        if (path == []):
            break

        #print('augment')
        orphans = Queue()
        AugmentationStage(path, rGraph, tree, parent, orphans)

        #print('adopt')
        AdoptionStage(rGraph, tree, parent, orphans, activeNodes, source_idx, sink_idx)

    visited = np.zeros(nNodes, dtype=bool)
    DFS(rGraph, source_idx, visited)
    cuts = np.where(visited == False)[0]

    print("Done")
    return cuts
