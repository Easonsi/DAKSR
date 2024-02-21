from easonsi.util.leetcode import *
from easonsi import utils
from tqdm import tqdm
from argparse import Namespace
from multiprocessing import Pool
# import torch.sparse as sparse
import scipy.sparse as sp
import numpy as np
from time import time


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s cost time: %.3f s' % (func.__name__, time_spend))
        return result
    return func_wrapper


@timer
def build_dist_mat(kg_edges, num_entities, max_depth=5, default_value=100):
    """ BFS from every node, with depth limit
    """

    G = defaultdict(list)
    for h, t, r in kg_edges:
        G[h].append(t)
        G[t].append(h)

    def bfs(u,max_depth=max_depth, default_value=inf):
        d = defaultdict(lambda: default_value)
        d[u] = 0
        q = deque([u])
        vis = set([u])
        while q:
            u = q.popleft()
            if d[u] >= max_depth: break
            for v in G[u]:
                if v not in vis:
                    d[v] = d[u] + 1
                    q.append(v)
                    vis.add(v)
        return d

    dist = [None] * num_entities
    for u in tqdm(range(num_entities)):
        dist[u] = bfs(u, default_value=default_value)
    
    cols, rows, vals = [], [], []
    for u,dd in enumerate(dist):
        for v,d in dd.items():
            # dist_sp[u,v] = d + 1
            cols.append(u)
            rows.append(v)
            vals.append(d + 1)
    dist_sp = sp.coo_matrix((vals, (cols, rows)), shape=(num_entities, num_entities))
    
    return dist_sp


def bfs(u):
    """ default_value: unreachable within depth limit
    """
    global args_src
    
    d = [args_src.default_value] * (args_src.num_entities+1)
    d[u] = 1     # NOTE: start with 1 (bias)
    q = deque([u])
    vis = set([u])
    while q:
        u = q.popleft()
        if d[u] >= args_src.max_depth: break
        for v in args_src.G[u]:
            if v not in vis:
                d[v] = d[u] + 1
                q.append(v)
                vis.add(v)
    return d[:args_src.num_items+1] + [args_src.default_value]       # NOTE: add last mask_token

@timer
def build_dist_mat_mp(kg_edges, num_entities, num_items, max_depth=3, default_value=100, workers=32, relational=False):
    def worker_init(args, ):
        global args_src
        args_src = args

    def apply_multiprocess(args, lines, workers):
        # create and configure the process pool
        with Pool(workers, initializer=worker_init, initargs=(args, )) as pool:
            result_lines = list(tqdm(pool.imap(bfs, lines), total=len(lines)))
            # pool.close()
            # pool.join()
        return result_lines

    def _build_single_mat():
        G = defaultdict(list)
        for h, t, r in kg_edges:
            G[h].append(t)
            G[t].append(h)
        args = Namespace(
            G=G,
            default_value=default_value,
            max_depth=max_depth,
            num_entities=num_entities,
            num_items=num_items,
        )
        dist = apply_multiprocess(args=args, lines=list(range(num_items+1)), workers=workers)
        dist.append([args.default_value] * (num_items+2) )
        dist[-1][-1] = 1
        dist = np.array(dist, dtype=np.int8)
        return dist

    def _build_relation_mats():
        rids = [r for h,t,r in kg_edges]
        num_relations = max(rids)
        dists = []
        for rid in tqdm(range(num_relations + 1)):
            G = defaultdict(list)
            for h, t, r in kg_edges:
                if r == rid:
                    G[h].append(t)
                    G[t].append(h)
            args = Namespace(
                G=G,
                default_value=default_value,
                max_depth=max_depth,
                num_entities=num_entities,
                num_items=num_items,
            )
            t_start = time()
            dist = apply_multiprocess(args=args, lines=list(range(num_items+1)), workers=workers)
            dist.append([args.default_value] * (num_items+2) )
            dist[-1][-1] = 1
            dists.append(dist)
        dists = np.array(dists, dtype=np.int8)
        return dists


    if relational:
        return _build_relation_mats()
    else:
        return _build_single_mat()

    dist_sp = np.array(dist, dtype=np.int8)
    return dist_sp

if __name__ == "__main__":
    kg_edges = [
        (1,2,2), (2,3,2),
        (1,4,1), (4,5,1),
        # (1,6,3), (6,7,3),
        # (1,8,4), (8,9,4),
        
    ]
    dist_mat = build_dist_mat_mp(kg_edges, num_entities=10, num_items=3, max_depth=3, default_value=100, workers=8)
    print(dist_mat.shape)
    print(dist_mat)
