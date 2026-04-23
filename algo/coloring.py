import numpy as np
from scipy.sparse import csr_matrix
from numba import njit, prange, set_num_threads
import time

import os
set_num_threads(min(16, os.cpu_count() or 1))

@njit(parallel=True, cache=True)
def update_partition_parallel(partition, indptr, indices, data, scores, dangling_term, rsp, n):
    out = np.empty(partition.size, dtype=np.float64)

    for k in prange(partition.size):
        j = partition[k]
        s = 0.0

        # Sum incoming contribution for node j from CSC column j
        for p in range(indptr[j], indptr[j + 1]):
            i = indices[p]
            s += data[p] * scores[i]

        out[k] = (1.0 - rsp) * (s + dangling_term) + rsp / n

    return out


@njit(cache=True)
def greedy_color_numba(indptr, indices, order, max_deg):
    n = indptr.size - 1
    colors = -np.ones(n, np.int32)
    marks = np.zeros(max_deg + 1, np.int32)
    stamp = 1

    for t in range(order.size):
        v = order[t]
        for p in range(indptr[v], indptr[v + 1]):
            c = colors[indices[p]]
            if c >= 0:
                marks[c] = stamp

        c = 0
        while marks[c] == stamp:
            c += 1

        colors[v] = c
        stamp += 1

    return colors

def fast_coloring_csr(matrix: csr_matrix):
    sym = (matrix + matrix.T).tocsr()
    sym.data[:] = 1

    indptr = sym.indptr
    indices = sym.indices
    n = sym.shape[0]

    deg = np.diff(indptr)

    if n == 0 or deg.size == 0:
        return []
    if deg.max() == 0:
        return [list(range(n))]

    order = np.argsort(-deg)   # high-degree first
    max_deg = int(deg.max())

    colors = greedy_color_numba(indptr, indices, order, max_deg)

    num_colors = int(colors.max()) + 1
    partitions = [[] for _ in range(num_colors)]
    for v, c in enumerate(colors):
        partitions[c].append(v)

    return partitions
def pagerank_coloring(
    matrix: csr_matrix,
    rsp: float = 0.15,
    epsilon: float = 1e-5,
    max_iterations: int = 1000,
) -> np.ndarray:
    n = matrix.shape[0]

    row_sums = np.asarray(matrix.sum(axis=1)).flatten()
    dangling_mask = (row_sums == 0).astype(np.float64)
    row_sums[row_sums == 0] = 1.0
    inv_sums = 1.0 / row_sums

    diag = csr_matrix((inv_sums, (np.arange(n), np.arange(n))), shape=(n, n))
    P = (diag @ matrix).tocsc()

    print("Running fast graph coloring...")
    partitions = fast_coloring_csr(matrix)
    print(f"Colored graph into {len(partitions)} partitions")

    # Convert partitions once to numpy arrays for numba
    partitions = [np.asarray(p, dtype=np.int32) for p in partitions]

    scores = np.full(n, 1.0 / n, dtype=np.float64)

    indptr = P.indptr
    indices = P.indices
    data = P.data
    start_time = time.time()
    for iteration in range(max_iterations):
        old_scores = scores.copy()
        dangling_sum = dangling_mask @ scores
        dangling_term = dangling_sum / n

        for part in partitions:
            scores[part] = update_partition_parallel(
                part, indptr, indices, data, scores, dangling_term, rsp, n
            )

        delta = np.linalg.norm(scores - old_scores, 1)
        if delta < epsilon:
            print(f"Converged in {iteration + 1} iterations")
            break
    else:
        import warnings
        warnings.warn(f"Did not converge after {max_iterations} iterations")
    end_time = time.time()
    print("Time for ONLY page rank", end_time - start_time)
    return scores

