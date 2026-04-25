import numpy as np
import os
from scipy.sparse import csr_matrix
from numba import njit, prange, set_num_threads
import time

_n_threads = min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
set_num_threads(_n_threads)


@njit(parallel=True, cache=True)
def update_all_parallel(indptr, indices, data, scores, dangling_term, rsp, n):
    """Jacobi-style bulk update: all nodes read from `scores`, write to `out`.
    No data races — each j writes only to out[j]."""
    out = np.empty(n, dtype=np.float64)
    for j in prange(n):
        s = 0.0
        for p in range(indptr[j], indptr[j + 1]):
            i = indices[p]
            s += data[p] * scores[i]
        out[j] = (1.0 - rsp) * (s + dangling_term) + rsp / n
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
    order = np.argsort(-deg)
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

    indptr = P.indptr
    indices = P.indices
    data = P.data

    scores = np.full(n, 1.0 / n, dtype=np.float64)

    # Warm up Numba JIT before timing
    _ = update_all_parallel(indptr, indices, data, scores, 0.0, rsp, n)

    for iteration in range(max_iterations):
        dangling_term = (dangling_mask @ scores) / n
        new_scores = update_all_parallel(indptr, indices, data, scores, dangling_term, rsp, n)
        delta = np.linalg.norm(new_scores - scores, 1)
        scores = new_scores
        if delta < epsilon:
            print(f"Converged in {iteration + 1} iterations")
            break
    else:
        import warnings
        warnings.warn(f"Did not converge after {max_iterations} iterations")

    return scores


