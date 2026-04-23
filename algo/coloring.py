import numpy as np
import os
from scipy.sparse import csr_matrix
from numba import njit, prange, set_num_threads
import time

set_num_threads(min(16, os.cpu_count() or 1))


@njit(parallel=True, cache=True)
def gs_sweep_color(start, end, color_nodes, indptr, indices, data, scores, dangling_term, rsp, n):
    """Parallel update for one color class. prange must be the outermost loop."""
    for k in prange(end - start):
        j = color_nodes[start + k]
        s = 0.0
        for p in range(indptr[j], indptr[j + 1]):
            s += data[p] * scores[indices[p]]
        scores[j] = (1.0 - rsp) * (s + dangling_term) + rsp / n


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
    """Returns (color_ptrs, color_nodes) flat CSR-style arrays."""
    sym = (matrix + matrix.T).tocsr()
    sym.data[:] = 1

    indptr = sym.indptr
    indices = sym.indices
    n = sym.shape[0]

    deg = np.diff(indptr)

    if n == 0 or deg.size == 0:
        return np.zeros(1, dtype=np.int32), np.empty(0, dtype=np.int32)
    if deg.max() == 0:
        return np.array([0, n], dtype=np.int32), np.arange(n, dtype=np.int32)

    order = np.argsort(-deg).astype(np.int64)
    max_deg = int(deg.max())

    colors = greedy_color_numba(indptr, indices, order, max_deg)

    num_colors = int(colors.max()) + 1

    # Build flat node list sorted by color, plus per-color pointer array.
    color_nodes = np.argsort(colors, kind='stable').astype(np.int32)
    color_ptrs = np.zeros(num_colors + 1, dtype=np.int32)
    for c in colors:
        color_ptrs[c + 1] += 1
    np.cumsum(color_ptrs, out=color_ptrs)

    return color_ptrs, color_nodes


def _warmup_jit():
    """Compile numba kernels on tiny dummy data so JIT doesn't count toward timing."""
    n = 4
    dummy_nodes = np.arange(4, dtype=np.int32)
    dummy_indptr = np.zeros(n + 1, dtype=np.int32)
    dummy_indices = np.empty(0, dtype=np.int32)
    dummy_data = np.empty(0, dtype=np.float64)
    dummy_scores = np.ones(n, dtype=np.float64) / n
    gs_sweep_color(0, 4, dummy_nodes, dummy_indptr, dummy_indices, dummy_data, dummy_scores, 0.0, 0.15, n)


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

    print("Running fast graph coloring...", flush=True)
    color_ptrs, color_nodes = fast_coloring_csr(matrix)
    num_colors = len(color_ptrs) - 1
    print(f"Colored graph into {num_colors} partitions", flush=True)

    indptr = P.indptr
    indices = P.indices
    data = P.data

    print("Warming up JIT...", flush=True)
    _warmup_jit()
    # Pre-extract int pairs so the loop doesn't re-convert numpy scalars each iteration
    color_ranges = [(int(color_ptrs[c]), int(color_ptrs[c + 1])) for c in range(num_colors)]

    scores = np.full(n, 1.0 / n, dtype=np.float64)

    start_time = time.time()
    for iteration in range(max_iterations):
        old_scores = scores.copy()
        dangling_sum = dangling_mask @ scores
        dangling_term = dangling_sum / n

        for start, end in color_ranges:
            gs_sweep_color(start, end, color_nodes, indptr, indices, data, scores, dangling_term, rsp, n)

        delta = np.linalg.norm(scores - old_scores, 1)
        if delta < epsilon:
            print(f"Converged in {iteration + 1} iterations", flush=True)
            break
    else:
        import warnings
        warnings.warn(f"Did not converge after {max_iterations} iterations")

    end_time = time.time()
    print("Time for ONLY page rank", end_time - start_time)
    return scores
