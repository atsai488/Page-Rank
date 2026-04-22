import numpy as np
import random
from scipy.sparse import csr_matrix
from typing import List, Set
from pagerank_utils import plot_all, parse_to_csr
from numba import njit
import pandas as pd

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
    """
    PageRank using graph-coloring-ordered updates (Gauss-Seidel style).

    Standard PageRank uses Jacobi-style updates — all nodes update from the
    previous iteration's scores simultaneously. This uses Gauss-Seidel-style
    updates — within each color partition, nodes update using the most recent
    scores available, which accelerates convergence.

    Nodes within the same color partition are independent (no edges between
    them) so their updates are safe to parallelize.
    """
    n = matrix.shape[0]

    row_sums = np.asarray(matrix.sum(axis=1)).flatten()
    dangling_mask = (row_sums == 0).astype(np.float64)
    row_sums[row_sums == 0] = 1.0
    inv_sums = 1.0 / row_sums

    # Row-stochastic transition matrix
    diag = csr_matrix(
        (inv_sums, (np.arange(n), np.arange(n))), shape=(n, n)
    )
    P = (diag @ matrix).tocsc()  # CSC for fast column slicing during updates

    print("Running fast graph coloring...")
    partitions = fast_coloring_csr(matrix)
    num_colors = len(partitions)
    print(f"Colored graph into {num_colors} partitions")
    print(f"Partition sizes: min={min(len(p) for p in partitions)}, "
          f"max={max(len(p) for p in partitions)}, "
          f"mean={np.mean([len(p) for p in partitions]):.1f}")

    scores = np.full(n, 1.0 / n, dtype=np.float64)

    for iteration in range(max_iterations):
        old_scores = scores.copy()

        dangling_sum = dangling_mask @ scores

        # Process one color partition at a time.
        # Within each partition: nodes are independent — safe to vectorize.
        # Across partitions: use latest scores (Gauss-Seidel), not old_scores (Jacobi).
        for partition in partitions:
            idx = np.array(partition)
            # Pull the latest scores for the update — not old_scores
            scores[idx] = (1 - rsp) * (
                np.asarray(P[:, idx].T @ scores).flatten() + dangling_sum / n
            ) + rsp / n

        delta = np.linalg.norm(scores - old_scores, 1)
        if delta < epsilon:
            print(f"Converged in {iteration + 1} iterations")
            break
    else:
        import warnings
        warnings.warn(f"Did not converge after {max_iterations} iterations")

    return scores
dataset = "data/web-BerkStan.txt"
matrix, nodes = parse_to_csr(dataset)
scores = pagerank_coloring(matrix)
result = pd.Series(scores, index=nodes)
plot_all(dataset, "Graph Coloring", result, matrix=matrix)

print(result.nlargest(10))