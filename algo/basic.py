import warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pagerank_utils import plot_all, parse_to_csr
import time


def pagerank_csr(
    matrix: csr_matrix,
    rsp: float = 0.15,
    epsilon: float = 1e-5,
    max_iterations: int = 20000,
) -> np.ndarray:
    n = matrix.shape[0]

    row_sums = np.asarray(matrix.sum(axis=1)).flatten()
    dangling_mask = (row_sums == 0).astype(np.float64)
    row_sums[row_sums == 0] = 1
    inv_sums = 1.0 / row_sums

    diag = csr_matrix(
        (inv_sums, (np.arange(n), np.arange(n))), shape=(n, n)
    )
    P = diag @ matrix

    scores = np.full(n, 1.0 / n, dtype=np.float64)
    start_time = time.time()
    for _ in range(max_iterations):
        dangling_sum = dangling_mask @ scores
        new_scores = (1 - rsp) * (P.T @ scores + dangling_sum / n) + rsp / n
        delta = np.linalg.norm(new_scores - scores, 1)
        scores = new_scores
        if delta < epsilon:
            break
    else:
        warnings.warn(f"PageRank did not converge after {max_iterations} iterations")
    end_time = time.time()
    print("Time for ONLY page rank", end_time - start_time)
    return scores

dataset = "data/web-BerkStan.txt"
matrix, nodes = parse_to_csr(dataset)
scores = pagerank_csr(matrix)
result = pd.Series(scores, index=nodes)
plot_all(dataset, "Power Iteration", result, matrix=matrix)

print(result.nlargest(10))