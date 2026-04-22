import warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def parse_to_csr(filepath: str) -> tuple[csr_matrix, np.ndarray]:
    edges = pd.read_csv(
        filepath,
        sep='\t',
        comment='#',
        names=['from', 'to'],
        dtype={'from': 'int32', 'to': 'int32'},
    )

    sorted_nodes, inverse = np.unique(
        np.concatenate([edges['from'].values, edges['to'].values]),
        return_inverse=True,
    )
    n = len(sorted_nodes)
    rows = inverse[:len(edges)]
    cols = inverse[len(edges):]
    data = np.ones(len(edges), dtype=np.float32)

    matrix = csr_matrix((data, (rows, cols)), shape=(n, n))
    return matrix, sorted_nodes


def pagerank_csr(
    matrix: csr_matrix,
    rsp: float = 0.15,
    epsilon: float = 1e-5,
    max_iterations: int = 1000,
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

    for _ in range(max_iterations):
        dangling_sum = dangling_mask @ scores
        new_scores = (1 - rsp) * (P.T @ scores + dangling_sum / n) + rsp / n
        delta = np.linalg.norm(new_scores - scores, 1)
        scores = new_scores
        if delta < epsilon:
            break
    else:
        warnings.warn(f"PageRank did not converge after {max_iterations} iterations")

    return scores


matrix, nodes = parse_to_csr("graph.txt")
scores = pagerank_csr(matrix)

result = pd.Series(scores, index=nodes)
print(result.nlargest(10))