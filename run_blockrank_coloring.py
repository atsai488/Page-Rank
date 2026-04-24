import time

import numpy as np
import pandas as pd

from algo.blockrank_coloring import blockrank_coloring_csr
from algo.pagerank_utils import parse_to_csr, plot_all

dataset = "data/web-BerkStan.txt"

matrix, nodes = parse_to_csr(dataset)
n = matrix.shape[0]

num_blocks = 100
block_assignments = np.arange(n) // (n // num_blocks + 1)

print(f"Nodes: {n}, Edges: {matrix.nnz}, Blocks: {len(np.unique(block_assignments))}")

start_time = time.time()
scores = blockrank_coloring_csr(
    matrix,
    block_assignments,
    rsp=0.15,
    epsilon=1e-5,
    max_iterations=20000,
)
end_time = time.time()
print(f"Time for BlockRank+Coloring: {end_time - start_time:.4f}s")

result = pd.Series(scores, index=nodes).sort_index()

plot_all(dataset, "BlockRank+Coloring", result, matrix=matrix)

print(result.nlargest(10))
