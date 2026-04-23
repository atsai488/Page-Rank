import numpy as np
import pandas as pd
from algo.blockrank import blockrank_csr
from algo.pagerank_utils import parse_to_csr, plot_all
import time

dataset = "data/web-BerkStan.txt"

# Parse the dataset into a CSR matrix
matrix, nodes = parse_to_csr(dataset)
n = matrix.shape[0]

# Assign blocks by bucketing node indices into ~100 groups
num_blocks = 100
block_assignments = np.arange(n) // (n // num_blocks + 1)

print(f"Nodes: {n}, Edges: {matrix.nnz}, Blocks: {len(np.unique(block_assignments))}")

start_time = time.time()
scores = blockrank_csr(
    matrix,
    block_assignments,
    rsp=0.15,
    epsilon=1e-5,
    max_iterations=20000,
)
end_time = time.time()
print(f"Time for BlockRank (CSR): {end_time - start_time:.4f}s")

# Convert to Series for plotting (use original node IDs)
result = pd.Series(scores, index=nodes).sort_index()

plot_all(dataset, "BlockRank", result, matrix=matrix)

print(result.nlargest(10))
