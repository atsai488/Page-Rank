import time
import pandas as pd
from algo.coloring import pagerank_coloring
from algo.pagerank_utils import parse_to_csr, plot_all

dataset = "data/web-BerkStan.txt"

matrix, nodes = parse_to_csr(dataset)

start_time = time.time()
scores = pagerank_coloring(matrix)
end_time = time.time()
print(f"Total time: {end_time - start_time:.4f}s")

result = pd.Series(scores, index=nodes)

plot_all(dataset, "Graph Coloring", result, matrix=matrix)

print(result.nlargest(10))
