import pandas as pd
import time
from algo.basic import pagerank_csr
from algo.pagerank_utils import parse_to_csr, plot_all

dataset = "data/web-Google.txt"

matrix, nodes = parse_to_csr(dataset)
start_time = time.time()
scores = pagerank_csr(matrix)
end_time = time.time()
print("Time for power iteration", end_time - start_time)
result = pd.Series(scores, index=nodes)

plot_all(dataset, "Power Iteration", result, matrix=matrix)

print(result.nlargest(10))