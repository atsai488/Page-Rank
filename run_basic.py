import pandas as pd
from algo.basic import pagerank_csr
from algo.pagerank_utils import parse_to_csr, plot_all

dataset = "data/web-BerkStan.txt"

matrix, nodes = parse_to_csr(dataset)
scores = pagerank_csr(matrix)

result = pd.Series(scores, index=nodes)

plot_all(dataset, "Power Iteration", result, matrix=matrix)

print(result.nlargest(10))