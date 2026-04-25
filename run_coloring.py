import time
import argparse

import pandas as pd

from algo.coloring import pagerank_coloring
from algo.pagerank_utils import parse_to_csr, plot_all


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run graph-coloring PageRank.")
	parser.add_argument("--dataset", default="data/web-Google.txt")
	parser.add_argument("--epsilon", type=float, default=1e-5)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	matrix, nodes = parse_to_csr(args.dataset)

	start_time = time.time()
	scores = pagerank_coloring(matrix, epsilon=args.epsilon)
	end_time = time.time()
	print(f"Total time: {end_time - start_time:.4f}s")

	result = pd.Series(scores, index=nodes)

	plot_all(args.dataset, "Graph Coloring", result, matrix=matrix)

	print(result.nlargest(10))


if __name__ == "__main__":
	main()
