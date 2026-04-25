import time
import argparse

import pandas as pd

from algo.basic import pagerank_csr
from algo.pagerank_utils import parse_to_csr, plot_all


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run basic PageRank power iteration.")
	parser.add_argument("--dataset", default="data/web-Google.txt")
	parser.add_argument("--epsilon", type=float, default=1e-5)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	matrix, nodes = parse_to_csr(args.dataset)
	start_time = time.time()
	scores = pagerank_csr(matrix, epsilon=args.epsilon)
	end_time = time.time()
	print("Time for power iteration", end_time - start_time)
	result = pd.Series(scores, index=nodes)

	plot_all(args.dataset, "Power Iteration", result, matrix=matrix)

	print(result.nlargest(10))


if __name__ == "__main__":
	main()