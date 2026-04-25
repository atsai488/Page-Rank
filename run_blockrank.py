import argparse
import time

import numpy as np
import pandas as pd

from algo.blockrank import blockrank_csr
from algo.pagerank_utils import parse_to_csr, plot_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BlockRank on an edge-list dataset.")
    parser.add_argument("--dataset", default="data/web-Google.txt")
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional CSV with columns node_id and block_id for semantic blocks.",
    )
    parser.add_argument("--num-blocks", type=int, default=100)
    parser.add_argument("--rsp", type=float, default=0.15)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--max-iterations", type=int, default=20000)
    parser.add_argument(
        "--disable-numba-parallel",
        action="store_true",
        help="Disable Numba parallel kernel in BlockRank internals.",
    )
    parser.add_argument(
        "--numba-global-min-n",
        type=int,
        default=200000,
        help="Minimum size to use Numba parallel kernel for block/global solves.",
    )
    parser.add_argument(
        "--numba-local-min-n",
        type=int,
        default=200000,
        help="Minimum block size to use Numba parallel kernel for local solves.",
    )
    return parser.parse_args()


def _build_bucket_blocks(n: int, num_blocks: int) -> np.ndarray:
    if num_blocks < 1:
        raise ValueError("--num-blocks must be >= 1")
    return np.arange(n) // (n // num_blocks + 1)


def _build_blocks_from_metadata(nodes: np.ndarray, metadata_path: str) -> np.ndarray:
    meta = pd.read_csv(metadata_path)
    required_cols = {"node_id", "block_id"}
    missing = required_cols.difference(meta.columns)
    if missing:
        raise ValueError(
            f"Metadata file is missing required columns: {sorted(missing)}"
        )

    block_by_node = pd.Series(meta["block_id"].values, index=meta["node_id"].values)
    aligned = block_by_node.reindex(nodes)
    if aligned.isna().any():
        missing_nodes = nodes[aligned.isna().to_numpy()]
        preview = missing_nodes[:10].tolist()
        raise ValueError(
            "Metadata does not contain block_id for all parsed nodes. "
            f"Missing {len(missing_nodes)} node(s), first few: {preview}"
        )

    return aligned.to_numpy(dtype=np.int64)


def main() -> None:
    args = parse_args()

    matrix, nodes = parse_to_csr(args.dataset)
    n = matrix.shape[0]

    if args.metadata:
        block_assignments = _build_blocks_from_metadata(nodes, args.metadata)
        block_source = f"metadata ({args.metadata})"
    else:
        block_assignments = _build_bucket_blocks(n, args.num_blocks)
        block_source = f"index buckets ({args.num_blocks} target blocks)"

    print(
        f"Nodes: {n}, Edges: {matrix.nnz}, Blocks: {len(np.unique(block_assignments))}, "
        f"Source: {block_source}"
    )

    start_time = time.time()
    scores = blockrank_csr(
        matrix,
        block_assignments,
        rsp=args.rsp,
        epsilon=args.epsilon,
        max_iterations=args.max_iterations,
        numba_parallel=not args.disable_numba_parallel,
        numba_global_min_n=args.numba_global_min_n,
        numba_local_min_n=args.numba_local_min_n,
    )
    end_time = time.time()
    print(f"Time for BlockRank (CSR): {end_time - start_time:.4f}s")

    result = pd.Series(scores, index=nodes).sort_index()
    plot_all(args.dataset, "BlockRank", result, matrix=matrix)
    print(result.nlargest(10))


if __name__ == "__main__":
    main()
