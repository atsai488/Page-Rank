import time
import argparse

import numpy as np
import pandas as pd

from algo.blockrank_coloring import blockrank_coloring_csr
from algo.pagerank_utils import parse_to_csr, plot_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BlockRank+Coloring PageRank.")
    parser.add_argument("--dataset", default="data/web-Google.txt")
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional CSV with columns node_id and block_id for semantic blocks.",
    )
    parser.add_argument("--num-blocks", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument(
        "--disable-local-parallel",
        action="store_true",
        help="Disable Step 1 local block parallelism.",
    )
    parser.add_argument(
        "--local-workers",
        type=int,
        default=None,
        help="Worker thread count for Step 1 local block solves (default: auto).",
    )
    parser.add_argument(
        "--local-chunk-size",
        type=int,
        default=64,
        help="Number of blocks per Step 1 local task.",
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

    # start_time = time.time()
    print("Block rank with args")
    scores = blockrank_coloring_csr(
        matrix,
        block_assignments,
        rsp=0.15,
        epsilon=args.epsilon,
        max_iterations=20000,
        local_parallel=not args.disable_local_parallel,
        local_workers=args.local_workers,
        local_chunk_size=args.local_chunk_size,
    )
    # end_time = time.time()
    # print(f"Time for BlockRank+Coloring: {end_time - start_time:.4f}s")

    result = pd.Series(scores, index=nodes).sort_index()

    plot_all(args.dataset, "BlockRank+Coloring", result, matrix=matrix)

    print(result.nlargest(10))


if __name__ == "__main__":
    main()
