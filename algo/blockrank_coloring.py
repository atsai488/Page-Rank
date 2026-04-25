import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numba import njit, prange, set_num_threads
from scipy.sparse import csr_matrix

_n_threads = min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
set_num_threads(_n_threads)


@njit(parallel=True, cache=True)
def _update_all_parallel(indptr, indices, data, scores, dangling_term, rsp, n):
    """Jacobi-style bulk update: all nodes read from `scores`, write to `out`.
    No data races — each j writes only to out[j]."""
    out = np.empty(n, dtype=np.float64)
    for j in prange(n):
        s = 0.0
        for p in range(indptr[j], indptr[j + 1]):
            i = indices[p]
            s += data[p] * scores[i]
        out[j] = (1.0 - rsp) * (s + dangling_term) + rsp / n
    return out


def _build_row_stochastic(matrix: csr_matrix) -> tuple[csr_matrix, np.ndarray]:
    """Return (P, dangling_mask) where P is row-stochastic."""
    n = matrix.shape[0]
    row_sums = np.asarray(matrix.sum(axis=1)).flatten()
    dangling_mask = (row_sums == 0).astype(np.float64)
    row_sums[row_sums == 0] = 1.0
    inv_sums = 1.0 / row_sums
    diag = csr_matrix((inv_sums, (np.arange(n), np.arange(n))), shape=(n, n))
    P = diag @ matrix
    return P, dangling_mask


def _pagerank_serial_internal(
    P: csr_matrix,
    dangling_mask: np.ndarray,
    n: int,
    rsp: float = 0.15,
    epsilon: float = 1e-5,
    max_iterations: int = 20000,
    start: np.ndarray | None = None,
) -> np.ndarray:
    """Serial power iteration for small problems where Numba overhead dominates."""
    scores = np.full(n, 1.0 / n, dtype=np.float64) if start is None else start.copy()
    PT = P.T.tocsr()

    for _ in range(max_iterations):
        dangling_sum = dangling_mask @ scores
        new_scores = (1 - rsp) * (PT @ scores + dangling_sum / n) + rsp / n
        delta = np.linalg.norm(new_scores - scores, 1)
        scores = new_scores
        if delta < epsilon:
            break
    else:
        warnings.warn(f"PageRank did not converge after {max_iterations} iterations")

    return scores


def _pagerank_parallel_internal(
    P: csr_matrix,
    dangling_mask: np.ndarray,
    n: int,
    rsp: float = 0.15,
    epsilon: float = 1e-5,
    max_iterations: int = 20000,
    start: np.ndarray | None = None,
) -> np.ndarray:
    """Power iteration using the Numba parallel Jacobi kernel. Use for large n."""
    P_csc = P.tocsc()
    indptr = P_csc.indptr
    indices = P_csc.indices
    data = P_csc.data

    scores = np.full(n, 1.0 / n, dtype=np.float64) if start is None else start.copy()

    for _ in range(max_iterations):
        dangling_term = (dangling_mask @ scores) / n
        new_scores = _update_all_parallel(
            indptr, indices, data, scores, dangling_term, rsp, n
        )
        delta = np.linalg.norm(new_scores - scores, 1)
        scores = new_scores
        if delta < epsilon:
            break
    else:
        warnings.warn(f"PageRank did not converge after {max_iterations} iterations")

    return scores


def _group_nodes_by_block(
    block_assignments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Build contiguous block IDs and grouped node indices in one pass."""
    unique_blocks, node_block_idx = np.unique(block_assignments, return_inverse=True)

    order = np.argsort(node_block_idx, kind="mergesort")
    counts = np.bincount(node_block_idx, minlength=len(unique_blocks))
    split_points = np.cumsum(counts)[:-1]
    if len(split_points) == 0:
        block_node_indices = [order]
    else:
        block_node_indices = np.split(order, split_points)

    return unique_blocks, node_block_idx.astype(np.int32, copy=False), block_node_indices


def _chunk_blocks(
    block_node_indices: list[np.ndarray],
    chunk_size: int,
) -> list[list[np.ndarray]]:
    """Split block index arrays into coarse chunks to reduce scheduler overhead."""
    return [
        block_node_indices[i : i + chunk_size]
        for i in range(0, len(block_node_indices), chunk_size)
    ]


def _solve_local_block_chunk(
    matrix: csr_matrix,
    block_chunk: list[np.ndarray],
    rsp: float,
    epsilon: float,
    max_iterations: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Solve local PR for a chunk of blocks and return (idx, scores) pairs."""
    chunk_results: list[tuple[np.ndarray, np.ndarray]] = []

    for idx in block_chunk:
        block_size = len(idx)
        if block_size == 0:
            continue

        sub = matrix[np.ix_(idx, idx)]
        P_local, dang_local = _build_row_stochastic(sub)
        local_pr = _pagerank_serial_internal(
            P_local,
            dang_local,
            block_size,
            rsp=rsp,
            epsilon=epsilon,
            max_iterations=max_iterations,
        )
        chunk_results.append((idx, local_pr))

    return chunk_results


def blockrank_coloring_csr(
    matrix: csr_matrix,
    block_assignments: np.ndarray,
    rsp: float = 0.15,
    epsilon: float = 1e-5,
    max_iterations: int = 20000,
    local_parallel: bool = True,
    local_workers: int | None = None,
    local_chunk_size: int = 64,
) -> np.ndarray:
    """BlockRank with a parallel (coloring-style) Numba kernel.

    Step 1: local PageRank inside each block (serial — subgraphs too small for Numba).
    Step 2: PageRank on the collapsed block graph (serial — tiny matrix).
    Step 3: weight local PR by block rank to form an initial approximation.
    Step 4: global power iteration from that approximation (parallel kernel).
    """
    n = matrix.shape[0]
    assert len(block_assignments) == n

    # JIT warmup (kept OUTSIDE all step timers so measurements reflect real work)
    _warm_indptr = np.array([0, 1], dtype=np.int32)
    _warm_indices = np.array([0], dtype=np.int32)
    _warm_data = np.array([1.0], dtype=np.float64)
    _warm_scores = np.array([1.0], dtype=np.float64)
    _ = _update_all_parallel(_warm_indptr, _warm_indices, _warm_data, _warm_scores, 0.0, 0.15, 1)

    total_start = time.time()

    prep_start = time.time()
    unique_blocks, node_block_idx, block_node_indices = _group_nodes_by_block(block_assignments)
    num_blocks = len(unique_blocks)
    prep_end = time.time()
    print(f"  Preprocess (group blocks):   {prep_end - prep_start:.4f}s")

    # ------------------------------------------------------------------
    # Step 1: local PageRank within each block (serial, matches blockrank.py)
    # ------------------------------------------------------------------
    step1_start = time.time()
    local_scores = np.zeros(n, dtype=np.float64)

    if local_parallel and num_blocks > 1:
        workers = local_workers if local_workers is not None else _n_threads
        workers = max(1, workers)
        chunk_size = max(1, local_chunk_size)
        block_chunks = _chunk_blocks(block_node_indices, chunk_size)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _solve_local_block_chunk,
                    matrix,
                    chunk,
                    rsp,
                    epsilon,
                    max_iterations,
                )
                for chunk in block_chunks
            ]

            for future in as_completed(futures):
                for idx, local_pr in future.result():
                    local_scores[idx] = local_pr
    else:
        for bi in range(num_blocks):
            idx = block_node_indices[bi]
            block_size = len(idx)
            if block_size == 0:
                continue

            sub = matrix[np.ix_(idx, idx)]
            P_local, dang_local = _build_row_stochastic(sub)
            local_pr = _pagerank_serial_internal(
                P_local, dang_local, block_size,
                rsp=rsp, epsilon=epsilon, max_iterations=max_iterations,
            )
            local_scores[idx] = local_pr

    step1_end = time.time()
    print(f"  Step 1 (local PR per block): {step1_end - step1_start:.4f}s")

    # ------------------------------------------------------------------
    # Step 2: block-level PageRank
    # ------------------------------------------------------------------
    step2_start = time.time()

    coo = matrix.tocoo()
    src_blocks = node_block_idx[coo.row]
    dst_blocks = node_block_idx[coo.col]

    block_matrix = csr_matrix(
        (np.ones(len(coo.data), dtype=np.float64), (src_blocks, dst_blocks)),
        shape=(num_blocks, num_blocks),
    )

    P_block, dang_block = _build_row_stochastic(block_matrix)
    block_ranks = _pagerank_serial_internal(
        P_block, dang_block, num_blocks,
        rsp=rsp, epsilon=epsilon, max_iterations=max_iterations,
    )

    step2_end = time.time()
    print(f"  Step 2 (block-level PR):     {step2_end - step2_start:.4f}s")

    # ------------------------------------------------------------------
    # Step 3: weight local PR by block rank
    # ------------------------------------------------------------------
    step3_start = time.time()

    approx = np.zeros(n, dtype=np.float64)
    for bi in range(num_blocks):
        idx = block_node_indices[bi]
        approx[idx] = block_ranks[bi] * local_scores[idx]

    total = approx.sum()
    if total > 0:
        approx /= total
    else:
        approx[:] = 1.0 / n

    step3_end = time.time()
    print(f"  Step 3 (build approximation): {step3_end - step3_start:.4f}s")

    # ------------------------------------------------------------------
    # Step 4: global refinement (the dominant cost — parallel kernel wins here)
    # ------------------------------------------------------------------
    step4_start = time.time()
    epsilon = 1e-3
    P_full, dang_full = _build_row_stochastic(matrix)
    scores = _pagerank_parallel_internal(
        P_full, dang_full, n,
        rsp=rsp, epsilon=epsilon, max_iterations=max_iterations,
        start=approx,
    )

    step4_end = time.time()
    print(f"  Step 4 (global refinement):  {step4_end - step4_start:.4f}s")

    total_end = time.time()
    print(f"  Total BlockRank+Coloring time: {total_end - total_start:.4f}s")

    return scores
