from __future__ import annotations

import time
import warnings
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from algo.pagerank_utils import normalize_distribution, pagerank_power_iteration


# ---------------------------------------------------------------------------
# CSR-based BlockRank (fast, numpy/scipy)
# ---------------------------------------------------------------------------

def _pagerank_csr_internal(
    P: csr_matrix,
    dangling_mask: np.ndarray,
    n: int,
    rsp: float = 0.15,
    epsilon: float = 1e-5,
    max_iterations: int = 20000,
    start: np.ndarray | None = None,
) -> np.ndarray:
    """Power iteration on a pre-normalized column-stochastic transpose matrix.

    This is a shared helper so that the same iteration logic is reused for
    local block solves, the block-level solve, and the final global solve.
    ``P`` must already be row-stochastic (rows sum to 1 for non-dangling nodes).
    """
    scores = np.full(n, 1.0 / n, dtype=np.float64) if start is None else start.copy()

    for _ in range(max_iterations):
        dangling_sum = dangling_mask @ scores
        new_scores = (1 - rsp) * (P.T @ scores + dangling_sum / n) + rsp / n
        delta = np.linalg.norm(new_scores - scores, 1)
        scores = new_scores
        if delta < epsilon:
            break
    else:
        warnings.warn(f"PageRank did not converge after {max_iterations} iterations")

    return scores


def _build_row_stochastic(matrix: csr_matrix) -> tuple[csr_matrix, np.ndarray]:
    """Return (P, dangling_mask) where P is row-stochastic."""
    n = matrix.shape[0]
    row_sums = np.asarray(matrix.sum(axis=1)).flatten()
    dangling_mask = (row_sums == 0).astype(np.float64)
    row_sums[row_sums == 0] = 1
    inv_sums = 1.0 / row_sums
    diag = csr_matrix((inv_sums, (np.arange(n), np.arange(n))), shape=(n, n))
    P = diag @ matrix
    return P, dangling_mask


def blockrank_csr(
    matrix: csr_matrix,
    block_assignments: np.ndarray,
    rsp: float = 0.15,
    epsilon: float = 1e-5,
    max_iterations: int = 20000,
) -> np.ndarray:
    """CSR-based BlockRank algorithm.

    Parameters
    ----------
    matrix : csr_matrix
        Adjacency matrix (n x n). Entry (i, j) means node i links to node j.
    block_assignments : np.ndarray
        Integer array of length n, mapping each node index to a block ID.
    rsp : float
        Random surfer probability (1 - damping factor).
    epsilon : float
        Convergence threshold for power iteration.
    max_iterations : int
        Maximum number of power iteration steps.

    Returns
    -------
    np.ndarray
        Final PageRank scores (length n), sums to 1.
    """
    n = matrix.shape[0]
    assert len(block_assignments) == n

    total_start = time.time()

    unique_blocks = np.unique(block_assignments)
    num_blocks = len(unique_blocks)

    # Build a mapping from block ID -> array of node indices in that block
    block_node_indices: list[np.ndarray] = []
    block_id_map: dict[int, int] = {}  # original block label -> contiguous index
    for bi, blk in enumerate(unique_blocks):
        block_id_map[int(blk)] = bi
        block_node_indices.append(np.where(block_assignments == blk)[0])

    # ------------------------------------------------------------------
    # Step 1: Local PageRank within each block
    # ------------------------------------------------------------------
    step1_start = time.time()
    local_scores = np.zeros(n, dtype=np.float64)

    for bi in range(num_blocks):
        idx = block_node_indices[bi]
        block_size = len(idx)
        if block_size == 0:
            continue

        # Extract the submatrix for intra-block edges
        sub = matrix[np.ix_(idx, idx)]
        P_local, dang_local = _build_row_stochastic(sub)
        local_pr = _pagerank_csr_internal(
            P_local, dang_local, block_size,
            rsp=rsp, epsilon=epsilon, max_iterations=max_iterations,
        )
        local_scores[idx] = local_pr

    step1_end = time.time()
    print(f"  Step 1 (local PR per block): {step1_end - step1_start:.4f}s")

    # ------------------------------------------------------------------
    # Step 2: Build block-level transition matrix and compute BlockRank
    # ------------------------------------------------------------------
    step2_start = time.time()

    # Map each node to its contiguous block index
    node_block_idx = np.array([block_id_map[int(b)] for b in block_assignments])

    # Count edges between blocks by iterating over non-zero entries
    coo = matrix.tocoo()
    src_blocks = node_block_idx[coo.row]
    dst_blocks = node_block_idx[coo.col]

    # Build the block-level adjacency matrix
    block_matrix = csr_matrix(
        (np.ones(len(coo.data), dtype=np.float64), (src_blocks, dst_blocks)),
        shape=(num_blocks, num_blocks),
    )
    # Duplicate entries are summed automatically by csr_matrix

    P_block, dang_block = _build_row_stochastic(block_matrix)
    block_ranks = _pagerank_csr_internal(
        P_block, dang_block, num_blocks,
        rsp=rsp, epsilon=epsilon, max_iterations=max_iterations,
    )

    step2_end = time.time()
    print(f"  Step 2 (block-level PR):     {step2_end - step2_start:.4f}s")

    # ------------------------------------------------------------------
    # Step 3: Weight local PR by block rank to form initial approximation
    # ------------------------------------------------------------------
    step3_start = time.time()

    approx = np.zeros(n, dtype=np.float64)
    for bi in range(num_blocks):
        idx = block_node_indices[bi]
        approx[idx] = block_ranks[bi] * local_scores[idx]

    # Normalize to a proper distribution
    total = approx.sum()
    if total > 0:
        approx /= total
    else:
        approx[:] = 1.0 / n

    step3_end = time.time()
    print(f"  Step 3 (build approximation): {step3_end - step3_start:.4f}s")

    # ------------------------------------------------------------------
    # Step 4: Full power iteration starting from the approximation
    # ------------------------------------------------------------------
    step4_start = time.time()

    P_full, dang_full = _build_row_stochastic(matrix)
    scores = _pagerank_csr_internal(
        P_full, dang_full, n,
        rsp=rsp, epsilon=epsilon, max_iterations=max_iterations,
        start=approx,
    )

    step4_end = time.time()
    print(f"  Step 4 (global refinement):  {step4_end - step4_start:.4f}s")

    total_end = time.time()
    print(f"  Total BlockRank time:        {total_end - total_start:.4f}s")

    return scores


# ---------------------------------------------------------------------------
# Original dict-based implementation (kept for backwards compatibility)
# ---------------------------------------------------------------------------


def _build_block_graph(
    graph: Mapping[str, Iterable[str]],
    page_to_block: Mapping[str, str],
) -> Tuple[Dict[str, Dict[str, float]], Set[str]]:
    """
    Collapse the page graph into a block graph.

    For each page edge u -> v, add one unit of weight from block(u) to block(v).
    Then normalize each block's outgoing weights into probabilities.
    """
    block_edges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    blocks: Set[str] = set(page_to_block.values())

    for u, nbrs in graph.items():
        if u not in page_to_block:
            continue
        bu = page_to_block[u]
        for v in nbrs:
            if v not in page_to_block:
                continue
            bv = page_to_block[v]
            block_edges[bu][bv] += 1.0

    # Normalize each row into transition probabilities
    block_graph: Dict[str, Dict[str, float]] = {}
    for b in blocks:
        row = dict(block_edges.get(b, {}))
        s = sum(row.values())
        if s > 0:
            block_graph[b] = {bb: w / s for bb, w in row.items()}
        else:
            block_graph[b] = {}  # dangling block
    return block_graph, blocks


def _local_pagerank_per_block(
    graph: Mapping[str, Iterable[str]],
    page_to_block: Mapping[str, str],
    alpha: float = 0.85,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> Dict[str, Dict[str, float]]:
    """
    Step 1 of BlockRank:
    Compute a local PageRank vector for each block independently,
    using only edges that stay inside the block.
    """
    block_to_pages: Dict[str, List[str]] = defaultdict(list)
    for page, block in page_to_block.items():
        block_to_pages[block].append(page)

    local_pr: Dict[str, Dict[str, float]] = {}

    for block, pages in block_to_pages.items():
        page_set = set(pages)

        # Restrict graph to intra-block links only
        subgraph: Dict[str, List[str]] = {}
        for p in pages:
            subgraph[p] = [q for q in graph.get(p, []) if q in page_set]

        local_pr[block] = pagerank_power_iteration(
            subgraph,
            pages,
            alpha=alpha,
            personalization=None,   # uniform over the block
            start=None,
            max_iter=max_iter,
            tol=tol,
        )

    return local_pr


def blockrank(
    graph: Mapping[str, Iterable[str]],
    page_to_block: Mapping[str, str],
    alpha: float = 0.85,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    BlockRank algorithm.

    Returns:
        global_pr: final PageRank vector after refinement
        block_rank: BlockRank of each block
        local_pr: local PageRank vector for each block

    Algorithm:
      1. Compute local PageRank inside each block.
      2. Collapse to block graph and compute BlockRank.
      3. Build a global approximation by weighting local PR by BlockRank.
      4. Run standard PageRank starting from that approximation.
    """
    all_pages = list(page_to_block.keys())
    all_blocks = sorted(set(page_to_block.values()))

    # Step 1
    local_pr = _local_pagerank_per_block(
        graph,
        page_to_block,
        alpha=alpha,
        tol=tol,
        max_iter=max_iter,
    )

    # Step 2
    block_graph, blocks = _build_block_graph(graph, page_to_block)
    block_rank = pagerank_power_iteration(
        block_graph,
        blocks,
        alpha=alpha,
        personalization=None,
        start=None,
        max_iter=max_iter,
        tol=tol,
    )

    # Step 3: approximate global PR by weighting local PR with BlockRank
    approx_global: Dict[str, float] = {}
    for page in all_pages:
        b = page_to_block[page]
        approx_global[page] = block_rank[b] * local_pr[b][page]
    approx_global = normalize_distribution(approx_global)

    # Step 4: refine with standard PageRank from the approximation
    global_pr = pagerank_power_iteration(
        graph,
        all_pages,
        alpha=alpha,
        personalization=None,
        start=approx_global,
        max_iter=max_iter,
        tol=tol,
    )

    return global_pr, block_rank, local_pr


def personalized_blockrank(
    graph: Mapping[str, Iterable[str]],
    page_to_block: Mapping[str, str],
    block_personalization: Mapping[str, float],
    alpha: float = 0.85,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Personalized BlockRank algorithm.

    block_personalization:
        personalization distribution over blocks/hosts.

    Returns:
        global_pr: personalized PageRank over pages
        personalized_block_rank: BlockRank over blocks
        local_pr: local PageRank vectors per block
    """
    all_pages = list(page_to_block.keys())
    block_to_pages: Dict[str, List[str]] = defaultdict(list)
    for page, block in page_to_block.items():
        block_to_pages[block].append(page)

    # Local PageRanks inside blocks (same as generic BlockRank)
    local_pr = _local_pagerank_per_block(
        graph,
        page_to_block,
        alpha=alpha,
        tol=tol,
        max_iter=max_iter,
    )

    # Step 1: personalized BlockRank on the block graph
    block_graph, blocks = _build_block_graph(graph, page_to_block)
    pb = {b: float(block_personalization.get(b, 0.0)) for b in blocks}
    pb = normalize_distribution(pb)

    personalized_block_rank = pagerank_power_iteration(
        block_graph,
        blocks,
        alpha=alpha,
        personalization=pb,
        start=None,
        max_iter=max_iter,
        tol=tol,
    )

    # Step 2: approximate global PR by weighting local PR with personalized BlockRank
    approx_global: Dict[str, float] = {}
    for page in all_pages:
        b = page_to_block[page]
        approx_global[page] = personalized_block_rank[b] * local_pr[b][page]
    approx_global = normalize_distribution(approx_global)

    # Step 3: induce personalization over pages from block personalization
    page_personalization: Dict[str, float] = {}
    for block, pages in block_to_pages.items():
        mass = pb.get(block, 0.0)
        if pages:
            per_page = mass / len(pages)
            for p in pages:
                page_personalization[p] = per_page

    page_personalization = normalize_distribution(page_personalization)

    # Step 4: final personalized PageRank refinement
    global_pr = pagerank_power_iteration(
        graph,
        all_pages,
        alpha=alpha,
        personalization=page_personalization,
        start=approx_global,
        max_iter=max_iter,
        tol=tol,
    )

    return global_pr, personalized_block_rank, local_pr