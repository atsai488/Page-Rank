from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple
from pagerank_utils import normalize_distribution, pagerank_power_iteration


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