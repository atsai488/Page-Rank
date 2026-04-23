import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from algo.blockrank import blockrank, blockrank_csr


# ---------------------------------------------------------------------------
# Helpers for the old dict-based tests
# ---------------------------------------------------------------------------

def build_graph(edges):
    """Build a dict-based adjacency list from edge tuples."""
    graph = {}
    all_nodes = set()
    for u, v in edges:
        u, v = str(u), str(v)
        graph.setdefault(u, []).append(v)
        all_nodes.add(u)
        all_nodes.add(v)
    for node in all_nodes:
        graph.setdefault(node, [])
    return graph, sorted(all_nodes, key=lambda x: int(x))


def single_block(nodes):
    """All nodes in one block."""
    return {n: "block_0" for n in nodes}


def per_node_blocks(nodes):
    """Each node in its own block."""
    return {n: f"block_{n}" for n in nodes}


def assert_valid_pr(pr, nodes, atol=1e-6):
    values = np.array([pr[n] for n in nodes])
    assert np.all(values >= -atol), "PageRank should not produce negative values"
    assert_allclose(values.sum(), 1.0, atol=atol)


# ---------------------------------------------------------------------------
# Helpers for CSR-based tests
# ---------------------------------------------------------------------------

def build_csr(n, edges):
    """Build an adjacency CSR matrix from (source, target) edge tuples."""
    if not edges:
        return csr_matrix((n, n), dtype=np.float64)
    rows = np.array([u for u, v in edges], dtype=np.int32)
    cols = np.array([v for u, v in edges], dtype=np.int32)
    data = np.ones(len(edges), dtype=np.float64)
    return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)


def per_node_block_assignments(n):
    """Each node in its own block (block ID == node index)."""
    return np.arange(n, dtype=np.int32)


def single_block_assignments(n):
    """All nodes in one block (block ID 0)."""
    return np.zeros(n, dtype=np.int32)


def assert_valid_pagerank(scores, n, atol=1e-6):
    assert scores.shape == (n,)
    assert np.all(scores >= -atol), "PageRank should not produce negative values"
    assert_allclose(scores.sum(), 1.0, atol=atol)


# ---------------------------------------------------------------------------
# Original dict-based tests (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def test_two_node_bidirectional_is_uniform():
    graph, nodes = build_graph([(0, 1), (1, 0)])
    page_to_block = per_node_blocks(nodes)
    global_pr, _, _ = blockrank(graph, page_to_block, tol=1e-12)

    assert_valid_pr(global_pr, nodes)
    assert_allclose(global_pr["0"], 0.5, atol=1e-4)
    assert_allclose(global_pr["1"], 0.5, atol=1e-4)


def test_three_node_cycle_is_uniform():
    graph, nodes = build_graph([(0, 1), (1, 2), (2, 0)])
    page_to_block = per_node_blocks(nodes)
    global_pr, _, _ = blockrank(graph, page_to_block, tol=1e-12)

    assert_valid_pr(global_pr, nodes)
    for n in nodes:
        assert_allclose(global_pr[n], 1 / 3, atol=1e-4)


def test_complete_graph_is_uniform():
    edges = [(i, j) for i in range(4) for j in range(4) if i != j]
    graph, nodes = build_graph(edges)
    page_to_block = per_node_blocks(nodes)
    global_pr, _, _ = blockrank(graph, page_to_block, tol=1e-12)

    assert_valid_pr(global_pr, nodes)
    for n in nodes:
        assert_allclose(global_pr[n], 0.25, atol=1e-4)


def test_all_dangling_nodes_stays_uniform():
    graph = {str(i): [] for i in range(5)}
    nodes = [str(i) for i in range(5)]
    page_to_block = single_block(nodes)
    global_pr, _, _ = blockrank(graph, page_to_block, tol=1e-12)

    assert_valid_pr(global_pr, nodes)
    for n in nodes:
        assert_allclose(global_pr[n], 0.2, atol=1e-4)


def test_dangling_node_ordering():
    # 0 -> 1, node 1 is dangling and should have higher rank
    graph, nodes = build_graph([(0, 1)])
    page_to_block = per_node_blocks(nodes)
    global_pr, _, _ = blockrank(graph, page_to_block, tol=1e-12)

    assert_valid_pr(global_pr, nodes)
    assert global_pr["1"] > global_pr["0"]


def test_chain_ordering():
    # 0 -> 1 -> 2, node 2 should rank highest
    graph, nodes = build_graph([(0, 1), (1, 2)])
    page_to_block = per_node_blocks(nodes)
    global_pr, _, _ = blockrank(graph, page_to_block, tol=1e-12)

    assert_valid_pr(global_pr, nodes)
    assert global_pr["2"] > global_pr["1"] > global_pr["0"]


def test_single_block_matches_ordering():
    # All nodes in one block — should still produce valid results
    graph, nodes = build_graph([(0, 1), (1, 2), (2, 0)])
    page_to_block = single_block(nodes)
    global_pr, _, _ = blockrank(graph, page_to_block, tol=1e-12)

    assert_valid_pr(global_pr, nodes)
    for n in nodes:
        assert_allclose(global_pr[n], 1 / 3, atol=1e-4)


def test_no_nan_or_inf():
    graph, nodes = build_graph([(0, 1), (1, 2)])
    page_to_block = per_node_blocks(nodes)
    global_pr, _, _ = blockrank(graph, page_to_block)

    values = np.array([global_pr[n] for n in nodes])
    assert np.isfinite(values).all()


# ---------------------------------------------------------------------------
# CSR-based tests for blockrank_csr
# ---------------------------------------------------------------------------

class TestBlockrankCSR:
    """Tests for the CSR-based blockrank_csr function."""

    def test_two_node_bidirectional_is_uniform(self):
        matrix = build_csr(2, [(0, 1), (1, 0)])
        blocks = per_node_block_assignments(2)
        scores = blockrank_csr(matrix, blocks, rsp=0.15, epsilon=1e-12)

        assert_valid_pagerank(scores, 2)
        assert_allclose(scores, [0.5, 0.5], atol=1e-4)

    def test_three_node_cycle_is_uniform(self):
        matrix = build_csr(3, [(0, 1), (1, 2), (2, 0)])
        blocks = per_node_block_assignments(3)
        scores = blockrank_csr(matrix, blocks, rsp=0.15, epsilon=1e-12)

        assert_valid_pagerank(scores, 3)
        assert_allclose(scores, [1/3, 1/3, 1/3], atol=1e-4)

    def test_complete_graph_is_uniform(self):
        edges = [(i, j) for i in range(4) for j in range(4) if i != j]
        matrix = build_csr(4, edges)
        blocks = per_node_block_assignments(4)
        scores = blockrank_csr(matrix, blocks, rsp=0.15, epsilon=1e-12)

        assert_valid_pagerank(scores, 4)
        assert_allclose(scores, np.full(4, 0.25), atol=1e-4)

    def test_all_dangling_nodes_stays_uniform(self):
        matrix = build_csr(5, [])
        blocks = single_block_assignments(5)
        scores = blockrank_csr(matrix, blocks, rsp=0.15, epsilon=1e-12)

        assert_valid_pagerank(scores, 5)
        assert_allclose(scores, np.full(5, 0.2), atol=1e-4)

    def test_dangling_node_ordering(self):
        # 0 -> 1, node 1 is dangling and should have higher rank
        matrix = build_csr(2, [(0, 1)])
        blocks = per_node_block_assignments(2)
        scores = blockrank_csr(matrix, blocks, rsp=0.15, epsilon=1e-12)

        assert_valid_pagerank(scores, 2)
        assert scores[1] > scores[0]

    def test_chain_ordering(self):
        # 0 -> 1 -> 2, node 2 should rank highest
        matrix = build_csr(3, [(0, 1), (1, 2)])
        blocks = per_node_block_assignments(3)
        scores = blockrank_csr(matrix, blocks, rsp=0.15, epsilon=1e-12)

        assert_valid_pagerank(scores, 3)
        assert scores[2] > scores[1] > scores[0]

    def test_single_block_matches_ordering(self):
        # All nodes in one block — should still produce valid results
        matrix = build_csr(3, [(0, 1), (1, 2), (2, 0)])
        blocks = single_block_assignments(3)
        scores = blockrank_csr(matrix, blocks, rsp=0.15, epsilon=1e-12)

        assert_valid_pagerank(scores, 3)
        assert_allclose(scores, [1/3, 1/3, 1/3], atol=1e-4)

    def test_no_nan_or_inf(self):
        matrix = build_csr(3, [(0, 1), (1, 2)])
        blocks = per_node_block_assignments(3)
        scores = blockrank_csr(matrix, blocks)

        assert np.isfinite(scores).all()

    def test_two_blocks_with_cross_edges(self):
        # Nodes 0,1 in block 0; nodes 2,3 in block 1
        # Cross-block edges: 1->2, 3->0
        edges = [(0, 1), (1, 0), (1, 2), (2, 3), (3, 2), (3, 0)]
        matrix = build_csr(4, edges)
        blocks = np.array([0, 0, 1, 1], dtype=np.int32)
        scores = blockrank_csr(matrix, blocks, rsp=0.15, epsilon=1e-12)

        assert_valid_pagerank(scores, 4)

    def test_self_loop_single_node(self):
        matrix = build_csr(1, [(0, 0)])
        blocks = np.array([0], dtype=np.int32)
        scores = blockrank_csr(matrix, blocks, rsp=0.15, epsilon=1e-12)

        assert_valid_pagerank(scores, 1)
        assert_allclose(scores, [1.0], atol=1e-10)

    def test_returns_numpy_array(self):
        matrix = build_csr(3, [(0, 1), (1, 2), (2, 0)])
        blocks = per_node_block_assignments(3)
        scores = blockrank_csr(matrix, blocks)

        assert isinstance(scores, np.ndarray)
        assert scores.dtype == np.float64
