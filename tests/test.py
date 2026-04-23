# test_pagerank.py
import numpy as np
import pytest
import sys
import os
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from algo.basic import pagerank_csr  # change this import


def build_csr(n, edges):
    """
    Build a row-stochastic adjacency-style CSR input.
    Each edge is (source, target), matching your implementation's row-normalization.
    """
    if not edges:
        return csr_matrix((n, n), dtype=np.float64)

    rows = np.array([u for u, v in edges], dtype=np.int32)
    cols = np.array([v for u, v in edges], dtype=np.int32)
    data = np.ones(len(edges), dtype=np.float64)
    return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)


def assert_valid_pagerank(scores, n, atol=1e-10):
    assert scores.shape == (n,)
    assert np.all(scores >= -atol), "PageRank should not produce negative values"
    assert_allclose(scores.sum(), 1.0, atol=atol)


def test_two_node_bidirectional_is_uniform():
    # 0 <-> 1
    matrix = build_csr(2, [(0, 1), (1, 0)])
    scores = pagerank_csr(matrix, rsp=0.15, epsilon=1e-12)

    assert_valid_pagerank(scores, 2)
    assert_allclose(scores, [0.5, 0.5], atol=1e-8)


def test_three_node_cycle_is_uniform():
    # 0 -> 1 -> 2 -> 0
    matrix = build_csr(3, [(0, 1), (1, 2), (2, 0)])
    scores = pagerank_csr(matrix, rsp=0.15, epsilon=1e-12)

    assert_valid_pagerank(scores, 3)
    assert_allclose(scores, [1 / 3, 1 / 3, 1 / 3], atol=1e-8)


def test_complete_graph_is_uniform():
    # Every node links to every other node, no self-loops
    edges = [(i, j) for i in range(4) for j in range(4) if i != j]
    matrix = build_csr(4, edges)
    scores = pagerank_csr(matrix, rsp=0.15, epsilon=1e-12)

    assert_valid_pagerank(scores, 4)
    assert_allclose(scores, np.full(4, 0.25), atol=1e-8)


def test_all_dangling_nodes_stays_uniform():
    # No edges at all
    matrix = build_csr(5, [])
    scores = pagerank_csr(matrix, rsp=0.15, epsilon=1e-12)

    assert_valid_pagerank(scores, 5)
    assert_allclose(scores, np.full(5, 0.2), atol=1e-8)


def test_dangling_node_example_matches_expected_values():
    """
    Graph:
        0 -> 1
        1 is dangling

    With rsp=0.15, the fixed point is:
        rank[0] = 0.3508771929824561
        rank[1] = 0.6491228070175439
    """
    matrix = build_csr(2, [(0, 1)])
    scores = pagerank_csr(matrix, rsp=0.15, epsilon=1e-12)

    assert_valid_pagerank(scores, 2)
    expected = np.array([0.3508771929824561, 0.6491228070175439])
    assert_allclose(scores, expected, atol=1e-8)


def test_self_loop_single_node_is_one():
    matrix = build_csr(1, [(0, 0)])
    scores = pagerank_csr(matrix, rsp=0.15, epsilon=1e-12)

    assert_valid_pagerank(scores, 1)
    assert_allclose(scores, [1.0], atol=1e-12)


def test_ordering_in_simple_chain():
    # 0 -> 1 -> 2, with 2 dangling
    matrix = build_csr(3, [(0, 1), (1, 2)])
    scores = pagerank_csr(matrix, rsp=0.15, epsilon=1e-12)

    assert_valid_pagerank(scores, 3)
    assert scores[2] > scores[1] > scores[0]