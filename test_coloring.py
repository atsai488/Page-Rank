import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from algo.coloring import pagerank_coloring


def build_csr(n, edges):
    if not edges:
        return csr_matrix((n, n), dtype=np.float64)
    rows = np.array([u for u, v in edges], dtype=np.int32)
    cols = np.array([v for u, v in edges], dtype=np.int32)
    data = np.ones(len(edges), dtype=np.float64)
    return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)


def assert_valid_pagerank(scores, n, atol=1e-6):
    assert scores.shape == (n,)
    assert np.all(scores >= -atol), "PageRank should not produce negative values"
    assert_allclose(scores.sum(), 1.0, atol=atol)


def test_two_node_bidirectional_is_uniform():
    matrix = build_csr(2, [(0, 1), (1, 0)])
    scores = pagerank_coloring(matrix, rsp=0.15, epsilon=1e-12)

    assert_valid_pagerank(scores, 2)
    assert_allclose(scores, [0.5, 0.5], atol=1e-8)


def test_three_node_cycle_is_uniform():
    matrix = build_csr(3, [(0, 1), (1, 2), (2, 0)])
    scores = pagerank_coloring(matrix, rsp=0.15, epsilon=1e-12)

    assert_valid_pagerank(scores, 3)
    assert_allclose(scores, [1 / 3, 1 / 3, 1 / 3], atol=1e-8)


def test_complete_graph_is_uniform():
    edges = [(i, j) for i in range(4) for j in range(4) if i != j]
    matrix = build_csr(4, edges)
    scores = pagerank_coloring(matrix, rsp=0.15, epsilon=1e-12)

    assert_valid_pagerank(scores, 4)
    assert_allclose(scores, np.full(4, 0.25), atol=1e-8)


def test_all_dangling_nodes_stays_uniform():
    matrix = build_csr(5, [])
    scores = pagerank_coloring(matrix, rsp=0.15, epsilon=1e-12)

    assert_valid_pagerank(scores, 5)
    assert_allclose(scores, np.full(5, 0.2), atol=1e-8)


def test_dangling_node_ordering():
    # 0 -> 1, node 1 is dangling and should have higher rank
    matrix = build_csr(2, [(0, 1)])
    scores = pagerank_coloring(matrix, rsp=0.15, epsilon=1e-12)

    assert_valid_pagerank(scores, 2)
    assert scores[1] > scores[0]


def test_chain_ordering():
    # 0 -> 1 -> 2, node 2 should rank highest
    matrix = build_csr(3, [(0, 1), (1, 2)])
    scores = pagerank_coloring(matrix, rsp=0.15, epsilon=1e-12)

    assert_valid_pagerank(scores, 3)
    assert scores[2] > scores[1] > scores[0]


def test_self_loop_single_node():
    matrix = build_csr(1, [(0, 0)])
    scores = pagerank_coloring(matrix, rsp=0.15, epsilon=1e-12)

    assert_valid_pagerank(scores, 1)
    assert_allclose(scores, [1.0], atol=1e-10)


def test_no_nan_or_inf():
    matrix = build_csr(3, [(0, 1), (1, 2)])
    scores = pagerank_coloring(matrix)

    assert np.isfinite(scores).all()
