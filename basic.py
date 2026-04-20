from collections.abc import Hashable, Mapping, Sequence
from typing import Any
import pandas

from pagerank_utils import pagerank_power_iteration


def _from_mapping(
    transition_weights: Mapping[Hashable, Mapping[Hashable, float | int]],
) -> tuple[dict[Hashable, dict[Hashable, float]], list[Hashable]]:
    nodes: set[Hashable] = set(transition_weights.keys())
    graph: dict[Hashable, dict[Hashable, float]] = {}

    for source, row in transition_weights.items():
        weights: dict[Hashable, float] = {}
        for dest, weight in row.items():
            nodes.add(dest)
            value = float(weight)
            if value > 0.0:
                weights[dest] = value
        graph[source] = weights

    # Deterministic ordering helps reproducibility across runs.
    ordered_nodes = sorted(nodes, key=str)
    return graph, ordered_nodes


def _from_sequence(
    transition_weights: Sequence[Sequence[float | int]],
) -> tuple[dict[int, dict[int, float]], list[int]]:
    row_count = len(transition_weights)
    col_count = max((len(row) for row in transition_weights), default=0)
    size = max(row_count, col_count)

    graph: dict[int, dict[int, float]] = {i: {} for i in range(size)}
    for i, row in enumerate(transition_weights):
        for j, weight in enumerate(row):
            value = float(weight)
            if value > 0.0:
                graph[i][j] = value

    return graph, list(range(size))


def power_iteration(
    transition_weights: Mapping[Hashable, Mapping[Hashable, float | int]]
    | Sequence[Sequence[float | int]],
    rsp: float = 0.15,
    epsilon: float = 0.00001,
    max_iterations: int = 1000,
) -> Any:
    """Apply PageRank algorithm using power iteration to find steady-state probabilities.

    This function applies the PageRank algorithm to a provided graph to determine
    the steady probabilities with which a random walk through the graph will end up
    at each node. It uses power iteration, an algorithm that iteratively refines
    the steady state probabilities until convergence.

    Args:
        transition_weights: Sparse representation of the graph as nested dicts or lists.
            Keys correspond to node names and values to weights. Elements need not be
            probabilities (rows need not be normalized).
        rsp: Random surfer probability controlling the chance of jumping to any node.
            Also known as the damping factor (1 - rsp is the damping factor).
        epsilon: Threshold of convergence; iteration stops when successive approximations
            are closer than this value.
        max_iterations: Maximum number of iterations before termination even without convergence.

    Returns:
        A mapping whose keys are node names and whose values are the corresponding
        steady state probabilities. If pandas is installed, a pandas Series is returned.

    Example:
        >>> import logging
        >>> logging.basicConfig(level=logging.INFO)
        >>> graph = {
        ...     "A": {"B": 1, "C": 1},
        ...     "B": {"C": 1},
        ...     "C": {"A": 1},
        ... }
        >>> scores = power_iteration(graph)
        >>> logging.info(scores)
    """
    if isinstance(transition_weights, Mapping):
        graph, nodes = _from_mapping(transition_weights)
    else:
        graph, nodes = _from_sequence(transition_weights)

    # Existing API exposes random-surfer probability (rsp); alpha is follow-link probability.
    alpha = 1.0 - rsp
    scores: dict[Any, float] = pagerank_power_iteration(
        graph,
        nodes,
        alpha=alpha,
        personalization=None,
        start=None,
        max_iter=max_iterations,
        tol=epsilon,
    )
    if pandas is not None:
        return pandas.Series(scores)
    return scores