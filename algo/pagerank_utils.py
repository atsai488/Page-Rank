from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import Dict, Optional, TypeVar
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

Node = TypeVar("Node", bound=Hashable)


def normalize_distribution(
    dist: Mapping[Node, float],
    nodes: Optional[Iterable[Node]] = None,
) -> Dict[Node, float]:
    """Normalize a non-negative distribution to sum to 1."""
    if nodes is None:
        result = {k: float(v) for k, v in dist.items()}
    else:
        result = {n: float(dist.get(n, 0.0)) for n in nodes}

    total = sum(result.values())
    if total <= 0.0:
        raise ValueError("Distribution has zero total mass.")

    return {k: v / total for k, v in result.items()}


def make_uniform(nodes: Iterable[Node]) -> Dict[Node, float]:
    """Create a uniform probability distribution over nodes."""
    node_list = list(nodes)
    if not node_list:
        raise ValueError("Cannot build a uniform distribution over an empty set.")

    mass = 1.0 / len(node_list)
    return {node: mass for node in node_list}


def pagerank_power_iteration(
    graph: Mapping[Node, Iterable[Node] | Mapping[Node, float]],
    nodes: Iterable[Node],
    alpha: float = 0.85,
    personalization: Optional[Mapping[Node, float]] = None,
    start: Optional[Mapping[Node, float]] = None,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> Dict[Node, float]:
    """Compute PageRank with power iteration.

    The graph can be either:
    - unweighted adjacency: node -> iterable of neighbors
    - weighted adjacency: node -> mapping(neighbor -> weight)
    """
    node_list = list(nodes)
    if not node_list:
        return {}

    node_set = set(node_list)

    if personalization is None:
        p = make_uniform(node_list)
    else:
        p = normalize_distribution(personalization, node_list)

    if start is None:
        rank = make_uniform(node_list)
    else:
        rank = normalize_distribution(start, node_list)

    weighted_out: Dict[Node, Dict[Node, float]] = {}
    out_weight_sum: Dict[Node, float] = {}

    for node in node_list:
        raw_adjacency = graph.get(node, {})

        if isinstance(raw_adjacency, Mapping):
            row = {
                neighbor: float(weight)
                for neighbor, weight in raw_adjacency.items()
                if neighbor in node_set and float(weight) > 0.0
            }
        else:
            row = {}
            for neighbor in raw_adjacency:
                if neighbor in node_set:
                    row[neighbor] = row.get(neighbor, 0.0) + 1.0

        weighted_out[node] = row
        out_weight_sum[node] = sum(row.values())

    for _ in range(max_iter):
        new_rank = {node: (1.0 - alpha) * p[node] for node in node_list}
        dangling_mass = 0.0

        for node in node_list:
            rank_value = rank[node]
            total = out_weight_sum[node]
            if total <= 0.0:
                dangling_mass += rank_value
                continue

            scale = alpha * rank_value / total
            for neighbor, weight in weighted_out[node].items():
                new_rank[neighbor] += scale * weight

        if dangling_mass > 0.0:
            redistributed = alpha * dangling_mass
            for node in node_list:
                new_rank[node] += redistributed * p[node]

        l1_delta = sum(abs(new_rank[node] - rank[node]) for node in node_list)
        rank = new_rank
        if l1_delta < tol:
            break

    return normalize_distribution(rank)


def plot_pagerank_distribution(filepath: str, scores: pd.Series | np.ndarray | dict) -> None:
    if isinstance(scores, dict):
        values = np.array(list(scores.values()))
        node_ids = np.array(list(scores.keys()))
    elif isinstance(scores, pd.Series):
        values = scores.values
        node_ids = scores.index.to_numpy()
    else:
        values = np.asarray(scores)
        node_ids = np.arange(len(values))

    # Sort by node ID for the left plot
    sort_by_node = np.argsort(node_ids)
    sorted_node_ids = node_ids[sort_by_node]
    sorted_by_node = values[sort_by_node]

    # Sort by score descending for the right plot
    sort_by_score = np.argsort(values)[::-1]
    ranked_scores = values[sort_by_score]
    ranks = np.arange(1, len(ranked_scores) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(sorted_node_ids, sorted_by_node, color="#1a1a4e", linewidth=0.5)
    axes[0].set_xlabel("Node ID")
    axes[0].set_ylabel("PageRank score")
    axes[0].set_title("Score per node")
    axes[0].set_yscale("log")

    axes[1].loglog(ranks, ranked_scores, color="#1a1a4e", linewidth=0.8)
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("PageRank score")
    axes[1].set_title("Rank vs score (log-log)")

    dataset_name = os.path.splitext(os.path.basename(filepath))[0]
    fig.suptitle(dataset_name, fontsize=13)
    plt.tight_layout()

    output_path = f"output/distributions/{dataset_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")