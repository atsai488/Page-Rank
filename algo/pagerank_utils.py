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


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.sparse import csr_matrix


def _extract(scores: pd.Series | np.ndarray | dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (node_ids, values) from any scores input."""
    if isinstance(scores, dict):
        return np.array(list(scores.keys())), np.array(list(scores.values()))
    elif isinstance(scores, pd.Series):
        return scores.index.to_numpy(), scores.values
    else:
        values = np.asarray(scores)
        return np.arange(len(values)), values


def plot_score_per_node(scores, ax=None, **kwargs):
    """Node ID vs PageRank score (log y). Good for spotting outlier nodes."""
    node_ids, values = _extract(scores)
    order = np.argsort(node_ids)
    ax = ax or plt.gca()
    ax.plot(node_ids[order], values[order], linewidth=0.5, color="#1a1a4e", **kwargs)
    ax.set_yscale("log")
    ax.set_xlabel("Node ID")
    ax.set_ylabel("PageRank score")
    ax.set_title("Score per node")
    return ax


def plot_rank_vs_score(scores, ax=None, **kwargs):
    """Log-log rank vs score. A straight line indicates a power-law distribution."""
    _, values = _extract(scores)
    ranked = np.sort(values)[::-1]
    ranks = np.arange(1, len(ranked) + 1)
    ax = ax or plt.gca()
    ax.loglog(ranks, ranked, linewidth=0.8, color="#1a1a4e", **kwargs)
    ax.set_xlabel("Rank")
    ax.set_ylabel("PageRank score")
    ax.set_title("Rank vs score (log-log)")
    return ax


def plot_score_distribution(scores, bins=100, ax=None, **kwargs):
    """Histogram of log10 scores. Shows where most nodes cluster."""
    _, values = _extract(scores)
    log_scores = np.log10(values[values > 0])
    ax = ax or plt.gca()
    ax.hist(log_scores, bins=bins, density=True, color="#1a1a4e", edgecolor="none", **kwargs)
    ax.set_xlabel("log₁₀(PageRank score)")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution")
    return ax


def plot_cumulative_distribution(scores, ax=None, **kwargs):
    """CDF of PageRank scores. Shows what fraction of nodes fall below a given score."""
    _, values = _extract(scores)
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax = ax or plt.gca()
    ax.plot(sorted_vals, cdf, linewidth=0.8, color="#1a1a4e", **kwargs)
    ax.set_xscale("log")
    ax.set_xlabel("PageRank score")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Cumulative distribution")
    return ax


def plot_top_n_nodes(scores, n=20, ax=None, **kwargs):
    """Horizontal bar chart of the top-n nodes by PageRank score."""
    node_ids, values = _extract(scores)
    top_idx = np.argsort(values)[::-1][:n]
    top_nodes = node_ids[top_idx].astype(str)
    top_scores = values[top_idx]
    ax = ax or plt.gca()
    bars = ax.barh(top_nodes[::-1], top_scores[::-1], color="#1a1a4e", **kwargs)
    ax.set_xlabel("PageRank score")
    ax.set_ylabel("Node ID")
    ax.set_title(f"Top {n} nodes")
    ax.bar_label(bars, fmt="%.2e", padding=3, fontsize=7)
    return ax


def plot_score_vs_out_degree(scores, matrix: csr_matrix, ax=None, **kwargs):
    """Scatter of out-degree vs PageRank score. Reveals whether degree predicts rank."""
    _, values = _extract(scores)
    out_degree = np.asarray(matrix.sum(axis=1)).flatten()
    ax = ax or plt.gca()
    ax.scatter(out_degree, values, s=1, alpha=0.3, color="#1a1a4e", rasterized=True, **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Out-degree")
    ax.set_ylabel("PageRank score")
    ax.set_title("Score vs out-degree")
    return ax


def plot_score_vs_in_degree(scores, matrix: csr_matrix, ax=None, **kwargs):
    """Scatter of in-degree vs PageRank score. In-degree is the stronger predictor."""
    _, values = _extract(scores)
    in_degree = np.asarray(matrix.sum(axis=0)).flatten()
    ax = ax or plt.gca()
    ax.scatter(in_degree, values, s=1, alpha=0.3, color="#1a1a4e", rasterized=True, **kwargs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("In-degree")
    ax.set_ylabel("PageRank score")
    ax.set_title("Score vs in-degree")
    return ax


def plot_score_concentration(scores, ax=None):
    """Lorenz curve showing how concentrated PageRank is among top nodes."""
    _, values = _extract(scores)
    sorted_vals = np.sort(values)
    cumulative = np.cumsum(sorted_vals) / sorted_vals.sum()
    x = np.linspace(0, 1, len(cumulative))
    ax = ax or plt.gca()
    ax.plot(x, cumulative, color="#1a1a4e", linewidth=0.8, label="PageRank")
    ax.plot([0, 1], [0, 1], color="gray", linewidth=0.8, linestyle="--", label="Perfect equality")
    ax.set_xlabel("Fraction of nodes")
    ax.set_ylabel("Fraction of total PageRank")
    ax.set_title("Lorenz curve (score concentration)")
    ax.legend(fontsize=8)
    return ax


def plot_all(filepath: str, technique: str, scores, matrix: csr_matrix = None) -> None:
    """
    Produce a full analysis dashboard — all 7 plots in one figure.
    Pass matrix to include degree-vs-score scatter plots.
    """
    has_matrix = matrix is not None
    n_plots = 8 if has_matrix else 6
    ncols = 4 if has_matrix else 3
    nrows = 2

    fig = plt.figure(figsize=(ncols * 4, nrows * 4))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.35)

    plot_score_per_node(scores,        ax=fig.add_subplot(gs[0, 0]))
    plot_rank_vs_score(scores,         ax=fig.add_subplot(gs[0, 1]))
    plot_score_distribution(scores,    ax=fig.add_subplot(gs[0, 2]))
    plot_cumulative_distribution(scores, ax=fig.add_subplot(gs[1, 0]))
    plot_top_n_nodes(scores,           ax=fig.add_subplot(gs[1, 1]))
    plot_score_concentration(scores,   ax=fig.add_subplot(gs[1, 2]))

    if has_matrix:
        plot_score_vs_out_degree(scores, matrix, ax=fig.add_subplot(gs[0, 3]))
        plot_score_vs_in_degree(scores,  matrix, ax=fig.add_subplot(gs[1, 3]))

    dataset_name = os.path.splitext(os.path.basename(filepath))[0]
    fig.suptitle(f"{dataset_name} with {technique}", fontsize=14, y=1.01)
    
    output_path = f"output/{technique}/{dataset_name}_analysis.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
def parse_to_csr(filepath: str) -> tuple[csr_matrix, np.ndarray]:
    edges = pd.read_csv(
        filepath,
        sep='\t',
        comment='#',
        names=['from', 'to'],
        dtype={'from': 'int32', 'to': 'int32'},
    )

    sorted_nodes, inverse = np.unique(
        np.concatenate([edges['from'].values, edges['to'].values]),
        return_inverse=True,
    )
    n = len(sorted_nodes)
    rows = inverse[:len(edges)]
    cols = inverse[len(edges):]
    data = np.ones(len(edges), dtype=np.float32)

    matrix = csr_matrix((data, (rows, cols)), shape=(n, n))
    return matrix, sorted_nodes
