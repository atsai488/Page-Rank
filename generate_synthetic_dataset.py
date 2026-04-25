from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class NodeInfo:
    node_id: int
    domain: str
    subdomain: str
    host: str
    url: str
    block_id: int


def build_hosts(
    n_domains: int,
    min_subdomains: int,
    max_subdomains: int,
    rng: random.Random,
) -> List[Tuple[str, str, str, int]]:
    """Create hosts with deterministic block IDs.

    Returns tuples: (domain, subdomain, host, block_id)
    """
    hosts: List[Tuple[str, str, str, int]] = []
    block_id = 0
    for d in range(n_domains):
        domain = f"domain{d}.com"
        n_sub = rng.randint(min_subdomains, max_subdomains)
        for s in range(n_sub):
            subdomain = f"sub{s}"
            host = f"{subdomain}.{domain}"
            hosts.append((domain, subdomain, host, block_id))
            block_id += 1
    return hosts


def build_nodes(
    hosts: Sequence[Tuple[str, str, str, int]],
    min_pages: int,
    max_pages: int,
    rng: random.Random,
) -> Tuple[List[NodeInfo], Dict[str, List[int]], Dict[str, List[int]]]:
    """Generate node metadata and lookup maps for sampling edges."""
    nodes: List[NodeInfo] = []
    host_to_nodes: Dict[str, List[int]] = {}
    domain_to_nodes: Dict[str, List[int]] = {}

    node_id = 0
    for domain, subdomain, host, block_id in hosts:
        n_pages = rng.randint(min_pages, max_pages)
        host_nodes: List[int] = []
        for page_idx in range(n_pages):
            url = f"https://{host}/page/{page_idx}"
            info = NodeInfo(
                node_id=node_id,
                domain=domain,
                subdomain=subdomain,
                host=host,
                url=url,
                block_id=block_id,
            )
            nodes.append(info)
            host_nodes.append(node_id)
            domain_to_nodes.setdefault(domain, []).append(node_id)
            node_id += 1
        host_to_nodes[host] = host_nodes

    return nodes, host_to_nodes, domain_to_nodes


def weighted_choice(
    rng: random.Random,
    p_same_subdomain: float,
    p_same_domain_other_subdomain: float,
) -> str:
    """Pick one of three edge target classes."""
    x = rng.random()
    if x < p_same_subdomain:
        return "same_subdomain"
    if x < p_same_subdomain + p_same_domain_other_subdomain:
        return "same_domain_other_subdomain"
    return "cross_domain"


def generate_edges(
    nodes: Sequence[NodeInfo],
    host_to_nodes: Dict[str, List[int]],
    domain_to_nodes: Dict[str, List[int]],
    avg_out_degree: int,
    p_same_subdomain: float,
    p_same_domain_other_subdomain: float,
    allow_self_loops: bool,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """Generate directed edges with host/domain locality bias."""
    all_node_ids = [n.node_id for n in nodes]
    edges: set[Tuple[int, int]] = set()

    for src in nodes:
        out_degree = max(1, int(rng.gauss(avg_out_degree, max(1.0, avg_out_degree * 0.25))))

        src_host_nodes = host_to_nodes[src.host]
        src_domain_nodes = domain_to_nodes[src.domain]
        other_domain_nodes = [nid for nid in all_node_ids if nid not in src_domain_nodes]

        for _ in range(out_degree):
            edge_type = weighted_choice(rng, p_same_subdomain, p_same_domain_other_subdomain)

            if edge_type == "same_subdomain":
                candidates = src_host_nodes
            elif edge_type == "same_domain_other_subdomain":
                candidates = [nid for nid in src_domain_nodes if nid not in src_host_nodes]
                if not candidates:
                    candidates = src_domain_nodes
            else:
                candidates = other_domain_nodes if other_domain_nodes else all_node_ids

            if not candidates:
                continue

            dst = rng.choice(candidates)
            if not allow_self_loops and dst == src.node_id:
                continue

            edges.add((src.node_id, dst))

    return sorted(edges)


def write_edge_list(path: Path, edges: Sequence[Tuple[int, int]]) -> None:
    """Write SNAP-style edge list (tab-separated, comment-compatible)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("# Synthetic directed web graph with subdomain structure\n")
        f.write("# Format: from\tto\n")
        for src, dst in edges:
            f.write(f"{src}\t{dst}\n")


def write_node_metadata(path: Path, nodes: Sequence[NodeInfo]) -> None:
    """Write per-node metadata including subdomain and block ID."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "domain", "subdomain", "host", "url", "block_id"])
        for n in nodes:
            writer.writerow([n.node_id, n.domain, n.subdomain, n.host, n.url, n.block_id])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic web graph dataset with domain/subdomain metadata."
    )
    parser.add_argument("--edge-output", default="data/synthetic-web-subdomains.txt")
    parser.add_argument("--metadata-output", default="data/synthetic-web-subdomains-nodes.csv")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n-domains", type=int, default=8)
    parser.add_argument("--min-subdomains", type=int, default=3)
    parser.add_argument("--max-subdomains", type=int, default=8)
    parser.add_argument("--min-pages-per-subdomain", type=int, default=40)
    parser.add_argument("--max-pages-per-subdomain", type=int, default=120)

    parser.add_argument("--avg-out-degree", type=int, default=20)
    parser.add_argument("--p-same-subdomain", type=float, default=0.65)
    parser.add_argument("--p-same-domain-other-subdomain", type=float, default=0.25)
    parser.add_argument("--allow-self-loops", action="store_true")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_domains < 1:
        raise ValueError("--n-domains must be >= 1")
    if args.min_subdomains < 1 or args.max_subdomains < args.min_subdomains:
        raise ValueError("Invalid subdomain range")
    if args.min_pages_per_subdomain < 1 or args.max_pages_per_subdomain < args.min_pages_per_subdomain:
        raise ValueError("Invalid pages-per-subdomain range")
    if args.avg_out_degree < 1:
        raise ValueError("--avg-out-degree must be >= 1")

    p1 = args.p_same_subdomain
    p2 = args.p_same_domain_other_subdomain
    if p1 < 0 or p2 < 0 or p1 + p2 > 1:
        raise ValueError("Probabilities must satisfy 0 <= p1, p2 and p1 + p2 <= 1")


def main() -> None:
    args = parse_args()
    validate_args(args)

    rng = random.Random(args.seed)

    hosts = build_hosts(
        n_domains=args.n_domains,
        min_subdomains=args.min_subdomains,
        max_subdomains=args.max_subdomains,
        rng=rng,
    )

    nodes, host_to_nodes, domain_to_nodes = build_nodes(
        hosts=hosts,
        min_pages=args.min_pages_per_subdomain,
        max_pages=args.max_pages_per_subdomain,
        rng=rng,
    )

    edges = generate_edges(
        nodes=nodes,
        host_to_nodes=host_to_nodes,
        domain_to_nodes=domain_to_nodes,
        avg_out_degree=args.avg_out_degree,
        p_same_subdomain=args.p_same_subdomain,
        p_same_domain_other_subdomain=args.p_same_domain_other_subdomain,
        allow_self_loops=args.allow_self_loops,
        rng=rng,
    )

    edge_output = Path(args.edge_output)
    metadata_output = Path(args.metadata_output)

    write_edge_list(edge_output, edges)
    write_node_metadata(metadata_output, nodes)

    print(f"Generated nodes: {len(nodes)}")
    print(f"Generated edges: {len(edges)}")
    print(f"Unique blocks (hosts): {len(hosts)}")
    print(f"Edge list: {edge_output}")
    print(f"Node metadata: {metadata_output}")
    print("Use `block_id` from metadata as block assignments for BlockRank.")


if __name__ == "__main__":
    main()