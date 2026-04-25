from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate synthetic graph edge list and node metadata."
    )
    parser.add_argument("--edge-file", required=True)
    parser.add_argument("--metadata-file", required=True)
    parser.add_argument("--expected-nodes", type=int, default=None)
    parser.add_argument("--expected-edges", type=int, default=None)
    return parser.parse_args()


def validate_metadata(path: Path) -> dict:
    required = {"node_id", "domain", "subdomain", "host", "url", "block_id"}

    node_count = 0
    min_node = None
    max_node = None
    sequential_nodes = True
    prev_node = -1

    block_counts: dict[int, int] = defaultdict(int)
    host_counts: dict[str, int] = defaultdict(int)
    host_block: dict[str, int] = {}
    domain_counts: dict[str, int] = defaultdict(int)

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Metadata CSV is missing a header row")

        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Metadata CSV missing columns: {sorted(missing)}")

        for row in reader:
            node_id = int(row["node_id"])
            block_id = int(row["block_id"])
            host = row["host"]
            domain = row["domain"]

            if node_id != prev_node + 1:
                sequential_nodes = False
            prev_node = node_id

            if min_node is None or node_id < min_node:
                min_node = node_id
            if max_node is None or node_id > max_node:
                max_node = node_id

            if host in host_block and host_block[host] != block_id:
                raise ValueError(
                    f"Host {host} appears with multiple block IDs: "
                    f"{host_block[host]} and {block_id}"
                )

            host_block[host] = block_id
            block_counts[block_id] += 1
            host_counts[host] += 1
            domain_counts[domain] += 1
            node_count += 1

    if node_count == 0:
        raise ValueError("Metadata CSV has zero nodes")

    return {
        "node_count": node_count,
        "min_node": min_node,
        "max_node": max_node,
        "sequential_nodes": sequential_nodes,
        "block_counts": block_counts,
        "host_counts": host_counts,
        "domain_counts": domain_counts,
    }


def validate_edges(path: Path) -> dict:
    edge_count = 0
    min_node_seen = None
    max_node_seen = None
    self_loops = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Invalid edge line: {line[:120]}")

            src = int(parts[0])
            dst = int(parts[1])
            edge_count += 1

            if src == dst:
                self_loops += 1

            if min_node_seen is None:
                min_node_seen = min(src, dst)
                max_node_seen = max(src, dst)
            else:
                min_node_seen = min(min_node_seen, src, dst)
                max_node_seen = max(max_node_seen, src, dst)

    if edge_count == 0:
        raise ValueError("Edge list has zero edges")

    return {
        "edge_count": edge_count,
        "min_node_seen": min_node_seen,
        "max_node_seen": max_node_seen,
        "self_loops": self_loops,
    }


def summarize_distribution(counts: dict[int | str, int], top_k: int = 5) -> tuple[int, int, list[tuple[int | str, int]]]:
    values = list(counts.values())
    return min(values), max(values), sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


def main() -> None:
    args = parse_args()
    edge_file = Path(args.edge_file)
    metadata_file = Path(args.metadata_file)

    if not edge_file.exists():
        raise FileNotFoundError(f"Edge file does not exist: {edge_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_file}")

    meta = validate_metadata(metadata_file)
    edges = validate_edges(edge_file)

    node_count = meta["node_count"]
    edge_count = edges["edge_count"]
    avg_out_degree = edge_count / node_count

    print("Validation summary")
    print(f"- Nodes in metadata: {node_count}")
    print(f"- Edges in edge list: {edge_count}")
    print(f"- Avg out-degree (edges/nodes): {avg_out_degree:.4f}")
    print(f"- Node ID range in metadata: [{meta['min_node']}, {meta['max_node']}]")
    print(f"- Node IDs sequential from 0 in metadata: {meta['sequential_nodes']}")
    print(f"- Node ID range seen in edges: [{edges['min_node_seen']}, {edges['max_node_seen']}]")
    print(f"- Self-loops in edges: {edges['self_loops']}")

    block_min, block_max, top_blocks = summarize_distribution(meta["block_counts"])
    host_min, host_max, top_hosts = summarize_distribution(meta["host_counts"])
    domain_min, domain_max, top_domains = summarize_distribution(meta["domain_counts"])

    print(f"- Unique blocks: {len(meta['block_counts'])} (min/max nodes per block: {block_min}/{block_max})")
    print(f"- Unique hosts: {len(meta['host_counts'])} (min/max nodes per host: {host_min}/{host_max})")
    print(f"- Unique domains: {len(meta['domain_counts'])} (min/max nodes per domain: {domain_min}/{domain_max})")

    print("- Top 5 blocks by node count:")
    for block_id, cnt in top_blocks:
        print(f"  block={block_id} nodes={cnt}")

    print("- Top 5 hosts by node count:")
    for host, cnt in top_hosts:
        print(f"  host={host} nodes={cnt}")

    print("- Top 5 domains by node count:")
    for domain, cnt in top_domains:
        print(f"  domain={domain} nodes={cnt}")

    if args.expected_nodes is not None:
        ok = node_count == args.expected_nodes
        print(f"- Expected nodes check ({args.expected_nodes}): {ok}")
        if not ok:
            raise ValueError(f"Expected {args.expected_nodes} nodes, got {node_count}")

    if args.expected_edges is not None:
        ok = edge_count == args.expected_edges
        print(f"- Expected edges check ({args.expected_edges}): {ok}")
        if not ok:
            raise ValueError(f"Expected {args.expected_edges} edges, got {edge_count}")


if __name__ == "__main__":
    main()