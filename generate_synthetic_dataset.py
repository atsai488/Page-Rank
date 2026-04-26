from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


@dataclass(frozen=True)
class HostInfo:
    domain_id: int
    subdomain_id: int
    domain: str
    subdomain: str
    host: str
    block_id: int
    node_start: int
    node_end: int


@dataclass(frozen=True)
class DomainInfo:
    domain_id: int
    domain: str
    node_start: int
    node_end: int


def build_layout(
    n_domains: int,
    min_subdomains: int,
    max_subdomains: int,
    min_pages_per_subdomain: int,
    max_pages_per_subdomain: int,
    rng: np.random.Generator,
) -> tuple[List[HostInfo], List[DomainInfo], int]:
    """Create host/domain layout and assign contiguous node-ID ranges."""
    hosts: List[HostInfo] = []
    domains: List[DomainInfo] = []

    block_id = 0
    node_cursor = 0

    for d in range(n_domains):
        domain = f"domain{d}.com"
        n_sub = int(rng.integers(min_subdomains, max_subdomains + 1))
        domain_start = node_cursor

        for s in range(n_sub):
            subdomain = f"sub{s}"
            host = f"{subdomain}.{domain}"
            n_pages = int(
                rng.integers(min_pages_per_subdomain, max_pages_per_subdomain + 1)
            )
            start = node_cursor
            end = start + n_pages

            hosts.append(
                HostInfo(
                    domain_id=d,
                    subdomain_id=s,
                    domain=domain,
                    subdomain=subdomain,
                    host=host,
                    block_id=block_id,
                    node_start=start,
                    node_end=end,
                )
            )

            node_cursor = end
            block_id += 1
        domains.append(
            DomainInfo(
                domain_id=d,
                domain=domain,
                node_start=domain_start,
                node_end=node_cursor,
            )
        )

    return hosts, domains, node_cursor


def _sample_in_range(start: int, end: int, rng: np.random.Generator) -> int:
    return int(rng.integers(start, end))


def _sample_outside_range(
    full_start: int,
    full_end: int,
    excluded_start: int,
    excluded_end: int,
    rng: np.random.Generator,
) -> int:
    """Uniformly sample from [full_start, full_end) excluding [excluded_start, excluded_end)."""
    full_size = full_end - full_start
    excluded_size = excluded_end - excluded_start
    allowed_size = full_size - excluded_size
    if allowed_size <= 0:
        raise ValueError("No nodes available after excluding range")

    k = int(rng.integers(0, allowed_size))
    left_size = excluded_start - full_start
    if k < left_size:
        return full_start + k
    return excluded_end + (k - left_size)


def _out_degree(
    degree_model: str,
    avg_out_degree: int,
    rng: np.random.Generator,
) -> int:
    if degree_model == "fixed":
        return avg_out_degree
    # poisson
    return max(1, int(rng.poisson(avg_out_degree)))


def _target_out_degree(node_id: int, total_nodes: int, target_edges: int) -> int:
    """Deterministically distribute an exact edge budget across nodes."""
    base = target_edges // total_nodes
    remainder = target_edges % total_nodes
    return base + (1 if node_id < remainder else 0)


def generate_edge_list_streaming(
    edge_path: Path,
    hosts: List[HostInfo],
    domain_starts: np.ndarray,
    domain_ends: np.ndarray,
    total_nodes: int,
    avg_out_degree: int,
    target_edges: int | None,
    degree_model: str,
    p_same_subdomain: float,
    p_same_domain_other_subdomain: float,
    allow_self_loops: bool,
    rng: np.random.Generator,
) -> int:
    """Generate and write edges directly to disk (memory-efficient)."""
    edge_path.parent.mkdir(parents=True, exist_ok=True)
    edges_written = 0

    with edge_path.open("w", encoding="utf-8", newline="", buffering=16 * 1024 * 1024) as f:
        f.write("# Synthetic directed web graph with subdomain structure\n")
        f.write("# Format: from\tto\n")

        write_buffer: list[str] = []
        flush_every = 200_000
        p_domain_total = p_same_subdomain + p_same_domain_other_subdomain

        for host in hosts:
            host_start, host_end = host.node_start, host.node_end
            domain_start = int(domain_starts[host.domain_id])
            domain_end = int(domain_ends[host.domain_id])
            host_size = host_end - host_start
            domain_size = domain_end - domain_start
            has_other_in_domain = (domain_size - host_size) > 0
            has_cross_domain = (total_nodes - domain_size) > 0

            for src in range(host_start, host_end):
                if target_edges is not None:
                    out_deg = _target_out_degree(src, total_nodes, target_edges)
                else:
                    out_deg = _out_degree(degree_model, avg_out_degree, rng)

                for _ in range(out_deg):
                    x = float(rng.random())

                    if x < p_same_subdomain:
                        dst = _sample_in_range(host_start, host_end, rng)
                    elif x < p_domain_total and has_other_in_domain:
                        dst = _sample_outside_range(
                            domain_start,
                            domain_end,
                            host_start,
                            host_end,
                            rng,
                        )
                    elif has_cross_domain:
                        dst = _sample_outside_range(
                            0,
                            total_nodes,
                            domain_start,
                            domain_end,
                            rng,
                        )
                    else:
                        dst = _sample_in_range(0, total_nodes, rng)

                    if not allow_self_loops and dst == src:
                        # Single-node ranges can still produce self-loops; skip those.
                        continue

                    write_buffer.append(f"{src}\t{dst}\n")
                    edges_written += 1

                    if len(write_buffer) >= flush_every:
                        f.writelines(write_buffer)
                        write_buffer.clear()

        if write_buffer:
            f.writelines(write_buffer)

    return edges_written


def _write_edge_shard(
    shard_path: str,
    hosts: List[HostInfo],
    domain_starts: np.ndarray,
    domain_ends: np.ndarray,
    total_nodes: int,
    avg_out_degree: int,
    target_edges: int | None,
    degree_model: str,
    p_same_subdomain: float,
    p_same_domain_other_subdomain: float,
    allow_self_loops: bool,
    seed: int,
) -> int:
    """Worker entry point: write one shard file and return number of edges written."""
    rng = np.random.default_rng(seed)
    shard = Path(shard_path)
    shard.parent.mkdir(parents=True, exist_ok=True)

    edges_written = 0
    p_domain_total = p_same_subdomain + p_same_domain_other_subdomain
    flush_every = 200_000
    write_buffer: list[str] = []

    with shard.open("w", encoding="utf-8", newline="", buffering=16 * 1024 * 1024) as f:
        for host in hosts:
            host_start, host_end = host.node_start, host.node_end
            domain_start = int(domain_starts[host.domain_id])
            domain_end = int(domain_ends[host.domain_id])
            host_size = host_end - host_start
            domain_size = domain_end - domain_start
            has_other_in_domain = (domain_size - host_size) > 0
            has_cross_domain = (total_nodes - domain_size) > 0

            for src in range(host_start, host_end):
                if target_edges is not None:
                    out_deg = _target_out_degree(src, total_nodes, target_edges)
                else:
                    out_deg = _out_degree(degree_model, avg_out_degree, rng)

                for _ in range(out_deg):
                    x = float(rng.random())

                    if x < p_same_subdomain:
                        dst = _sample_in_range(host_start, host_end, rng)
                    elif x < p_domain_total and has_other_in_domain:
                        dst = _sample_outside_range(
                            domain_start,
                            domain_end,
                            host_start,
                            host_end,
                            rng,
                        )
                    elif has_cross_domain:
                        dst = _sample_outside_range(
                            0,
                            total_nodes,
                            domain_start,
                            domain_end,
                            rng,
                        )
                    else:
                        dst = _sample_in_range(0, total_nodes, rng)

                    if not allow_self_loops and dst == src:
                        continue

                    write_buffer.append(f"{src}\t{dst}\n")
                    edges_written += 1

                    if len(write_buffer) >= flush_every:
                        f.writelines(write_buffer)
                        write_buffer.clear()

        if write_buffer:
            f.writelines(write_buffer)

    return edges_written


def _chunk_hosts_by_node_count(hosts: List[HostInfo], workers: int) -> List[List[HostInfo]]:
    """Split hosts into contiguous chunks with roughly balanced node counts."""
    if workers <= 1:
        return [hosts]

    total_nodes = sum(h.node_end - h.node_start for h in hosts)
    target = max(1, total_nodes // workers)

    chunks: List[List[HostInfo]] = []
    current: List[HostInfo] = []
    current_nodes = 0
    assigned_hosts = 0

    for host in hosts:
        host_nodes = host.node_end - host.node_start
        current.append(host)
        current_nodes += host_nodes
        assigned_hosts += 1

        remaining_hosts = len(hosts) - assigned_hosts
        # Keep one slot for the tail chunk so final chunk count <= workers.
        can_split_more = len(chunks) < workers - 1
        if current_nodes >= target and can_split_more and remaining_hosts > 0:
            chunks.append(current)
            current = []
            current_nodes = 0

    if current:
        chunks.append(current)

    return chunks


def _merge_shards(shard_paths: List[Path], edge_output: Path) -> None:
    """Merge shard files into final SNAP-style edge list with one header."""
    edge_output.parent.mkdir(parents=True, exist_ok=True)
    with edge_output.open("w", encoding="utf-8", newline="", buffering=16 * 1024 * 1024) as out:
        out.write("# Synthetic directed web graph with subdomain structure\n")
        out.write("# Format: from\tto\n")

        for shard in shard_paths:
            with shard.open("r", encoding="utf-8", newline="", buffering=16 * 1024 * 1024) as src:
                shutil.copyfileobj(src, out, length=16 * 1024 * 1024)


def generate_edge_list_parallel(
    edge_output: Path,
    hosts: List[HostInfo],
    domain_starts: np.ndarray,
    domain_ends: np.ndarray,
    total_nodes: int,
    workers: int,
    avg_out_degree: int,
    target_edges: int | None,
    degree_model: str,
    p_same_subdomain: float,
    p_same_domain_other_subdomain: float,
    allow_self_loops: bool,
    seed: int,
) -> int:
    """Generate edge shards in parallel and merge into one output file."""
    chunks = _chunk_hosts_by_node_count(hosts, workers)
    tmp_dir = edge_output.parent / f".edge_shards_{edge_output.stem}_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = [tmp_dir / f"part_{i:03d}.txt" for i in range(len(chunks))]
    total_edges = 0

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = []
            for i, host_chunk in enumerate(chunks):
                worker_seed = seed + (i + 1) * 1_000_003
                futures.append(
                    pool.submit(
                        _write_edge_shard,
                        str(shard_paths[i]),
                        host_chunk,
                        domain_starts,
                        domain_ends,
                        total_nodes,
                        avg_out_degree,
                        target_edges,
                        degree_model,
                        p_same_subdomain,
                        p_same_domain_other_subdomain,
                        allow_self_loops,
                        worker_seed,
                    )
                )

            for future in as_completed(futures):
                total_edges += future.result()

        _merge_shards(shard_paths, edge_output)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return total_edges


def write_node_metadata(path: Path, hosts: List[HostInfo]) -> int:
    """Write per-node metadata including subdomain and block ID."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes_written = 0

    with path.open("w", encoding="utf-8", newline="", buffering=16 * 1024 * 1024) as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "domain", "subdomain", "host", "url", "block_id"])
        for host in hosts:
            for node_id in range(host.node_start, host.node_end):
                page_idx = node_id - host.node_start
                url = f"https://{host.host}/page/{page_idx}"
                writer.writerow(
                    [
                        node_id,
                        host.domain,
                        host.subdomain,
                        host.host,
                        url,
                        host.block_id,
                    ]
                )
                nodes_written += 1

    return nodes_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic web graph dataset with domain/subdomain metadata."
    )
    parser.add_argument("--edge-output", default="data/synthetic-web-subdomains.txt")
    parser.add_argument("--metadata-output", default="data/synthetic-web-subdomains-nodes.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-edges",
        type=int,
        default=None,
        help="Exact number of edges to generate. Overrides --avg-out-degree when set.",
    )

    parser.add_argument("--n-domains", type=int, default=8)
    parser.add_argument("--min-subdomains", type=int, default=3)
    parser.add_argument("--max-subdomains", type=int, default=8)
    parser.add_argument("--min-pages-per-subdomain", type=int, default=40)
    parser.add_argument("--max-pages-per-subdomain", type=int, default=120)

    parser.add_argument("--avg-out-degree", type=int, default=20)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for edge generation. Use >1 for shard-based parallel mode.",
    )
    parser.add_argument(
        "--degree-model",
        choices=["fixed", "poisson"],
        default="fixed",
        help="Out-degree model. 'fixed' is faster and more predictable at scale.",
    )
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
    if args.target_edges is not None and args.target_edges < 1:
        raise ValueError("--target-edges must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    p1 = args.p_same_subdomain
    p2 = args.p_same_domain_other_subdomain
    if p1 < 0 or p2 < 0 or p1 + p2 > 1:
        raise ValueError("Probabilities must satisfy 0 <= p1, p2 and p1 + p2 <= 1")


def main() -> None:
    args = parse_args()
    validate_args(args)

    rng = np.random.default_rng(args.seed)

    hosts, domains, total_nodes = build_layout(
        n_domains=args.n_domains,
        min_subdomains=args.min_subdomains,
        max_subdomains=args.max_subdomains,
        min_pages_per_subdomain=args.min_pages_per_subdomain,
        max_pages_per_subdomain=args.max_pages_per_subdomain,
        rng=rng,
    )

    domain_starts = np.array([d.node_start for d in domains], dtype=np.int64)
    domain_ends = np.array([d.node_end for d in domains], dtype=np.int64)

    edge_output = Path(args.edge_output)
    metadata_output = Path(args.metadata_output)

    if args.workers == 1:
        generated_edges = generate_edge_list_streaming(
            edge_path=edge_output,
            hosts=hosts,
            domain_starts=domain_starts,
            domain_ends=domain_ends,
            total_nodes=total_nodes,
            avg_out_degree=args.avg_out_degree,
            target_edges=args.target_edges,
            degree_model=args.degree_model,
            p_same_subdomain=args.p_same_subdomain,
            p_same_domain_other_subdomain=args.p_same_domain_other_subdomain,
            allow_self_loops=args.allow_self_loops,
            rng=rng,
        )
    else:
        generated_edges = generate_edge_list_parallel(
            edge_output=edge_output,
            hosts=hosts,
            domain_starts=domain_starts,
            domain_ends=domain_ends,
            total_nodes=total_nodes,
            workers=args.workers,
            avg_out_degree=args.avg_out_degree,
            target_edges=args.target_edges,
            degree_model=args.degree_model,
            p_same_subdomain=args.p_same_subdomain,
            p_same_domain_other_subdomain=args.p_same_domain_other_subdomain,
            allow_self_loops=args.allow_self_loops,
            seed=args.seed,
        )

    generated_nodes = write_node_metadata(metadata_output, hosts)

    print(f"Generated nodes: {generated_nodes}")
    print(f"Generated edges: {generated_edges}")
    if args.target_edges is not None:
        print(f"Target edges: {args.target_edges}")
    print(f"Unique blocks (hosts): {len(hosts)}")
    print(f"Workers used for edges: {args.workers}")
    print(f"Edge list: {edge_output}")
    print(f"Node metadata: {metadata_output}")
    print("Use `block_id` from metadata as block assignments for BlockRank.")


if __name__ == "__main__":
    main()