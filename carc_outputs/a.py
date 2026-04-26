import re
import os
import matplotlib.pyplot as plt
import pandas as pd

LOG_DIR = "block_coloring_sweep/worker_threads-local_block_size"
OUT_DIR = "viz_sweep"

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Patterns
# ----------------------------
pattern_dataset   = re.compile(r"synth_([0-9]+(?:\.[0-9]+)?)([mk])_([0-9]+(?:\.[0-9]+)?)([mk])", re.IGNORECASE)
pattern_workers   = re.compile(r"local-workers\s*[=:]\s*([0-9]+)")
pattern_chunk     = re.compile(r"local-chunk-size\s*[=:]\s*([0-9]+)")
pattern_total_time = re.compile(
    r"^(?:"
    r"Total\s+\S[\w+\s]*time\s*:\s*([0-9.]+)s?"    # "Total BlockRank+Coloring time: X.Xs"
    r"|Total\s+time\s*:\s*([0-9.]+)s?"              # "Total time: X.Xs"
    r"|Time\s+for\s+\w[\w\s]*?\s+([0-9.]+)\s*$"    # "Time for power iteration X.X"
    r")",
    re.IGNORECASE,
)

def parse_count(value, suffix):
    value = float(value)
    return int(value * 1_000_000) if suffix == "m" else int(value * 1_000) if suffix == "k" else int(value)

# ----------------------------
# Parse: one "block" per === separator
# ----------------------------
data = []

for file in sorted(os.listdir(LOG_DIR)):
    if not file.endswith(".log"):
        continue

    path = os.path.join(LOG_DIR, file)
    with open(path) as f:
        content = f.read()

    # Split into per-run blocks on the === separator
    blocks = re.split(r"={10,}", content)

    # State carried across adjacent header / run blocks
    workers    = None
    chunk_size = None
    nodes      = None
    edges      = None

    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue

        # Check if this block is a sweep header (contains local-workers)
        w_match = pattern_workers.search(block)
        c_match = pattern_chunk.search(block)
        d_match = pattern_dataset.search(block)

        if w_match:
            workers = int(w_match.group(1))
        if c_match:
            chunk_size = int(c_match.group(1))
        if d_match:
            nodes = parse_count(d_match.group(1), d_match.group(2))
            edges = parse_count(d_match.group(3), d_match.group(4))

        # Look for a total-time line anywhere in this block
        for line in lines:
            m = pattern_total_time.match(line.strip())
            if m and workers is not None and chunk_size is not None:
                time_str = m.group(1) or m.group(2) or m.group(3)
                data.append({
                    "nodes":      nodes,
                    "edges":      edges,
                    "workers":    workers,
                    "chunk_size": chunk_size,
                    "time":       float(time_str),
                })
                break   # one total time per run block

df = pd.DataFrame(data)

if df.empty:
    raise SystemExit(f"No timing data found in {LOG_DIR}")

print(f"Parsed {len(df)} records")
print(df.to_string(index=False))

# Average across any duplicate (nodes, edges, workers, chunk_size) combos
df = df.groupby(["nodes", "edges", "workers", "chunk_size"], as_index=False).agg(time=("time", "mean"))

# ----------------------------
# Heatmaps: one per (nodes, edges) dataset combo
# ----------------------------
for (nodes, edges), group in df.groupby(["nodes", "edges"]):
    pivot = group.pivot_table(
        index="workers",       # rows
        columns="chunk_size",  # columns
        values="time",
        aggfunc="mean",
    )
    pivot = pivot.sort_index(ascending=True)          # workers ascending top→bottom
    pivot = pivot[sorted(pivot.columns)]              # chunk sizes ascending left→right

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")

    fig.colorbar(im, ax=ax, label="Total time (s)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(r) for r in pivot.index])

    ax.set_xlabel("Local chunk size")
    ax.set_ylabel("Local workers")
    ax.set_title(
        f"BlockRank+Coloring: time (s)\n"
        f"Nodes={nodes:,}  Edges≈{edges:,}"
    )

    # Annotate every cell
    vmin, vmax = pivot.values.min(), pivot.values.max()
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                # White text on dark cells, black on light
                brightness = (val - vmin) / (vmax - vmin + 1e-9)
                color = "white" if brightness < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    plt.tight_layout()
    outpath = f"{OUT_DIR}/heatmap_workers_chunk_{nodes}n_{edges}e.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved {outpath}")
