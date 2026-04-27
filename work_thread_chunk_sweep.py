import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOG_DIR = "logs_sweep_blockrank_coloring"

WORKER_COUNTS  = [4, 8, 12, 16, 24, 32]
CHUNK_SIZES    = [4, 32, 64, 128, 256]

# Regex to pull the total BlockRank+Coloring wall time from a log file
TIME_RE    = re.compile(r"Total BlockRank\+Coloring time:\s*([\d.]+)s")
HEADER_RE  = re.compile(
    r"Nodes:\s*([\d,]+).*?Edges:\s*([\d,]+)", re.DOTALL
)


def parse_log(filepath: str) -> float | None:
    with open(filepath) as f:
        content = f.read()
    m = TIME_RE.search(content)
    return float(m.group(1)) if m else None


def parse_graph_info(log_dir: str):
    """Grab Nodes/Edges from any available log for the subtitle."""
    for fname in os.listdir(log_dir):
        if not fname.endswith(".log"):
            continue
        with open(os.path.join(log_dir, fname)) as f:
            content = f.read()
        m = HEADER_RE.search(content)
        if m:
            nodes = int(m.group(1).replace(",", ""))
            edges = int(m.group(2).replace(",", ""))
            return nodes, edges
    return None, None


def build_time_matrix(log_dir: str) -> np.ndarray:
    """Rows = workers (ascending), Cols = chunk sizes (ascending)."""
    matrix = np.full((len(WORKER_COUNTS), len(CHUNK_SIZES)), np.nan)

    for i, w in enumerate(WORKER_COUNTS):
        for j, c in enumerate(CHUNK_SIZES):
            path = os.path.join(log_dir, f"w{w}_c{c}.log")
            if os.path.isfile(path):
                t = parse_log(path)
                if t is not None:
                    matrix[i, j] = t
            else:
                print(f"  [warn] missing: {path}")

    return matrix


def plot_heatmap(matrix: np.ndarray, nodes, edges, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        matrix,
        cmap="viridis",
        aspect="auto",
        origin="upper",          # row 0 = fewest workers (top)
    )

    # ── Axes ticks ──────────────────────────────────────────────────────────
    ax.set_xticks(range(len(CHUNK_SIZES)))
    ax.set_xticklabels(CHUNK_SIZES, fontsize=11)
    ax.set_yticks(range(len(WORKER_COUNTS)))
    ax.set_yticklabels(WORKER_COUNTS, fontsize=11)

    ax.set_xlabel("Local chunk size", fontsize=12)
    ax.set_ylabel("Local workers", fontsize=12)

    # ── Title ────────────────────────────────────────────────────────────────
    subtitle = ""
    if nodes is not None:
        subtitle = (
            f"Nodes={nodes:,}  Edges≈{round(edges / 1e6):,},000,000"
            if edges else f"Nodes={nodes:,}"
        )
    title = "BlockRank+Coloring: time (s)"
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, pad=10)

    # ── Colourbar ────────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Total time (s)", fontsize=11)
    cbar.ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # ── Cell annotations ─────────────────────────────────────────────────────
    vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
    mid = (vmin + vmax) / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            color = "white" if val < mid else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=11, fontweight="bold", color=color,
            )
    print("here")
    print(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def main():
    if not os.path.isdir(LOG_DIR):
        raise FileNotFoundError(f"Log directory '{LOG_DIR}' not found.")
    matrix = build_time_matrix(LOG_DIR)
    
    nodes, edges = parse_graph_info(LOG_DIR)

    out_path = os.path.join(LOG_DIR, "heatmap_blockrank_coloring.png")
    plot_heatmap(matrix, nodes, edges, out_path)


if __name__ == "__main__":
    main()