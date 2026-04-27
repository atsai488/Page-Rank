import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = "logs_real"

DATASETS = ["web-BerkStan", "web-Google"]

METHODS = ["Coloring", "BlockRank+Coloring", "BlockRank", "Basic"]

TIME_PATTERNS = {
    "Coloring":           re.compile(r"\[Coloring\].*?Total time:\s*([\d.]+)s",                           re.DOTALL),
    "BlockRank+Coloring": re.compile(r"\[BlockRank\+Coloring\].*?Total BlockRank\+Coloring time:\s*([\d.]+)s", re.DOTALL),
    "BlockRank":          re.compile(r"\[BlockRank\].*?Total BlockRank time:\s*([\d.]+)s",                re.DOTALL),
    "Basic":              re.compile(r"\[Basic\].*?Time for power iteration\s+([\d.]+)",                  re.DOTALL),
}

COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]


def parse_log(dataset: str) -> dict[str, float]:
    path = os.path.join(LOG_DIR, f"{dataset}.log")
    if not os.path.isfile(path):
        print(f"[warn] missing log: {path}")
        return {}
    with open(path) as f:
        content = f.read()
    times = {}
    for method, pattern in TIME_PATTERNS.items():
        m = pattern.search(content)
        if m:
            times[method] = float(m.group(1))
        else:
            print(f"[warn] no time found for '{method}' in {path}")
    return times


def plot(data: dict[str, dict[str, float]], out_path: str):
    n_datasets = len(DATASETS)
    n_methods  = len(METHODS)
    bar_width  = 0.18
    group_gap  = 0.8          # centre-to-centre distance between dataset groups
    offsets    = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_width

    fig, ax = plt.subplots(figsize=(10, 6))

    group_centres = np.arange(n_datasets) * group_gap

    for m_idx, (method, color) in enumerate(zip(METHODS, COLORS)):
        heights = [data[ds].get(method, 0) for ds in DATASETS]
        x_pos   = group_centres + offsets[m_idx]
        bars = ax.bar(x_pos, heights, width=bar_width, label=method,
                      color=color, edgecolor="white", linewidth=0.6)

        for bar, h in zip(bars, heights):
            if h:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01 * max(v for d in data.values() for v in d.values()),
                    f"{h:.2f}s",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

    ax.set_xticks(group_centres)
    ax.set_xticklabels(DATASETS, fontsize=12)
    ax.set_ylabel("Execution Time (s)", fontsize=12)
    ax.set_title("PageRank Method Comparison\nweb-BerkStan vs web-Google", fontsize=13, fontweight="bold", pad=12)
    ax.legend(title="Method", fontsize=10, title_fontsize=10, framealpha=0.7)
    ax.set_ylim(0, 3)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def main():
    data = {ds: parse_log(ds) for ds in DATASETS}
    out_path = os.path.join(LOG_DIR, "comparison_berkstan_google.png")
    plot(data, out_path)


if __name__ == "__main__":
    main()