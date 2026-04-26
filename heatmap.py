import re
import os
import matplotlib.pyplot as plt
import pandas as pd

LOG_DIR = "logs_4w_64_chunk"
OUT_DIR = "viz"

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Parse logs
# ----------------------------
data = []

pattern_dataset = re.compile(r"synth_(\d+)([mk])_(\d+)([mk])")
pattern_method = re.compile(r"\[(.*?)\]")
pattern_time = re.compile(r"Total.*time:\s*([0-9.]+)")

def parse_count(value, suffix):
    value = int(value)
    if suffix == "m":
        return value * 1_000_000
    elif suffix == "k":
        return value * 1_000
    return value

for file in os.listdir(LOG_DIR):
    if not file.endswith(".log"):
        continue

    path = os.path.join(LOG_DIR, file)

    with open(path) as f:
        lines = f.readlines()

    dataset_line = next(l for l in lines if "Dataset:" in l)
    match = pattern_dataset.search(dataset_line)

    nodes = parse_count(match.group(1), match.group(2))
    edges = parse_count(match.group(3), match.group(4))

    current_method = None

    for line in lines:
        m_method = pattern_method.search(line)
        if m_method:
            current_method = m_method.group(1)

        m_time = pattern_time.search(line)
        if m_time and current_method:
            time = float(m_time.group(1))

            data.append({
                "nodes": nodes,
                "edges": edges,
                "method": current_method,
                "time": time
            })

df = pd.DataFrame(data)

# ----------------------------
# 1️⃣ Line plots (per node count)
# ----------------------------
methods = df["method"].unique()
node_values = sorted(df["nodes"].unique())

for nodes in node_values:
    subset = df[df["nodes"] == nodes]

    plt.figure(figsize=(8, 5))

    for method in methods:
        mdata = subset[subset["method"] == method].sort_values("edges")

        plt.plot(
            mdata["edges"],
            mdata["time"],
            marker="o",
            label=method
        )

    plt.title(f"Execution Time vs Edges (Nodes = {nodes:,})")
    plt.xlabel("Number of Edges")
    plt.ylabel("Execution Time (s)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/line_nodes_{nodes}.png", dpi=300)
    plt.close()

# ----------------------------
# 2️⃣ Heatmaps (per method)
# ----------------------------
for method in methods:
    subset = df[df["method"] == method]

    pivot = subset.pivot(index="nodes", columns="edges", values="time")

    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot, aspect="auto")

    plt.colorbar(im, label="Time (s)")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)

    plt.xlabel("Edges")
    plt.ylabel("Nodes")
    plt.title(f"Heatmap: {method}")

    # annotate values
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/heatmap_{method}.png", dpi=300)
    plt.close()