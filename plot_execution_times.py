import re
import os
import matplotlib.pyplot as plt
import pandas as pd

LOG_DIR = "logs_8w_4_chunk"
OUT_DIR = "viz"

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Parse logs
# ----------------------------
data = []

pattern_dataset = re.compile(r"synth_([0-9]+(?:\.[0-9]+)?)([mk])_([0-9]+(?:\.[0-9]+)?)([mk])")
pattern_method = re.compile(r"^\[(.+?)\]\s*$")

pattern_total_time = re.compile(
    r"^(?:"
    r"Total\s+\w[\w+\s]*time\s*:\s*([0-9.]+)s?"
    r"|Total\s+time\s*:\s*([0-9.]+)s?"
    r"|Time\s+for\s+\w[\w\s]*?\s+([0-9.]+)\s*$"
    r")",
    re.IGNORECASE,
)


def parse_count(value, suffix):
    value = float(value)
    if suffix == "m":
        return int(value * 1_000_000)
    elif suffix == "k":
        return int(value * 1_000)
    return int(value)


def extract_nodes_edges(lines, filename):
    for line in lines:
        match = pattern_dataset.search(line)
        if match:
            nodes = parse_count(match.group(1), match.group(2))
            edges = parse_count(match.group(3), match.group(4))
            return nodes, edges
    match = pattern_dataset.search(filename)
    if match:
        nodes = parse_count(match.group(1), match.group(2))
        edges = parse_count(match.group(3), match.group(4))
        return nodes, edges
    return None, None


for file in sorted(os.listdir(LOG_DIR)):
    if not file.endswith(".log"):
        continue

    path = os.path.join(LOG_DIR, file)

    with open(path) as f:
        lines = [l.rstrip("\n") for l in f.readlines()]

    nodes, edges = extract_nodes_edges(lines, file)
    if nodes is None or edges is None:
        print(f"Skipping {file}: could not determine dataset size")
        continue

    current_method = None

    for line in lines:
        m_method = pattern_method.match(line.strip())
        if m_method:
            current_method = m_method.group(1)
            continue

        m_time = pattern_total_time.match(line.strip())
        if m_time and current_method:
            time_str = m_time.group(1) or m_time.group(2) or m_time.group(3)
            time = float(time_str)

            data.append({
                "nodes": nodes,
                "edges": edges,
                "method": current_method,
                "time": time,
            })
            current_method = None

df = pd.DataFrame(data)

if df.empty:
    raise SystemExit(f"No timing data found in {LOG_DIR}")

print(f"Parsed {len(df)} records from {LOG_DIR}")
print(df.to_string(index=False))

# Collapse duplicates
df = df.groupby(["nodes", "edges", "method"], as_index=False).agg(time=("time", "mean"))
df = df.sort_values(by=["edges", "nodes", "method"])

methods = df["method"].unique()
edge_values = sorted(df["edges"].unique())

# ----------------------------
# 1️⃣ Line plots (per edge count) — nodes on x-axis
# ----------------------------
for edges in edge_values:
    subset = df[df["edges"] == edges]

    plt.figure(figsize=(8, 5))

    for method in methods:
        mdata = subset[subset["method"] == method].sort_values("nodes")
        if mdata.empty:
            continue
        plt.plot(mdata["nodes"], mdata["time"], marker="o", label=method)

    plt.title(f"Execution Time vs Nodes (Edges ≈ {edges:,})")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Execution Time (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    outpath = f"{OUT_DIR}/line_edges_{edges}.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved {outpath}")

# ----------------------------
# 2️⃣ Heatmaps (per method) — nodes on y-axis, edges on x-axis
# ----------------------------
for method in methods:
    subset = df[df["method"] == method]

    pivot = subset.pivot_table(index="nodes", columns="edges", values="time", aggfunc="mean")

    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot, aspect="auto")

    plt.colorbar(im, label="Time (s)")
    plt.xticks(range(len(pivot.columns)), [f"{c:,}" for c in pivot.columns], rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), [f"{r:,}" for r in pivot.index])

    plt.xlabel("Edges")
    plt.ylabel("Nodes")
    plt.title(f"Heatmap: {method}")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    safe_name = re.sub(r"[^a-zA-Z0-9_+\-]", "_", method)
    outpath = f"{OUT_DIR}/heatmap_{safe_name}.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved {outpath}")