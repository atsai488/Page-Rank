#!/bin/bash
#SBATCH --job-name=synth_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --partition=debug
#SBATCH --output=slurm-%j.out

module purge
module load gcc python/3.11

set -euo pipefail

cd ~/ee451/Page-Rank

# pip install --user -r requirements.txt

export NUMBA_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

format_count() {
	local value="$1"
	awk -v n="$value" 'BEGIN {
		if (n >= 1000000) {
			if (n % 1000000 == 0) printf "%dm", n / 1000000;
			else printf "%.1fm", n / 1000000;
		} else if (n >= 1000) {
			if (n % 1000 == 0) printf "%dk", n / 1000;
			else printf "%.1fk", n / 1000;
		} else {
			printf "%d", n;
		}
	}'
}

node_counts=(500000 1000000 1500000 2000000)
edge_counts=(1000000 5000000 10000000 20000000 40000000)

subdomains_per_domain=10
pages_per_subdomain=100

for num_nodes in "${node_counts[@]}"; do
	n_domains=$((num_nodes / (subdomains_per_domain * pages_per_subdomain)))
	if (( num_nodes % (subdomains_per_domain * pages_per_subdomain) != 0 )); then
		echo "Skipping num_nodes=${num_nodes} because it is not divisible by 1000"
		continue
	fi

	for num_edges in "${edge_counts[@]}"; do
		nodes_tag="$(format_count "${num_nodes}")"
		edges_tag="$(format_count "${num_edges}")"
		dataset="data/synth_${nodes_tag}_${edges_tag}.txt"
		metadata="data/synth_${nodes_tag}_${edges_tag}_nodes.csv"

		echo "=================================================="
		echo "Generating dataset: nodes=${num_nodes}, edges=${num_edges}"
		echo "output=${dataset}"
		echo "metadata=${metadata}"
		echo "=================================================="

		python generate_synthetic_dataset.py \
			--edge-output "${dataset}" \
			--metadata-output "${metadata}" \
			--seed 42 \
			--n-domains "${n_domains}" \
			--min-subdomains "${subdomains_per_domain}" \
			--max-subdomains "${subdomains_per_domain}" \
			--min-pages-per-subdomain "${pages_per_subdomain}" \
			--max-pages-per-subdomain "${pages_per_subdomain}" \
			--target-edges "${num_edges}" \
			--degree-model fixed \
			--allow-self-loops \
			--workers 32
		done
done