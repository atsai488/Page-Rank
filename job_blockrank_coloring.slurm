#!/bin/bash
#SBATCH --job-name=blockrank_coloring
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --partition=main
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

dataset=data/synth_2m_40m.txt
metadata=data/synth_2m_40m_nodes.csv
epsilon=1e-9

local_workers_list=(4 8 12 16 24 32)
local_chunk_sizes=(32 64 128 256)

for local_workers in "${local_workers_list[@]}"; do
	for local_chunk_size in "${local_chunk_sizes[@]}"; do
		echo "=================================================="
		echo "Running BlockRank+Coloring sweep"
		echo "local-workers=${local_workers}"
		echo "local-chunk-size=${local_chunk_size}"
		echo "=================================================="

		python run_blockrank_coloring.py \
			--dataset "${dataset}" \
			--metadata "${metadata}" \
			--epsilon "${epsilon}" \
			--local-workers "${local_workers}" \
			--local-chunk-size "${local_chunk_size}"
	done
done