#!/bin/bash
#SBATCH --job-name=full_run_web
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --partition=main
#SBATCH --output=slurm-%j.out

module purge
module load gcc python/3.11

set -euo pipefail

export NUMBA_NUM_THREADS=32
export OMP_NUM_THREADS=32
export SLURM_CPUS_PER_TASK=32
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python plot_execution_times.py