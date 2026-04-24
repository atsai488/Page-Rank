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

cd ~/Page-Rank

pip install --user -r requirements.txt

export NUMBA_NUM_THREADS=32
python run_blockrank_coloring.py
