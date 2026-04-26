export NUMBA_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

epsilon=1e-9
num_blocks=100
local_workers=4
local_chunk_size=64

mkdir -p logs_4w_64_chunk

for dataset in data/synth_*.txt; do

    metadata="${dataset%.txt}_nodes.csv"
    base=$(basename "${dataset}" .txt)

    log_file="logs_4w_64_chunk/${base}.log"

    echo "==================================================" | tee "${log_file}"
    echo "Dataset: ${dataset}" | tee -a "${log_file}"
    echo "Metadata: ${metadata}" | tee -a "${log_file}"
    echo "workers=${local_workers}, chunk=${local_chunk_size}, blocks=${num_blocks}" | tee -a "${log_file}"
    echo "==================================================" | tee -a "${log_file}"

    # -------------------------------
    # Coloring
    # -------------------------------
    echo "[Coloring]" | tee -a "${log_file}"
    python run_coloring.py \
        --dataset "${dataset}" \
        --epsilon "${epsilon}" \
        2>&1 | tee -a "${log_file}"
    
    # -------------------------------
    # BlockRank + Coloring
    # -------------------------------
    echo "[BlockRank+Coloring]" | tee -a "${log_file}"
    python run_blockrank_coloring.py \
        --dataset "${dataset}" \
        --metadata "${metadata}" \
        --num-blocks "${num_blocks}" \
        --epsilon "${epsilon}" \
        --local-workers "${local_workers}" \
        --local-chunk-size "${local_chunk_size}" \
        2>&1 | tee -a "${log_file}"

    # -------------------------------
    # BlockRank
    # -------------------------------
    echo "[BlockRank]" | tee -a "${log_file}"
    python run_blockrank.py \
        --dataset "${dataset}" \
        --metadata "${metadata}" \
        --num-blocks "${num_blocks}" \
        --epsilon "${epsilon}" \
        2>&1 | tee -a "${log_file}"


    # -------------------------------
    # Basic PageRank
    # -------------------------------
    echo "[Basic]" | tee -a "${log_file}"
    python run_basic.py \
        --dataset "${dataset}" \
        --epsilon "${epsilon}" \
        2>&1 | tee -a "${log_file}"

    echo "" | tee -a "${log_file}"
done