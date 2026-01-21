#!/bin/bash
#SBATCH --job-name=gl_hs2c
#SBATCH --mem=60GB
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00
#SBATCH --mail-type=end
#SBATCH --output=hs2c_all_%j.out
#SBATCH --error=hs2c_all_%j.err


echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $GPU_ID"
echo "=========================================="
nvidia-smi

SAVE_DIR="./hs2c_new_results"
LOG_DIR="./hs2c_new_logs"
mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

seeds=(0 1 2 3 4)

models=(
  "Llama-3.2-3b-Instruct"
  "Vicuna-7b-v1.5"
  "Llama-3.1-8b-Instruct"
)

datasets=(
  "Graph-Twitter"
  "Graph-SST5"
  "Graph-SST2"
  "ClinTox"
  "SIDER"
  "MUV"
  "BBBP"
)

for seed in "${seeds[@]}"; do
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do

        log_prefix="${LOG_DIR}/${dataset}_${model}_seed${seed}"
        out_file="${log_prefix}.out"
        err_file="${log_prefix}.err"

        {
        echo "=========================================="
        echo "Running: seed=$seed, model=$model, dataset=$dataset"
        echo "Started at: $(date)"
        echo "=========================================="
        } > "$out_file"

        # 清空 err
        : > "$err_file"

        python main.py \
          --dataset "$dataset" \
          --model_name "$model" \
          --data_seed "$seed" \
          --max_length 4000 \
          --use_knn_enhancement \
          --ori_text_in_classification \
          >>"$out_file" 2>>"$err_file"

        echo "Finished at: $(date)" >> "$out_file"
        echo "" >> "$out_file"

        echo "[DONE] dataset=${dataset}, model=${model}, seed=${seed}"
    done
  done
done

echo "=========================================="
echo "All tasks finished at: $(date)"
echo "=========================================="
