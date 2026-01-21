#!/bin/bash
#SBATCH --job-name=hs2c_pipeline
#SBATCH --mem=60GB
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00
#SBATCH --mail-type=end
#SBATCH --output=slurm_logs/hs2c_%j.out
#SBATCH --error=slurm_logs/hs2c_%j.err


DATASET=""
# ["ogb_arxiv", "arxiv_2023", "citeseer", "instagram", "reddit", "products", "book_hete", "book_homo", "dblp_hete", "dblp_homo"]

echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "SLURM GPU: $CUDA_VISIBLE_DEVICES"
echo "Dataset: $DATASET"
echo "=========================================="
nvidia-smi

seeds=(0 1 2 3 4)

models=(
  "Llama-3.2-3b-Instruct"
  "Vicuna-7b-v1.5"
  "Llama-3.1-8b-Instruct"
)

for seed in "${seeds[@]}"; do
  for model in "${models[@]}"; do
      echo "=========================================="
      echo "Running: dataset=$DATASET, seed=$seed, model=$model"
      echo "Started at: $(date)"
      echo "=========================================="

      python main.py \
        --dataset "$DATASET" \
        --new_date 20250606 \
        --data_preprocess_seed 0 \
        --sample_neighbor_seed 0 \
        --model_name "$model" \
        --tree_height 2 \
        --save_target_graph \
        --save_original_cluster_data \
        --save_cluster \
        --save_reconstructed_cluster_data \
        --save_condensation_cluster_data \
        --run_classification \
        --llm_classification_answer_seed "$seed" \
        --llm_length 3980

      echo "Finished: dataset=$DATASET, seed=$seed, model=$model, strategy=$strategy at $(date)"
      echo ""
  done
done

echo "=========================================="
echo "All tasks finished at: $(date)"
echo "=========================================="
