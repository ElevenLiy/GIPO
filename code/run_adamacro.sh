#!/bin/bash
#SBATCH --job-name=adamacro
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/adamacro_%j.out
#SBATCH --error=logs/adamacro_%j.err 
#SBATCH --partition=fengl2

# ===========================================================================
# AdaMacro Experiment Runner
# ===========================================================================
# Usage:
#   sbatch run_adamacro.sh                          # Default: qwen2.5-7b, all steps
#   sbatch run_adamacro.sh qwen2.5-1.5b 1,2,3,4,5  # Specify model and steps
#   sbatch run_adamacro.sh llama3.2-3b 3,4,5        # Only training + eval
#   sbatch run_adamacro.sh qwen2.5-7b 5 sft         # Evaluate SFT only
# ===========================================================================

# --- Parameters (modify via command line or here) ---
MODEL=${1:-"qwen2.5-7b"}
STEPS=${2:-"1,2,3,4,5"}
STAGE=${3:-"grpo"}

# --- Project paths ---
# !! MODIFY THIS to your AdaMacro project directory !!
ADAMACRO_DIR="/seu_share2/home/fenglei/230250004/Agent_Tool/tool-use/AdaMacro/code"
cd ${ADAMACRO_DIR}

# Create log directory
mkdir -p logs

echo "============================================================"
echo "AdaMacro Experiment"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES}"
echo "Model:      ${MODEL}"
echo "Steps:      ${STEPS}"
echo "Stage:      ${STAGE}"
echo "Time:       $(date)"
echo "============================================================"

# --- Environment setup ---
# Activate your conda environment (modify as needed)
source /seu_share/home/fenglei/230250004/anaconda3/etc/profile.d/conda.sh
conda activate tool
# OR
# module load cuda/12.1
# module load python/3.10

# Install dependencies if needed (first run only)
# pip install torch transformers peft trl datasets accelerate --break-system-packages

# --- Run pipeline ---
python scripts/run_pipeline.py \
    --model ${MODEL} \
    --steps ${STEPS} \
    --stage ${STAGE} \
    --max-merges 50 \
    --min-freq 3 \
    --max-macro-len 6 \
    --max-episodes 100 \
    --max-turns 30 \
    --max-atomic-calls 50

echo "============================================================"
echo "Experiment completed at $(date)"
echo "============================================================"
