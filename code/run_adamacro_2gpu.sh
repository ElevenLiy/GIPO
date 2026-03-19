#!/bin/bash
#SBATCH --job-name=adamacro_2gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/adamacro_2gpu_%j.out
#SBATCH --error=logs/adamacro_2gpu_%j.err
#SBATCH --partition=fengl2

# ===========================================================================
# AdaMacro: 2-GPU Runner (for 7B/8B models)
# ===========================================================================
# Usage:
#   sbatch run_adamacro_2gpu.sh qwen2.5-7b 3,4,5
#   sbatch run_adamacro_2gpu.sh llama3.1-8b 3,4,5
# ===========================================================================

MODEL=${1:-"qwen2.5-7b"}
STEPS=${2:-"3,4,5"}
STAGE=${3:-"grpo"}

ADAMACRO_DIR="/seu_share2/home/fenglei/230250004/Agent_Tool/tool-use/AdaMacro/code"
cd ${ADAMACRO_DIR}
mkdir -p logs

echo "============================================================"
echo "AdaMacro 2-GPU Experiment"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "GPUs:       2"
echo "Model:      ${MODEL}"
echo "Steps:      ${STEPS}"
echo "Time:       $(date)"
echo "============================================================"

source /seu_share/home/fenglei/230250004/anaconda3/etc/profile.d/conda.sh
conda activate tool


# Use torchrun for multi-GPU DDP
torchrun --nproc_per_node=2 --master_port=$(shuf -i 29500-29999 -n 1) \
    scripts/run_pipeline.py \
    --model ${MODEL} \
    --steps ${STEPS} \
    --stage ${STAGE} \
    --max-merges 50 \
    --min-freq 3 \
    --max-macro-len 6 \
    --max-episodes 100

echo "============================================================"
echo "Completed at $(date)"
echo "============================================================"