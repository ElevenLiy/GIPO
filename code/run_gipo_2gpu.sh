#!/bin/bash
#SBATCH --job-name=gipo_2gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/gipo_2gpu_%j.out
#SBATCH --error=logs/gipo_2gpu_%j.err
#SBATCH --partition=fengl2

# ===========================================================================
# GIPO 2-GPU Experiment Runner (for 7B/8B models)
# ===========================================================================
# Uses model parallelism (device_map="auto"), NOT DDP/torchrun.
# GIPO rollouts are serial, so we split the model across 2 GPUs for memory.
#
# Usage:
#   sbatch run_gipo_2gpu.sh                              # Default: qwen2.5-7b, steps 3,4,5
#   sbatch run_gipo_2gpu.sh qwen2.5-7b 3,4,5             # SFT + GIPO + eval
#   sbatch run_gipo_2gpu.sh llama3.1-8b 4,5               # GIPO training + eval only
#   sbatch run_gipo_2gpu.sh qwen2.5-7b 5 sft              # Evaluate SFT only
#
# Checkpoint directories:
#   GIPO checkpoints: checkpoints/gipo_qwen7b/ or checkpoints/gipo_llama8b/
#   GIPO eval results: eval_gipo_results/
#   (SFT checkpoints are shared with other pipelines)
# ===========================================================================

# --- Parameters ---
MODEL=${1:-"qwen2.5-7b"}
STEPS=${2:-"3,4,5"}
STAGE=${3:-"grpo"}

# --- Project paths ---
ADAMACRO_DIR="/seu_share2/home/fenglei/230250004/Agent_Tool/tool-use/AdaMacro/code"
cd ${ADAMACRO_DIR}

mkdir -p logs

echo "============================================================"
echo "GIPO 2-GPU Experiment (Model Parallelism)"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       ${SLURM_NODELIST}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES}"
echo "Model:      ${MODEL}"
echo "Steps:      ${STEPS}"
echo "Stage:      ${STAGE}"
echo "Time:       $(date)"
echo "============================================================"

# --- Environment setup ---
source /seu_share/home/fenglei/230250004/anaconda3/etc/profile.d/conda.sh
conda activate tool

# --- Run GIPO 2-GPU pipeline ---
# NOTE: Using plain python, NOT torchrun. Model parallelism is handled
# inside the script via device_map="auto" and accelerate.
python scripts/run_pipeline_gipo_2gpu.py \
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
echo "GIPO 2-GPU experiment completed at $(date)"
echo "============================================================"
