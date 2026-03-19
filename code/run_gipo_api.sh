#!/bin/bash
#SBATCH --job-name=gipo_api
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/gipo_api_%j.out
#SBATCH --error=logs/gipo_api_%j.err
#SBATCH --partition=fengl2

# ===========================================================================
# GIPO-API Experiment Runner (API-based tool simulation)
# ===========================================================================
# Usage:
#   sbatch run_gipo_api.sh                                # Default: qwen2.5-7b, all steps
#   sbatch run_gipo_api.sh qwen2.5-1.5b 1,2,3,4,5        # Specify model and steps
#   sbatch run_gipo_api.sh qwen2.5-1.5b 4,5               # Only GIPO-API training + eval
#   sbatch run_gipo_api.sh qwen2.5-7b 5 sft               # Evaluate SFT only
#
# Checkpoint directories:
#   GIPO-API checkpoints: checkpoints/gipo_api_qwen1.5b/
#   GIPO-API eval results: eval_gipo_api_results/
#   (SFT checkpoints are shared with run_adamacro.sh)
#
# Note: This variant uses an external LLM API (DashScope/Qwen) to simulate
# tool outputs instead of the static tool_simulator_database. Requires
# network access to the API endpoint.
# ===========================================================================

# --- Parameters ---
MODEL=${1:-"qwen2.5-7b"}
STEPS=${2:-"1,2,3,4,5"}
STAGE=${3:-"grpo"}

# --- Project paths ---
ADAMACRO_DIR="/seu_share2/home/fenglei/230250004/Agent_Tool/tool-use/AdaMacro/code"
cd ${ADAMACRO_DIR}

mkdir -p logs

echo "============================================================"
echo "GIPO-API Experiment (API-based tool simulation)"
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
source /seu_share/home/fenglei/230250004/anaconda3/etc/profile.d/conda.sh
conda activate tool

# --- Run GIPO-API pipeline ---
python scripts/run_pipeline_gipo_api.py \
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
echo "GIPO-API experiment completed at $(date)"
echo "============================================================"
