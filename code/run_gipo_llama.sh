#!/bin/bash
#SBATCH --job-name=gipo_llama3b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/gipo_llama3b_%j.out
#SBATCH --error=logs/gipo_llama3b_%j.err
#SBATCH --partition=fengl2

# ===========================================================================
# GIPO Single-GPU Experiment for LLaMA 3.2-3B
# ===========================================================================
# Uses GIPO3BConfig (lr=3e-6, lora_rank=48, no gradient checkpointing).
# LLaMA 3.2-3B fits comfortably on a single GPU.
#
# Checkpoint directory:  checkpoints/gipo_llama32_3b/
# Eval directory:        eval_gipo_llama32_3b_results/
#
# Usage:
#   sbatch run_gipo_llama.sh                              # Default: SFT + GIPO + eval
#   sbatch run_gipo_llama.sh llama3.2-3b 4,5              # GIPO training + eval only (if SFT already done)
#   sbatch run_gipo_llama.sh llama3.2-3b 1,2,3,4,5        # Full pipeline from scratch
#   sbatch run_gipo_llama.sh llama3.2-3b 5 sft            # Evaluate SFT only
#
# NOTE: Steps 1-2 (BPE mining + Skill instantiation) are model-agnostic and
#       shared across all pipelines. Only run them if not done before.
#       Step 3 (SFT) checkpoint is shared at checkpoints/sft/llama3.2-3b/.
#       Only re-run Step 3 if SFT hasn't been trained for this model yet.
# ===========================================================================

# --- Parameters ---
MODEL=${1:-"llama3.2-3b"}
STEPS=${2:-"3,4,5"}
STAGE=${3:-"grpo"}

# --- Project paths ---
ADAMACRO_DIR="/seu_share2/home/fenglei/230250004/Agent_Tool/tool-use/AdaMacro/code"
cd ${ADAMACRO_DIR}

mkdir -p logs

echo "============================================================"
echo "GIPO Experiment: LLaMA 3.2-3B (Single GPU)"
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

# --- Run GIPO pipeline ---
python scripts/run_pipeline_gipo_llama.py \
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
echo "GIPO LLaMA 3.2-3B experiment completed at $(date)"
echo "============================================================"
