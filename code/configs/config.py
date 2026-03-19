"""
AdaMacro: Centralized Configuration
====================================
All paths and hyperparameters are defined here.
To switch datasets or models, only modify this file.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================================
# Path Configuration — MODIFY THIS SECTION FOR DIFFERENT DATASETS/ENVIRONMENTS
# ============================================================================

# Project root on your server
PROJECT_ROOT = "/seu_share2/home/fenglei/230250004/Agent_Tool/tool-use/tool-use"

# Dataset name (used for sub-directory naming)
DATASET_NAME = "TOOLATHLON"

# --- Input data paths ---
# Raw trajectory files (JSONL directory)
TRAJECTORIES_DIR = os.path.join(PROJECT_ROOT, "Toolathlon-Trajectories-merge")

# All tools definition JSON
ALL_TOOLS_PATH = os.path.join(PROJECT_ROOT, "json_file", "all_tools_v2.json")

# MCP RL graph (tool transitions)
MCP_GRAPH_PATH = os.path.join(PROJECT_ROOT, "json_file", "mcp_rl_graph_v2.json")

# All messages (full trajectory JSON)
ALL_MESSAGES_PATH = os.path.join(PROJECT_ROOT, "json_file", "all_messages.json")

# RL dataset (processed trajectories with tool sequences)
RL_DATASET_PATH = os.path.join(PROJECT_ROOT, "GRPO-ACO", "data", "rl_dataset_llm_v3.json")

# Tool simulator database
TOOL_SIMULATOR_DB_PATH = os.path.join(PROJECT_ROOT, "GRPO-ACO", "data", "tool_simulator_database.json")

# --- Output paths ---
# AdaMacro output directory
ADAMACRO_OUTPUT_DIR = os.path.join("/seu_share2/home/fenglei/230250004/Agent_Tool/tool-use/AdaMacro", "outputs", DATASET_NAME)

# Skill library output
SKILL_LIBRARY_PATH = os.path.join(ADAMACRO_OUTPUT_DIR, "skill_library.json")

# Augmented tool list (atoms + skills)
AUGMENTED_TOOLS_PATH = os.path.join(ADAMACRO_OUTPUT_DIR, "augmented_tools.json")

# SFT training data
SFT_DATA_PATH = os.path.join(ADAMACRO_OUTPUT_DIR, "sft_data.json")

# GRPO training data
GRPO_DATA_PATH = os.path.join(ADAMACRO_OUTPUT_DIR, "grpo_data.json")

# Evaluation results
EVAL_RESULTS_DIR = os.path.join(ADAMACRO_OUTPUT_DIR, "eval_results")

# LoRA checkpoint directory
CHECKPOINT_DIR = os.path.join(ADAMACRO_OUTPUT_DIR, "checkpoints")

# --- Model paths ---
MODEL_PATHS = {
    "qwen2.5-1.5b": "/seu_share2/home/fenglei/sharedata/Qwen2.5-1.5B-Instruct",
    "qwen2.5-7b": "/seu_share2/home/fenglei/sharedata/Qwen2.5-7B-Instruct",
    "llama3.1-8b": "/seu_share2/home/fenglei/sharedata/Llama-3.1-8B-Instruct",
    "llama3.2-3b": "/seu_share2/home/fenglei/sharedata/Llama-3.2-3B-Instruct",
}

# Default model
DEFAULT_MODEL = "qwen2.5-7b"


# ============================================================================
# Hyperparameter Configuration
# ============================================================================

@dataclass
class BPEConfig:
    """BPE macro mining hyperparameters."""
    # Maximum number of merge rounds (vocabulary budget K)
    max_merges: int = 50
    # Minimum frequency threshold for a bigram to be merged
    min_freq: int = 3
    # Maximum macro length (number of atomic tools)
    max_macro_len: int = 6
    # Minimum macro length
    min_macro_len: int = 2
    # Only use successful trajectories
    success_only: bool = True
    # Pruning: minimum usage ratio after merging
    min_usage_ratio: float = 0.01


@dataclass
class SkillConfig:
    """Skill instantiation configuration."""
    # Available skill templates
    templates: List[str] = field(default_factory=lambda: [
        "sequential",   # Pure sequential execution
        "select",       # Previous step outputs candidates, next step selects
        "conditional",  # Conditional branching based on output
    ])
    # select_strategy options exposed to LLM
    select_strategies: List[str] = field(default_factory=lambda: [
        "rank-0",    # Select first/top result
        "rank-1",    # Select second result
        "random",    # Random selection
        "filter",    # Filter by condition
    ])
    # Whether to enable trace injection
    enable_trace: bool = True
    # Whether to enable soft interrupt
    enable_soft_interrupt: bool = True


@dataclass
class SFTConfig:
    """SFT training configuration."""
    # Number of training epochs (2-3 for small data, 1 for >10k examples)
    num_epochs: int = 3
    # Learning rate
    learning_rate: float = 1e-5
    # Batch size per device
    per_device_batch_size: int = 2
    # Gradient accumulation steps
    gradient_accumulation_steps: int = 8
    # Maximum sequence length
    max_seq_length: int = 4096
    # Warmup ratio
    warmup_ratio: float = 0.05
    # LoRA rank
    lora_rank: int = 64
    # LoRA alpha
    lora_alpha: int = 128
    # LoRA dropout
    lora_dropout: float = 0.1
    # LoRA target modules
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    # Weight decay
    weight_decay: float = 0.01
    # Save steps
    save_steps: int = 200
    # Logging steps
    logging_steps: int = 10


@dataclass
class GRPOConfig:
    """GRPO training configuration."""
    # Number of training epochs
    num_epochs: int = 3
    # Learning rate
    learning_rate: float = 5e-6
    # Group sampling size (G in GRPO)
    group_size: int = 4
    # KL penalty coefficient (beta)
    kl_coeff: float = 0.05
    # Maximum generation length
    max_gen_length: int = 512
    # Batch size per device
    per_device_batch_size: int = 1
    # Gradient accumulation steps
    gradient_accumulation_steps: int = 8
    # Maximum sequence length
    max_seq_length: int = 4096
    # LoRA rank (same as SFT)
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    # Reward weights
    lambda_skill: float = 0.3  # λ in R(τ) = R_task(τ) + λ·R_skill(τ)
    r_complete: float = 1.0    # Reward for complete skill success
    r_pass: float = 0.2        # Per-step reward for passed steps before interrupt
    r_fail: float = -0.5       # Penalty for the failed step
    # Temperature for sampling
    temperature: float = 0.7
    # Dynamic forced steps range
    min_forced_lower: int = 3     # minimum of min_forced_steps
    min_forced_upper: int = 6     # maximum of min_forced_steps
    # R_efficiency gating threshold
    r_eff_gate: float = 0.4       # minimum r_task to grant efficiency bonus
    # Under-exploration penalty
    under_explore_penalty: float = 0.05
    under_explore_threshold: int = 3
    # Save steps
    save_steps: int = 100
    # Logging steps
    logging_steps: int = 1
    # --- GIPO: Granularity-Imagination Policy Optimization ---
    # Per-step imagination reward scale (max absolute value per step)
    gipo_step_reward_scale: float = 0.15
    # Max per-step imagination reward (clip)
    gipo_step_reward_cap: float = 0.1
    # Max total imagination reward per rollout (clip)
    gipo_total_reward_cap: float = 0.3


@dataclass
class GIPOAPIConfig(GRPOConfig):
    """GIPO-API: use external LLM to simulate tool outputs instead of tool_simulator_database."""
    api_key: str = "sk-jgHe5j4sJg4C0JVeJN4e4r9aI8znVW60IwdM3EqT8WeD7dRX"
    api_base_url: str = "http://172.22.2.242:3010/v1"
    api_model: str = "qwen3.5-397b-a17b"
    api_timeout: int = 60
    api_max_retries: int = 3
    api_temperature: float = 0.7


@dataclass
class GIPO3BConfig(GRPOConfig):
    """GIPO config tuned for 3B models on single GPU.

    LLaMA 3.2-3B is between 1.5B and 7B in size.
    Uses slightly lower lr and LoRA rank than 1.5B default, but no gradient
    checkpointing (fits on a single GPU comfortably).
    """
    learning_rate: float = 3e-6
    lora_rank: int = 48
    lora_alpha: int = 96
    lora_dropout: float = 0.05
    max_seq_length: int = 4096
    max_gen_length: int = 512
    gradient_accumulation_steps: int = 8
    save_steps: int = 100
    gradient_checkpointing: bool = False


@dataclass
class GIPO7BConfig(GRPOConfig):
    """GIPO config tuned for 7B/8B models on 2×GPU (model parallelism)."""
    # Lower lr for larger models — 7B is more sensitive to large updates
    learning_rate: float = 2e-6
    # Smaller LoRA rank to save memory (7B has more params per layer)
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    # Shorter sequences to fit in memory (7B KV cache is 4x larger)
    max_seq_length: int = 3072
    max_gen_length: int = 384
    # Fewer grad accum steps — each step is slower, so fewer prompts per update
    gradient_accumulation_steps: int = 4
    # Save less frequently (each step takes longer)
    save_steps: int = 50
    # Enable gradient checkpointing (set in training script, flag here for clarity)
    gradient_checkpointing: bool = True


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    # Maximum turns for agent execution
    max_turns: int = 30
    # Maximum atomic tool calls (budget)
    max_atomic_calls: int = 50
    # Temperature for inference
    temperature: float = 0.0
    # Top-p for inference
    top_p: float = 0.95
    # Batch size for inference
    batch_size: int = 1
    # Whether to use continuation mechanism (match GRPO training)
    enable_continuation: bool = True


def get_model_path(model_name: str) -> str:
    """Get model path by name."""
    if model_name in MODEL_PATHS:
        return MODEL_PATHS[model_name]
    raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_PATHS.keys())}")


def ensure_dirs():
    """Create all necessary output directories."""
    dirs = [
        ADAMACRO_OUTPUT_DIR,
        EVAL_RESULTS_DIR,
        CHECKPOINT_DIR,
        os.path.join(CHECKPOINT_DIR, "sft"),
        os.path.join(CHECKPOINT_DIR, "grpo"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def print_config():
    """Print current configuration for logging."""
    print("=" * 70)
    print("AdaMacro Configuration")
    print("=" * 70)
    print(f"  Dataset:          {DATASET_NAME}")
    print(f"  Project root:     {PROJECT_ROOT}")
    print(f"  Trajectories:     {TRAJECTORIES_DIR}")
    print(f"  All tools:        {ALL_TOOLS_PATH}")
    print(f"  RL dataset:       {RL_DATASET_PATH}")
    print(f"  Tool simulator:   {TOOL_SIMULATOR_DB_PATH}")
    print(f"  Output dir:       {ADAMACRO_OUTPUT_DIR}")
    print(f"  Default model:    {DEFAULT_MODEL}")
    print("=" * 70)
