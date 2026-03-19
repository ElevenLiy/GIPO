"""
AdaMacro: Main Pipeline Runner
================================

Orchestrates the full AdaMacro pipeline:
  Step 1: BPE Macro Mining
  Step 2: Skill Instantiation
  Step 3: SFT Training
  Step 4: GRPO Training
  Step 5: Evaluation

Usage:
  python run_pipeline.py --model qwen2.5-7b --steps 1,2,3,4,5
  python run_pipeline.py --model llama3.2-3b --steps 1,2  # Only mining + instantiation
  python run_pipeline.py --model qwen2.5-7b --steps 5 --stage sft  # Evaluate SFT only
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    ensure_dirs, print_config,
    RL_DATASET_PATH, ALL_TOOLS_PATH, SKILL_LIBRARY_PATH,
    AUGMENTED_TOOLS_PATH, SFT_DATA_PATH, GRPO_DATA_PATH,
    TOOL_SIMULATOR_DB_PATH, CHECKPOINT_DIR, EVAL_RESULTS_DIR,
    ADAMACRO_OUTPUT_DIR,
    BPEConfig, SkillConfig, SFTConfig, GRPOConfig, EvalConfig,
    get_model_path, DEFAULT_MODEL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(
            Path(__file__).resolve().parent.parent, "logs", "pipeline.log"
        ), mode="a"),
    ]
)
logger = logging.getLogger(__name__)


def run_step1(args):
    """BPE Macro Mining."""
    from step1_bpe_mining import load_successful_sequences, BPEMacroMiner, load_tool_schemas
    import json

    bpe_config = BPEConfig(
        max_merges=args.max_merges,
        min_freq=args.min_freq,
        max_macro_len=args.max_macro_len,
    )

    sequences = load_successful_sequences(RL_DATASET_PATH, bpe_config.success_only)
    if not sequences:
        logger.error("No sequences found!")
        return

    miner = BPEMacroMiner(bpe_config)
    macros = miner.mine(sequences)

    # Enrich with tool schemas
    tool_schemas = load_tool_schemas(ALL_TOOLS_PATH)
    for mid, macro in macros.items():
        enriched = []
        for tname in macro["tool_names"]:
            if tname in tool_schemas:
                enriched.append({
                    "name": tname,
                    "description": tool_schemas[tname]["description"][:200],
                    "params": tool_schemas[tname]["actual_keys"],
                })
            else:
                enriched.append({"name": tname, "description": "", "params": []})
        macro["tool_details"] = enriched

    output = {
        "meta": {
            "algorithm": "BPE", "max_merges": bpe_config.max_merges,
            "min_freq": bpe_config.min_freq, "num_macros": len(macros),
            "num_sequences": len(sequences),
        },
        "macros": macros,
        "merge_history": miner.merge_history,
    }
    Path(SKILL_LIBRARY_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(SKILL_LIBRARY_PATH, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Step 1 done: {len(macros)} macros -> {SKILL_LIBRARY_PATH}")


def run_step2(args):
    """Skill Instantiation."""
    from step2_skill_instantiation import build_augmented_tools
    import json

    tool_schemas = {}
    with open(ALL_TOOLS_PATH, "r") as f:
        tools = json.load(f)
    for t in tools:
        tool_schemas[t.get("name", "")] = t

    augmented = build_augmented_tools(
        ALL_TOOLS_PATH, SKILL_LIBRARY_PATH, tool_schemas, SkillConfig()
    )
    Path(AUGMENTED_TOOLS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(AUGMENTED_TOOLS_PATH, "w") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)
    logger.info(f"Step 2 done: {len(augmented)} tools -> {AUGMENTED_TOOLS_PATH}")


def run_step3(args):
    """SFT Training."""
    from step3_sft_training import generate_sft_data, train_sft

    sft_config = SFTConfig()
    if args.epochs: sft_config.num_epochs = args.epochs
    if args.lr: sft_config.learning_rate = args.lr
    if args.batch_size: sft_config.per_device_batch_size = args.batch_size
    if args.lora_rank: sft_config.lora_rank = args.lora_rank

    sft_dir = os.path.join(CHECKPOINT_DIR, "sft", args.model)
    os.makedirs(sft_dir, exist_ok=True)

    logger.info("[Step 3.1] Generating SFT data...")
    generate_sft_data(RL_DATASET_PATH, AUGMENTED_TOOLS_PATH, SKILL_LIBRARY_PATH, SFT_DATA_PATH)

    logger.info(f"[Step 3.2] Training SFT with {args.model}...")
    train_sft(args.model, SFT_DATA_PATH, sft_dir, sft_config)
    logger.info(f"Step 3 done: SFT checkpoint -> {sft_dir}")


def run_step4(args):
    """GRPO Training (Online mode)."""
    from step4_grpo_training import generate_grpo_rollouts, train_grpo

    grpo_config = GRPOConfig()
    if args.epochs: grpo_config.num_epochs = args.epochs
    if args.lr: grpo_config.learning_rate = args.lr
    if args.group_size: grpo_config.group_size = args.group_size

    sft_dir = os.path.join(CHECKPOINT_DIR, "sft", args.model)
    grpo_dir = os.path.join(CHECKPOINT_DIR, "grpo", args.model)
    os.makedirs(grpo_dir, exist_ok=True)

    # Online GRPO: rollouts generated on-the-fly, no offline generation needed
    generate_grpo_rollouts()

    logger.info(f"[Step 4] Online GRPO training with {args.model}...")
    train_grpo(args.model, sft_dir, GRPO_DATA_PATH, grpo_dir, grpo_config)
    logger.info(f"Step 4 done: GRPO checkpoint -> {grpo_dir}")


def run_step5(args):
    """Evaluation."""
    from step5_evaluation import evaluate

    eval_config = EvalConfig(
        max_turns=args.max_turns,
        max_atomic_calls=args.max_atomic_calls,
    )

    # Determine checkpoint
    if args.stage == "base":
        lora_path = None
    elif args.stage == "sft":
        lora_path = os.path.join(CHECKPOINT_DIR, "sft", args.model)
    else:
        lora_path = os.path.join(CHECKPOINT_DIR, "grpo", args.model)

    output_path = os.path.join(
        EVAL_RESULTS_DIR,
        f"eval_{args.model}_{args.stage}_{int(time.time())}.json"
    )
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    logger.info(f"[Step 5] Evaluating {args.model} ({args.stage})...")
    evaluate(
        model_name=args.model, lora_path=lora_path,
        rl_dataset_path=RL_DATASET_PATH, augmented_tools_path=AUGMENTED_TOOLS_PATH,
        tool_simulator_db_path=TOOL_SIMULATOR_DB_PATH,
        output_path=output_path, eval_config=eval_config,
        max_episodes=args.max_episodes,
    )
    logger.info(f"Step 5 done: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AdaMacro Pipeline Runner")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       choices=["qwen2.5-1.5b", "qwen2.5-7b", "llama3.1-8b", "llama3.2-3b"])
    parser.add_argument("--steps", type=str, default="1,2,3,4,5",
                       help="Comma-separated steps to run (1-5)")
    parser.add_argument("--stage", type=str, default="grpo",
                       choices=["base", "sft", "grpo"],
                       help="Evaluation stage")

    # BPE params
    parser.add_argument("--max-merges", type=int, default=50)
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--max-macro-len", type=int, default=6)

    # Training params
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)

    # Eval params
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-atomic-calls", type=int, default=50)
    parser.add_argument("--max-episodes", type=int, default=100)

    args = parser.parse_args()

    ensure_dirs()
    print_config()

    steps = [int(s.strip()) for s in args.steps.split(",")]
    logger.info(f"Running steps: {steps} with model: {args.model}")

    step_fns = {1: run_step1, 2: run_step2, 3: run_step3, 4: run_step4, 5: run_step5}

    for step in steps:
        if step in step_fns:
            logger.info(f"\n{'='*70}\nStep {step}\n{'='*70}")
            start = time.time()
            step_fns[step](args)
            logger.info(f"Step {step} completed in {time.time()-start:.1f}s")
        else:
            logger.warning(f"Unknown step: {step}")


if __name__ == "__main__":
    main()