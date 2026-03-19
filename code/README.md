# AdaMacro: Budgeted Skill Discovery by Consolidating Frequent Tool Trajectories

## Project Structure

```
AdaMacro/
├── configs/
│   └── config.py              # ★ ALL PATHS & HYPERPARAMETERS — modify here for new datasets
├── scripts/
│   ├── step1_bpe_mining.py        # BPE macro mining (Section 2.1)
│   ├── step2_skill_instantiation.py # Skill instantiation + trace (Section 2.2)
│   ├── step3_sft_training.py      # SFT data generation + LoRA training
│   ├── step4_grpo_training.py     # GRPO with phased reward (Section 2.3)
│   ├── step5_evaluation.py        # Evaluation with full metrics (Section 3.3)
│   └── run_pipeline.py            # Orchestrates all steps
├── run_adamacro.sh                # Single-model sbatch script
├── run_all_experiments.sh         # Batch submission for all models
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Modify paths in `configs/config.py`

All file paths are centralized. To switch datasets (e.g., TOOLATHLON → TOUCAN):
- Change `DATASET_NAME`, `TRAJECTORIES_DIR`, and relevant input paths
- Output paths auto-adapt via `DATASET_NAME`

### 2. Run full pipeline for one model

```bash
sbatch run_adamacro.sh qwen2.5-7b 1,2,3,4,5
```

### 3. Run all models

```bash
bash run_all_experiments.sh
```

### 4. Run individual steps

```bash
# Step 1: BPE Mining only
python scripts/run_pipeline.py --model qwen2.5-7b --steps 1

# Step 3: SFT only (after steps 1,2)
python scripts/run_pipeline.py --model qwen2.5-7b --steps 3

# Step 5: Evaluate SFT checkpoint
python scripts/run_pipeline.py --model qwen2.5-7b --steps 5 --stage sft
```

## Pipeline Steps

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `step1_bpe_mining.py` | `rl_dataset_llm_v3.json` | `skill_library.json` |
| 2 | `step2_skill_instantiation.py` | `skill_library.json` + `all_tools_v2.json` | `augmented_tools.json` |
| 3 | `step3_sft_training.py` | `rl_dataset` + `augmented_tools` | SFT LoRA checkpoint |
| 4 | `step4_grpo_training.py` | SFT checkpoint + rollouts | GRPO LoRA checkpoint |
| 5 | `step5_evaluation.py` | Checkpoint + test data | Evaluation metrics |

## Switching Datasets

Edit `configs/config.py`:
```python
DATASET_NAME = "TOUCAN"  # was "TOOLATHLON"
TRAJECTORIES_DIR = "/path/to/toucan/trajectories"
# ... other paths
```

## Models

| Model | Path |
|-------|------|
| `qwen2.5-1.5b` | `/seu_share2/home/fenglei/sharedata/Qwen2.5-1.5B-Instruct` |
| `qwen2.5-7b` | `/seu_share2/home/fenglei/sharedata/Qwen2.5-7B-Instruct` |
| `llama3.1-8b` | `/seu_share2/home/fenglei/sharedata/Llama-3.1-8B-Instruct` |
| `llama3.2-3b` | `/seu_share2/home/fenglei/sharedata/Llama-3.2-3B-Instruct` |
