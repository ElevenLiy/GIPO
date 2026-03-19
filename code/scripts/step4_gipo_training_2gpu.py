"""
AdaMacro Step 4: GIPO Training — 2-GPU Version for 7B/8B Models
================================================================

This script adapts GIPO training for larger models (Qwen2.5-7B, Llama3.1-8B)
that require 2 GPUs to fit in memory.

Key differences from single-GPU step4_gipo_training.py:
  - Model parallelism via device_map="auto" with explicit max_memory per GPU
  - Gradient checkpointing to reduce activation memory
  - Adjusted hyperparameters (lower lr, smaller LoRA, shorter sequences)
  - NOT using DDP/torchrun — rollout generation is serial, so model
    parallelism (splitting layers across GPUs) is the correct approach

All reward, rollout, and imagination logic is imported from the original
step4_gipo_training.py to avoid code duplication.

Usage:
  python step4_gipo_training_2gpu.py --model qwen2.5-7b
  python step4_gipo_training_2gpu.py --model llama3.1-8b --epochs 2 --lr 1e-6
"""

import json
import logging
import os
import math
import random
import time
from typing import List, Dict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    RL_DATASET_PATH, AUGMENTED_TOOLS_PATH, SKILL_LIBRARY_PATH,
    TOOL_SIMULATOR_DB_PATH, GRPO_DATA_PATH, CHECKPOINT_DIR,
    ADAMACRO_OUTPUT_DIR, GIPO7BConfig, get_model_path, DEFAULT_MODEL,
)

# Import all shared components from the original GIPO training script
from step4_gipo_training import (
    ExecutionLogger,
    AdaMacroReward,
    ToolEnvironment,
    normalize_tool_name,
    find_best_matching_skill,
    find_counterfactual_action,
    run_imagination_branch,
    run_rollout,
    tokenize_with_assistant_mask,
    parse_tool_call,
    _truncate_args,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# 2-GPU Model Loading
# ============================================================================

def load_model_2gpu(model_path: str, sft_checkpoint_dir: str, config: GIPO7BConfig):
    """
    Load a 7B/8B model across 2 GPUs using model parallelism.

    Strategy:
      - device_map="auto" splits layers across available GPUs
      - max_memory limits per-GPU usage to leave room for KV cache & gradients
      - Gradient checkpointing trades compute for memory
      - LoRA keeps trainable params small (~0.5% of 7B)

    Returns: model, tokenizer, device (device of the first parameter)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType

    n_gpus = torch.cuda.device_count()
    logger.info(f"Available GPUs: {n_gpus}")
    if n_gpus < 2:
        logger.warning("Less than 2 GPUs available! Falling back to single GPU.")

    # Set max_memory per GPU — leave ~15GB headroom for gradients/KV cache/activations
    max_memory = {}
    for i in range(n_gpus):
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        # Use ~80% of each GPU (e.g., 64GB out of 80GB H100)
        max_memory[i] = f"{int(total_mem * 0.80)}GiB"
    max_memory["cpu"] = "32GiB"  # offload buffer
    logger.info(f"Max memory allocation: {max_memory}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model with model parallelism
    if os.path.exists(os.path.join(sft_checkpoint_dir, "adapter_config.json")):
        logger.info(f"Loading SFT LoRA from {sft_checkpoint_dir}")
        base = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map="auto", max_memory=max_memory,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, sft_checkpoint_dir)
        model = model.merge_and_unload()
        logger.info("SFT LoRA merged into base model")
    else:
        logger.info("No SFT checkpoint; starting from base model")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map="auto", max_memory=max_memory,
            trust_remote_code=True,
        )

    # Log device map
    if hasattr(model, "hf_device_map"):
        devices_used = set(str(v) for v in model.hf_device_map.values())
        logger.info(f"Model device map uses: {devices_used} "
                    f"({len(model.hf_device_map)} layers distributed)")

    model.config.use_cache = False

    # Enable gradient checkpointing — critical for 7B on 2×GPU
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Get device of first parameter (inputs should go here)
    device = next(model.parameters()).device
    logger.info(f"Input tensor device: {device}")

    return model, tokenizer, device


# ============================================================================
# Training Loop (adapted for 2-GPU)
# ============================================================================

def train_grpo(
    model_name: str,
    sft_checkpoint_dir: str,
    grpo_data_path: str,
    output_dir: str,
    grpo_config: GIPO7BConfig,
):
    import torch
    from transformers import get_cosine_schedule_with_warmup

    # ---------------------------------------------------------------- model (2-GPU)
    model_path = get_model_path(model_name)
    logger.info(f"Loading model for 2-GPU: {model_path}")
    model, tokenizer, device = load_model_2gpu(model_path, sft_checkpoint_dir, grpo_config)

    # ---------------------------------------------------------------- env
    env = ToolEnvironment(AUGMENTED_TOOLS_PATH, TOOL_SIMULATOR_DB_PATH, RL_DATASET_PATH)
    reward_fn = AdaMacroReward(grpo_config, skill_definitions=env.skills)

    # Build tool description matching SFT format (with parameters)
    with open(AUGMENTED_TOOLS_PATH, "r") as f:
        all_augmented_tools = json.load(f)

    tool_desc_map = {}
    norm_to_orig = {}
    for t in all_augmented_tools:
        orig_name = t["name"]
        norm_name = normalize_tool_name(orig_name)
        tag = "[SKILL]" if t.get("is_skill") else ""
        params = t.get("parameters_schema", {})
        param_str = ""
        if params:
            props = params.get("properties", {})
            if props:
                param_names = list(props.keys())[:5]
                param_str = f" params: {param_names}"

        chain_str = ""
        if t.get("is_skill"):
            chain = t.get("tool_chain", [])
            if not chain:
                chain = [s.get("tool_name", "") for s in t.get("execution_plan", [])]
            if chain:
                norm_chain = [normalize_tool_name(c) for c in chain[:5]]
                chain_str = f" [chain: {' → '.join(norm_chain)}]"

        tool_desc_map[norm_name] = f"- {tag}{norm_name}: {t.get('description','')[:100]}{chain_str}{param_str}"
        if norm_name not in norm_to_orig:
            norm_to_orig[norm_name] = orig_name

    def build_system_prompt(task_name: str) -> str:
        lines = []
        seen = set()

        task_lower = task_name.lower().replace("-", "_")
        category_keywords = set()
        for part in task_lower.split("_"):
            if len(part) >= 3:
                category_keywords.add(part)

        norm_skill_names = set(normalize_tool_name(s) for s in env.skills)
        for tn, desc in tool_desc_map.items():
            if tn in norm_skill_names:
                lines.append(desc)
                seen.add(tn)
            if len(seen) >= 15:
                break

        for tn, desc in tool_desc_map.items():
            if tn in seen:
                continue
            tn_lower = tn.lower().replace("-", "_")
            if any(kw in tn_lower for kw in category_keywords):
                lines.append(desc)
                seen.add(tn)
            if len(seen) >= 40:
                break

        other_tools = [tn for tn in tool_desc_map if tn not in seen]
        random.shuffle(other_tools)
        for tn in other_tools:
            lines.append(tool_desc_map[tn])
            seen.add(tn)
            if len(seen) >= 50:
                break

        tool_list = "\n".join(lines)
        n_skills = sum(1 for tn in seen if tn in norm_skill_names)

        return (
            "You are a tool-calling agent. You MUST use tools to complete tasks. "
            "Do NOT answer directly — always call at least one tool first.\n\n"
            "You have access to both atomic tools and composite skills. "
            "Skills are pre-composed tool chains that execute multiple tools in sequence. "
            "Choose whichever tools (atomic or skill) best fit the task.\n\n"
            f"Available tools ({len(seen)} total, including {n_skills} skills):\n"
            f"{tool_list}\n\n"
            "To call a tool, respond ONLY with:\n"
            "<tool_call>\n"
            '{"name": "tool_name", "arguments": {"param": "value"}}\n'
            "</tool_call>\n\n"
            "After receiving all tool responses, provide a brief text summary to finish."
        )

    # ---------------------------------------------------------------- prompts
    with open(RL_DATASET_PATH, "r") as f:
        rl_data = json.load(f)
    episodes = rl_data.get("episodes", [])

    train_prompts = []
    for ep in episodes:
        if ep.get("success", 0) == 1 and ep.get("user_prompt") and ep.get("tool_names"):
            raw_gt = ep.get("tool_names", [])
            norm_gt = list(dict.fromkeys(normalize_tool_name(t) for t in raw_gt))
            train_prompts.append({
                "user_prompt": ep["user_prompt"],
                "task_name": ep.get("task_name", ""),
                "gt_tools": norm_gt,
            })
    logger.info(f"Training prompts: {len(train_prompts)}")

    # ---------------------------------------------------------------- hyper-params
    G = grpo_config.group_size
    num_epochs = grpo_config.num_epochs
    lr = grpo_config.learning_rate
    grad_accum = grpo_config.gradient_accumulation_steps
    max_turns = 10
    max_gen_tok = grpo_config.max_gen_length
    base_temp = grpo_config.temperature

    total_prompts = len(train_prompts) * num_epochs
    total_steps = total_prompts // grad_accum
    warmup_steps = int(total_steps * 0.1)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max(total_steps, 1))

    # ---------------------------------------------------------------- execution logger
    log_path = os.path.join(output_dir, "execution_log.jsonl")
    exec_logger = ExecutionLogger(log_path)

    logger.info(f"GIPO-2GPU Training  G={G}  epochs={num_epochs}  lr={lr}  "
                f"grad_accum={grad_accum}  total_steps≈{total_steps}  "
                f"max_seq_len={grpo_config.max_seq_length}  "
                f"lora_rank={grpo_config.lora_rank}  "
                f"grad_ckpt={grpo_config.gradient_checkpointing}")
    logger.info(f"GIPO params: img_scale={grpo_config.gipo_step_reward_scale} "
                f"step_cap={grpo_config.gipo_step_reward_cap} "
                f"total_cap={grpo_config.gipo_total_reward_cap}")

    # Log GPU memory before training
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        logger.info(f"GPU {i}: allocated={alloc:.1f}GB  reserved={reserved:.1f}GB")

    # ---------------------------------------------------------------- training loop
    # (identical to step4_gipo_training.py from here)
    global_step = 0
    acc_loss = acc_reward = acc_steps_ep = acc_skill_r = 0.0
    acc_img_steps = 0.0
    acc_cnt = 0
    best_reward = -1e9

    for epoch in range(num_epochs):
        random.shuffle(train_prompts)
        logger.info(f"\n{'='*60}\nEpoch {epoch+1}/{num_epochs}\n{'='*60}")

        for pidx, pdata in enumerate(train_prompts):
            user_prompt = pdata["user_prompt"]
            gt_tools = pdata["gt_tools"]

            # ========== GIPO Phase 1: Generate base rollouts + branches ==========
            model.eval()
            rollouts = []

            system_prompt = build_system_prompt(pdata["task_name"])

            skill_biased_prompt = system_prompt + (
                "\n\nIMPORTANT: You SHOULD prefer [SKILL] tools over atomic tools when possible. "
                "Skills chain multiple steps and are more efficient. "
                "Check the [SKILL] entries in the tool list first."
            )

            _skill_chains = {}
            for sname, sdef in env.skills.items():
                chain = sdef.get("tool_chain", [])
                if not chain:
                    chain = [s.get("tool_name", "") for s in sdef.get("execution_plan", [])]
                _skill_chains[sname] = [normalize_tool_name(t) for t in chain]

            def _compute_reward(ro):
                ut = [name for name, _ in ro["actions"]]
                sn = list(ro["skill_names_used"])
                rw, bd = reward_fn.compute(
                    used_tools=ut, gt_tools=gt_tools,
                    skill_traces=ro["skill_traces"], skill_names=sn,
                    num_decision_steps=ro["num_steps"],
                    num_skill_calls=ro["num_skill_calls"],
                    total_atomic_cost=ro["total_atomic"],
                    completed=ro["completed"], max_steps=max_turns,
                )
                ro["reward"] = rw
                ro["reward_breakdown"] = bd
                return rw, bd

            # --- Base rollout 0: skill-biased ---
            t0 = max(0.1, base_temp * 0.8)
            base_0 = run_rollout(
                model, tokenizer, env,
                skill_biased_prompt, user_prompt,
                max_turns=max_turns, max_new_tokens=max_gen_tok,
                temperature=t0, device=device,
                gt_tools_len=len(gt_tools),
            )
            base_0["rollout_type"] = "base"
            _compute_reward(base_0)

            # --- Base rollout 1: oracle-seeded ---
            t1 = max(0.1, base_temp * 1.2)
            oracle_first = None
            if gt_tools:
                best_skill = find_best_matching_skill(gt_tools, env.skills)
                if best_skill and random.random() < 0.5:
                    oracle_first = best_skill
                else:
                    oracle_first = random.choice(gt_tools)

            base_1 = run_rollout(
                model, tokenizer, env,
                system_prompt, user_prompt,
                max_turns=max_turns, max_new_tokens=max_gen_tok,
                temperature=t1, device=device,
                oracle_first_tool=oracle_first,
                gt_tools_len=len(gt_tools),
            )
            base_1["rollout_type"] = "base"
            _compute_reward(base_1)

            rollouts = [base_0, base_1]

            # --- 0-step resample ---
            has_any_action = any(r["num_steps"] > 0 for r in rollouts)
            resample_attempts = 0
            while not has_any_action and resample_attempts < 3:
                resample_attempts += 1
                t = min(base_temp * (1.5 + resample_attempts * 0.5), 2.0)
                retry_ro = run_rollout(
                    model, tokenizer, env,
                    system_prompt, user_prompt,
                    max_turns=max_turns, max_new_tokens=max_gen_tok,
                    temperature=t, device=device,
                    gt_tools_len=len(gt_tools),
                )
                retry_ro["rollout_type"] = "base"
                _compute_reward(retry_ro)
                if retry_ro["num_steps"] > 0:
                    for ri in range(len(rollouts)):
                        if rollouts[ri]["num_steps"] == 0:
                            rollouts[ri] = retry_ro
                            break
                    has_any_action = True

            # --- GIPO: counterfactual branches ---
            n_branches = 0
            for ri, base_ro in enumerate(list(rollouts)):
                if base_ro["num_steps"] == 0:
                    continue

                branch_step = None
                cf_action = None
                actions = base_ro["actions"]
                offsets = base_ro.get("action_msg_offsets", [])

                for si, (tool_name, tool_args) in enumerate(actions):
                    is_skill = tool_name in env.skills or any(
                        normalize_tool_name(s) == normalize_tool_name(tool_name)
                        for s in env.skills
                    )
                    cf = find_counterfactual_action(
                        chosen_tool=tool_name,
                        is_skill=is_skill,
                        skills=env.skills,
                        skill_chains=_skill_chains,
                        original_arguments=tool_args,
                    )
                    if cf is not None:
                        branch_step = si
                        cf_action = cf
                        break

                if branch_step is not None and branch_step < len(offsets):
                    msg_offset = offsets[branch_step]
                    prefix_messages = base_ro["messages"][:msg_offset]
                    prefix_actions = actions[:branch_step]

                    _prefix_atomic = 0
                    _prefix_skill_traces = []
                    _prefix_skill_names = []
                    for _pa_name, _ in prefix_actions:
                        _pa_norm = normalize_tool_name(_pa_name)
                        _is_prefix_skill = (_pa_name in env.skills or any(
                            normalize_tool_name(s) == _pa_norm for s in env.skills
                        ))
                        if _is_prefix_skill:
                            _sk_def = env.skills.get(_pa_name)
                            if not _sk_def:
                                for _sn, _sd in env.skills.items():
                                    if normalize_tool_name(_sn) == _pa_norm:
                                        _sk_def = _sd
                                        break
                            if _sk_def:
                                _base_skill_names = base_ro.get("skill_names_used", [])
                                _base_skill_traces = base_ro.get("skill_traces", [])
                                _found_trace = False
                                for _bsi, _bsn in enumerate(_base_skill_names):
                                    if _bsn == _pa_name and _bsi < len(_base_skill_traces):
                                        _prefix_atomic += max(len(_base_skill_traces[_bsi]), 1)
                                        _prefix_skill_traces.append(_base_skill_traces[_bsi])
                                        _prefix_skill_names.append(_bsn)
                                        _found_trace = True
                                        break
                                if not _found_trace:
                                    _chain = _sk_def.get("tool_chain", [])
                                    if not _chain:
                                        _chain = [s.get("tool_name", "") for s in _sk_def.get("execution_plan", [])]
                                    _prefix_atomic += max(len(_chain), 1)
                            else:
                                _prefix_atomic += 1
                        else:
                            _prefix_atomic += 1

                    branch = run_imagination_branch(
                        model, tokenizer, env,
                        prefix_messages=prefix_messages,
                        prefix_actions=prefix_actions,
                        cf_tool_name=cf_action["name"],
                        cf_arguments=cf_action["arguments"],
                        max_turns=max_turns,
                        max_new_tokens=max_gen_tok,
                        temperature=base_ro["temperature"],
                        device=device,
                        gt_tools_len=len(gt_tools),
                        prefix_total_atomic=_prefix_atomic,
                        prefix_skill_traces=_prefix_skill_traces,
                        prefix_skill_names=_prefix_skill_names,
                    )
                    branch["rollout_type"] = "branch"
                    branch["branch_info"] = {
                        "parent_idx": ri,
                        "branch_step": branch_step,
                        "original_tool": actions[branch_step][0],
                        "cf_tool": cf_action["name"],
                        "cf_is_skill": cf_action["is_skill"],
                    }
                    _compute_reward(branch)
                    rollouts.append(branch)
                    n_branches += 1

            # ========== GIPO Phase 1.5: Imagination Reward ==========
            import re
            img_scale = grpo_config.gipo_step_reward_scale
            img_step_cap = grpo_config.gipo_step_reward_cap
            img_total_cap = grpo_config.gipo_total_reward_cap

            for ro in rollouts:
                if ro.get("rollout_type") != "branch":
                    continue
                binfo = ro.get("branch_info", {})
                parent_idx = binfo.get("parent_idx")
                if parent_idx is None or parent_idx >= len(rollouts):
                    continue
                base_ro = rollouts[parent_idx]

                delta = ro["reward"] - base_ro["reward"]
                r_img = delta * img_scale
                r_img = max(-img_step_cap, min(img_step_cap, r_img))
                r_img = max(-img_total_cap, min(img_total_cap, r_img))

                ro["reward"] += r_img
                ro["reward_breakdown"]["r_imagination"] = round(r_img, 4)
                base_ro["reward"] -= r_img
                base_ro["reward_breakdown"]["r_imagination"] = round(-r_img, 4)

            for ro in rollouts:
                ro["reward"] = max(ro["reward"], 0.0)

            # ========== Phase 2: group-relative advantage ==========
            rewards = [r["reward"] for r in rollouts]
            mu = sum(rewards) / len(rewards)
            raw_std = math.sqrt(sum((r - mu)**2 for r in rewards) / len(rewards))

            MIN_ADV_STD = 0.05
            if raw_std < 1e-6:
                for r in rollouts:
                    r["advantage"] = 0.0
            else:
                std = max(raw_std, MIN_ADV_STD)
                for i, r in enumerate(rollouts):
                    r["advantage"] = (rewards[i] - mu) / std
                    r["advantage"] = max(-3.0, min(3.0, r["advantage"]))

            # ========== Logging ==========
            show_detail = pidx < 5 or pidx % 50 == 0
            if show_detail:
                logger.info(f"  [prompt {pidx}] gt_tools={gt_tools[:4]}... "
                            f"rollouts={len(rollouts)} (branches={n_branches}/2)")
                for _, r in enumerate(rollouts):
                    used = [name for name, _ in r["actions"]]
                    bd = r["reward_breakdown"]
                    rtype = r.get("rollout_type", "?")
                    binfo = r.get("branch_info", {})
                    branch_tag = ""
                    if rtype == "branch" and binfo:
                        cf_g = "SKILL" if binfo.get("cf_is_skill") else "ATOM"
                        branch_tag = f" [branch@{binfo['branch_step']}→{cf_g}:{binfo['cf_tool'][:25]}]"

                    logger.info(
                        f"    [{rtype}] steps={r['num_steps']} skills={r['num_skill_calls']} "
                        f"completed={r['completed']} reward={r['reward']:.3f} "
                        f"(task={bd['r_task']:.2f} sk={bd.get('skill_bonus',0):+.2f} "
                        f"eff={bd['r_efficiency']:.2f} img={bd.get('r_imagination',0):+.2f}) "
                        f"adv={r['advantage']:+.2f} tools={used[:5]}{branch_tag}"
                    )
            exec_logger.log_prompt(epoch, pidx, global_step, user_prompt, gt_tools, rollouts)

            # ========== Phase 3: policy gradient ==========
            model.train()
            prompt_loss = 0.0

            for rollout in rollouts:
                adv = rollout["advantage"]
                if rollout["num_steps"] == 0:
                    continue

                input_ids, attn_mask, labels = tokenize_with_assistant_mask(
                    rollout["messages"], tokenizer, max_length=grpo_config.max_seq_length)

                n_asst = (labels != -100).sum().item()
                if n_asst == 0:
                    continue

                input_ids = input_ids.unsqueeze(0).to(device)
                attn_mask = attn_mask.unsqueeze(0).to(device)
                labels = labels.unsqueeze(0).to(device)

                out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                n_rollouts = len(rollouts)
                pg_loss = adv * out.loss / (n_rollouts * grad_accum)

                if torch.isfinite(pg_loss):
                    pg_loss.backward()
                    prompt_loss += pg_loss.item() * n_rollouts * grad_accum

            # Stats
            n_rollouts = len(rollouts)
            acc_loss += prompt_loss
            acc_reward += sum(rewards) / len(rewards)
            skill_ratio = (sum(r["num_skill_calls"] for r in rollouts)
                           / max(sum(r["num_steps"] for r in rollouts), 1))
            acc_skill_r += skill_ratio
            acc_steps_ep += sum(r["num_steps"] for r in rollouts) / n_rollouts
            acc_img_steps += n_branches
            acc_cnt += 1

            # ========== Phase 4: optimizer step ==========
            if (pidx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 5 == 0 or global_step <= 10:
                    al = acc_loss / max(acc_cnt, 1)
                    ar = acc_reward / max(acc_cnt, 1)
                    sr = acc_skill_r / max(acc_cnt, 1)
                    se = acc_steps_ep / max(acc_cnt, 1)
                    br = acc_img_steps / max(acc_cnt, 1)
                    clr = scheduler.get_last_lr()[0]

                    # Log GPU memory periodically
                    mem_info = ""
                    for gi in range(torch.cuda.device_count()):
                        alloc = torch.cuda.memory_allocated(gi) / (1024**3)
                        mem_info += f" GPU{gi}={alloc:.1f}GB"

                    logger.info(
                        f"step {global_step}/{total_steps} | "
                        f"loss={al:.4f}  reward={ar:.3f}  "
                        f"skill_ratio={sr:.2f}  avg_steps={se:.1f}  "
                        f"branches={br:.1f}/2  "
                        f"lr={clr:.2e}  epoch={epoch + (pidx+1)/len(train_prompts):.2f}"
                        f"  mem:[{mem_info.strip()}]"
                    )
                    if ar > best_reward:
                        best_reward = ar
                    acc_loss = acc_reward = acc_skill_r = acc_steps_ep = 0.0
                    acc_img_steps = 0.0
                    acc_cnt = 0

                if global_step > 0 and global_step % grpo_config.save_steps == 0:
                    sp = os.path.join(output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(sp)
                    tokenizer.save_pretrained(sp)
                    logger.info(f"Checkpoint → {sp}")

        # Flush remaining accumulated gradients at epoch end
        if (pidx + 1) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            logger.info(f"  [epoch {epoch+1} end] flushed remaining gradients, step={global_step}")

        logger.info(f"Epoch {epoch+1} done.  best_reward={best_reward:.3f}")

    # Final save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    exec_logger.save_summary()
    logger.info(f"GIPO-2GPU complete → {output_dir}  best_reward={best_reward:.3f}")


# ============================================================================
# Backward-compatible wrapper
# ============================================================================

def generate_grpo_rollouts(*args, **kwargs):
    logger.info("Online GRPO: rollouts are generated on-the-fly, skipping offline generation.")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AdaMacro Step 4: GIPO Training (2-GPU for 7B/8B)")
    parser.add_argument("--model", type=str, default="qwen2.5-7b")
    parser.add_argument("--sft-checkpoint", type=str, default=os.path.join(CHECKPOINT_DIR, "sft"))
    parser.add_argument("--rl-dataset", type=str, default=RL_DATASET_PATH)
    parser.add_argument("--augmented-tools", type=str, default=AUGMENTED_TOOLS_PATH)
    parser.add_argument("--skill-library", type=str, default=SKILL_LIBRARY_PATH)
    parser.add_argument("--tool-simulator-db", type=str, default=TOOL_SIMULATOR_DB_PATH)
    parser.add_argument("--grpo-data", type=str, default=GRPO_DATA_PATH)
    parser.add_argument("--output-dir", type=str, default=os.path.join(CHECKPOINT_DIR, "gipo_7b"))
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    grpo_config = GIPO7BConfig()
    if args.group_size: grpo_config.group_size = args.group_size
    if args.epochs: grpo_config.num_epochs = args.epochs
    if args.lr: grpo_config.learning_rate = args.lr

    logger.info("=" * 70)
    logger.info("AdaMacro Step 4: GIPO Training (2-GPU for 7B/8B models)")
    logger.info("=" * 70)

    if args.generate_only:
        logger.info("Online GRPO — no offline generation needed.")
        return

    train_grpo(args.model, args.sft_checkpoint, args.grpo_data,
               args.output_dir, grpo_config)


if __name__ == "__main__":
    main()
