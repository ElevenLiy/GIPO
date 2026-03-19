"""
AdaMacro Step 3: SFT Data Generation & Training
=================================================

Implements the SFT stage from Section 2.3:
- Generate SFT training data from successful trajectories
- Convert trajectories to use skills where applicable
- Train with LoRA on the augmented tool library (atoms + skills)
- Model learns basic tool call format AND that skills are available

Input:  rl_dataset, augmented_tools.json, skill_library.json
Output: SFT training data, LoRA checkpoint
"""

import json
import copy
import logging
import os
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dataclasses import asdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    RL_DATASET_PATH, AUGMENTED_TOOLS_PATH, SKILL_LIBRARY_PATH,
    SFT_DATA_PATH, CHECKPOINT_DIR, ADAMACRO_OUTPUT_DIR,
    SFTConfig, get_model_path, DEFAULT_MODEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# SFT Data Generation
# ============================================================================

def build_skill_matcher(skill_library_path: str) -> Dict[str, Dict]:
    """
    Build a matcher that identifies subsequences in trajectories that
    can be replaced by skill invocations.

    Tool chains are normalized (version suffixes stripped) so they match
    the normalized tool_names in trajectories.

    Returns:
        {
            skill_name: {
                "tool_chain": [normalized_name1, normalized_name2, ...],
                "length": int,
                "skill_def": dict,
            }
        }
    """
    import re as _re

    def _normalize(name: str) -> str:
        return _re.sub(r'_v\d+\w*$', '', name)

    with open(skill_library_path, "r", encoding="utf-8") as f:
        skill_lib = json.load(f)

    macros = skill_lib.get("macros", {})
    matchers = {}

    for macro_id, macro in macros.items():
        skill_name = f"skill_{macro_id}"
        raw_chain = macro.get("tool_names", [])
        # Normalize chain names to match trajectory tool names
        tool_chain = [_normalize(t) for t in raw_chain]
        if len(tool_chain) >= 2:
            matchers[skill_name] = {
                "tool_chain": tool_chain,
                "length": len(tool_chain),
                "macro": macro,
            }

    return matchers


def match_skills_in_sequence(
    tool_names: List[str],
    tool_args: List[str],
    output_texts: List[str],
    skill_matchers: Dict[str, Dict],
) -> List[Dict]:
    """
    Identify skill-substitutable subsequences in a tool sequence.
    
    Uses greedy longest-match: for each position, try matching the longest
    skill first.
    
    Returns:
        List of actions, each either:
        - {"type": "atomic", "tool_name": str, "args": str, "output": str}
        - {"type": "skill", "skill_name": str, "args": dict, "sub_steps": [...],
           "output": str}
    """
    n = len(tool_names)
    actions = []
    
    # Sort matchers by chain length (longest first for greedy matching)
    sorted_matchers = sorted(
        skill_matchers.items(),
        key=lambda x: x[1]["length"],
        reverse=True,
    )
    
    i = 0
    while i < n:
        matched = False
        
        for skill_name, matcher in sorted_matchers:
            chain = matcher["tool_chain"]
            chain_len = matcher["length"]
            
            if i + chain_len > n:
                continue
            
            # Check if tool_names[i:i+chain_len] matches the skill chain
            if tool_names[i:i + chain_len] == chain:
                # Build skill invocation
                sub_steps = []
                sub_args = {}
                sub_output = ""
                
                for j in range(chain_len):
                    idx = i + j
                    step_args_str = tool_args[idx] if idx < len(tool_args) else "{}"
                    step_output = output_texts[idx] if idx < len(output_texts) else ""
                    
                    try:
                        step_args = json.loads(step_args_str) if isinstance(step_args_str, str) else step_args_str
                    except:
                        step_args = {}
                    
                    sub_steps.append({
                        "tool_name": tool_names[idx],
                        "args": step_args,
                        "output": step_output,
                    })
                    
                    # First step args become exposed params
                    if j == 0:
                        sub_args = step_args
                    
                    # Last step output is the skill output
                    if j == chain_len - 1:
                        sub_output = step_output
                
                actions.append({
                    "type": "skill",
                    "skill_name": skill_name,
                    "args": sub_args,
                    "sub_steps": sub_steps,
                    "output": sub_output,
                })
                
                i += chain_len
                matched = True
                break
        
        if not matched:
            # Atomic action
            args_str = tool_args[i] if i < len(tool_args) else "{}"
            output = output_texts[i] if i < len(output_texts) else ""
            
            actions.append({
                "type": "atomic",
                "tool_name": tool_names[i],
                "args": args_str,
                "output": output,
            })
            i += 1
    
    return actions


def format_tool_call_message(tool_name: str, args: Any) -> str:
    """Format a tool call as the assistant would generate it."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except:
            pass
    args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
    return json.dumps({
        "name": tool_name,
        "arguments": args if isinstance(args, dict) else {}
    }, ensure_ascii=False)


def generate_sft_data(
    rl_dataset_path: str,
    augmented_tools_path: str,
    skill_library_path: str,
    output_path: str,
) -> List[Dict]:
    """
    Generate SFT training data with diversity.
    
    For each successful trajectory, generate MULTIPLE training examples:
    1. Original atomic trajectory (teaches correct tool sequence)
    2. Skill-substituted trajectory (teaches skill usage)
    3. Partial-skill trajectory (teaches mixing skills + atomic)
    
    Key improvements:
    - System prompt matches GRPO/eval format exactly
    - Tool list is per-task (includes gt_tools + skills + sample of others)
    - Multiple trajectory variants per task for diversity
    """
    import random

    with open(rl_dataset_path, "r", encoding="utf-8") as f:
        rl_data = json.load(f)

    with open(augmented_tools_path, "r", encoding="utf-8") as f:
        augmented_tools = json.load(f)

    # Build tool description map (matching GRPO format exactly)
    tool_desc_map = {}
    skill_names = set()

    def _normalize_tool_name(name: str) -> str:
        """Strip version suffixes like _v1, _v13."""
        import re as _re
        return _re.sub(r'_v\d+\w*$', '', name)

    for t in augmented_tools:
        orig_name = t["name"]
        norm_name = _normalize_tool_name(orig_name)
        tag = "[SKILL]" if t.get("is_skill") else ""
        params = t.get("parameters_schema", {})
        param_str = ""
        if params:
            props = params.get("properties", {})
            if props:
                param_names = list(props.keys())[:5]
                param_str = f" params: {param_names}"

        # For skills: show the tool chain so model knows what's inside
        chain_str = ""
        if t.get("is_skill"):
            chain = t.get("tool_chain", [])
            if not chain:
                chain = [s.get("tool_name", "") for s in t.get("execution_plan", [])]
            if chain:
                norm_chain = [_normalize_tool_name(c) for c in chain[:5]]
                chain_str = f" [chain: {' → '.join(norm_chain)}]"

        tool_desc_map[norm_name] = f"- {tag}{norm_name}: {t.get('description','')[:100]}{chain_str}{param_str}"
        if t.get("is_skill"):
            skill_names.add(norm_name)

    # Build skill matcher
    skill_matchers = build_skill_matcher(skill_library_path)

    def build_per_task_tool_list(gt_tools: List[str]) -> str:
        """Build tool list for system prompt. Includes gt_tools + skills + sample."""
        lines = []
        seen = set()

        # 1. Skills first (always visible)
        for tn in skill_names:
            if tn in tool_desc_map:
                lines.append(tool_desc_map[tn])
                seen.add(tn)
            if len(seen) >= 15:
                break

        # 2. GT tools (so model sees correct tools during training)
        for tn in gt_tools:
            if tn not in seen and tn in tool_desc_map:
                lines.append(tool_desc_map[tn])
                seen.add(tn)

        # 3. Tools related to gt (same prefix/category)
        gt_prefixes = set()
        for tn in gt_tools:
            parts = tn.replace("-", "_").split("_")
            if parts:
                gt_prefixes.add(parts[0])
        for tn in tool_desc_map:
            if tn in seen:
                continue
            tn_prefix = tn.replace("-", "_").split("_")[0] if tn else ""
            if tn_prefix in gt_prefixes:
                lines.append(tool_desc_map[tn])
                seen.add(tn)
            if len(seen) >= 35:
                break

        # 4. Random fill to 50
        remaining = [tn for tn in tool_desc_map if tn not in seen]
        random.shuffle(remaining)
        for tn in remaining:
            lines.append(tool_desc_map[tn])
            seen.add(tn)
            if len(seen) >= 50:
                break

        return "\n".join(lines), len([tn for tn in seen if tn in skill_names])

    def build_system_prompt(tool_list_str: str, n_tools: int, n_skills: int) -> str:
        """Build system prompt matching GRPO/eval format exactly."""
        return (
            "You are a tool-calling agent. You MUST use tools to complete tasks. "
            "Do NOT answer directly — always call at least one tool first.\n\n"
            "You have access to both atomic tools and composite skills. "
            "Skills are pre-composed tool chains that execute multiple tools in sequence. "
            "Choose whichever tools (atomic or skill) best fit the task.\n\n"
            f"Available tools ({n_tools} total, including {n_skills} skills):\n"
            f"{tool_list_str}\n\n"
            "To call a tool, respond ONLY with:\n"
            "<tool_call>\n"
            '{"name": "tool_name", "arguments": {"param": "value"}}\n'
            "</tool_call>\n\n"
            "After receiving all tool responses, provide a brief text summary to finish."
        )

    def build_messages(system_prompt, user_prompt, actions) -> List[Dict]:
        """Build conversation messages from action sequence.

        Uses XML <tool_call>/<tool_response> format matching GRPO/eval exactly.
        This ensures SFT teaches the same format the model needs at RL time.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for action in actions:
            if action["type"] == "skill":
                tool_name = action["skill_name"]
                args = action["args"] if isinstance(action["args"], dict) else {}
                output = action["output"][:1500] if isinstance(action["output"], str) else ""
            else:
                tool_name = action["tool_name"]
                try:
                    args = json.loads(action["args"]) if isinstance(action["args"], str) else action["args"]
                except:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                output = action["output"][:1500] if isinstance(action["output"], str) else ""

            # Assistant: XML tool call (matches GRPO format)
            tc_json = json.dumps({"name": tool_name, "arguments": args}, ensure_ascii=False)
            messages.append({
                "role": "assistant",
                "content": f"<tool_call>\n{tc_json}\n</tool_call>",
            })
            # User: XML tool response (matches GRPO format)
            messages.append({
                "role": "user",
                "content": f"<tool_response name=\"{tool_name}\">\n{output}\n</tool_response>",
            })
        messages.append({"role": "assistant", "content": "Task completed successfully."})
        return messages

    episodes = rl_data.get("episodes", [])
    sft_examples = []
    skill_used_count = 0
    atomic_only_count = 0
    mixed_count = 0
    stepwise_count = 0

    for ep in episodes:
        if ep.get("success", 0) != 1:
            continue

        tool_names = [_normalize_tool_name(t) for t in ep.get("tool_names", [])]
        tool_args = ep.get("tool_args", [])
        output_texts = ep.get("output_texts", [])
        user_prompt = ep.get("user_prompt", "")
        gt_tools = [_normalize_tool_name(t) for t in ep.get("tool_names", [])]

        if not tool_names or not user_prompt:
            continue

        # Build per-task tool list and system prompt
        tool_list_str, n_skills = build_per_task_tool_list(gt_tools)
        system_prompt = build_system_prompt(tool_list_str, 50, n_skills)

        # ============================================================
        # Variant 1: Original atomic trajectory
        # Teaches: correct tool sequence for this task
        # ============================================================
        atomic_actions = []
        for i, tn in enumerate(tool_names):
            atomic_actions.append({
                "type": "atomic",
                "tool_name": tn,
                "args": tool_args[i] if i < len(tool_args) else "{}",
                "output": output_texts[i] if i < len(output_texts) else "",
            })

        msgs_atomic = build_messages(system_prompt, user_prompt, atomic_actions)
        sft_examples.append({
            "messages": msgs_atomic,
            "task_name": ep.get("task_name", ""),
            "variant": "atomic",
            "has_skill": False,
            "num_actions": len(atomic_actions),
            "num_skill_actions": 0,
            "num_atomic_actions": len(atomic_actions),
        })
        atomic_only_count += 1

        # ============================================================
        # Variant 2: Full skill substitution
        # Teaches: use skills where possible
        # ============================================================
        skill_actions = match_skills_in_sequence(
            tool_names, tool_args, output_texts, skill_matchers
        )
        has_skill = any(a["type"] == "skill" for a in skill_actions)

        if has_skill:
            msgs_skill = build_messages(system_prompt, user_prompt, skill_actions)
            sft_examples.append({
                "messages": msgs_skill,
                "task_name": ep.get("task_name", ""),
                "variant": "skill",
                "has_skill": True,
                "num_actions": len(skill_actions),
                "num_skill_actions": sum(1 for a in skill_actions if a["type"] == "skill"),
                "num_atomic_actions": sum(1 for a in skill_actions if a["type"] == "atomic"),
            })
            skill_used_count += 1

        # ============================================================
        # Variant 3: Partial skill (only first skill match, rest atomic)
        # Teaches: mixing skills and atomic tools
        # ============================================================
        if has_skill and len(skill_actions) >= 2:
            partial_actions = []
            first_skill_done = False
            for action in skill_actions:
                if action["type"] == "skill" and not first_skill_done:
                    partial_actions.append(action)
                    first_skill_done = True
                elif action["type"] == "skill" and first_skill_done:
                    # Expand this skill back to atomic
                    for sub in action.get("sub_steps", []):
                        partial_actions.append({
                            "type": "atomic",
                            "tool_name": sub["tool_name"],
                            "args": sub.get("args", {}),
                            "output": sub.get("output", ""),
                        })
                else:
                    partial_actions.append(action)

            msgs_partial = build_messages(system_prompt, user_prompt, partial_actions)
            sft_examples.append({
                "messages": msgs_partial,
                "task_name": ep.get("task_name", ""),
                "variant": "partial_skill",
                "has_skill": True,
                "num_actions": len(partial_actions),
                "num_skill_actions": sum(1 for a in partial_actions if a["type"] == "skill"),
                "num_atomic_actions": sum(1 for a in partial_actions if a["type"] == "atomic"),
            })
            mixed_count += 1

        # ============================================================
        # Variant 4: Continuation from intermediate state
        #
        # For trajectory [A, B, C, D], generate:
        #   prefix=[A]     → continuation=[B, C, D, done]
        #   prefix=[A,B]   → continuation=[C, D, done]
        #   prefix=[A,B,C] → continuation=[D, done]
        #
        # Skip step0 (= full trajectory, same as Variant 1).
        # Each sample teaches: "given what's done, here's the
        # COMPLETE remaining sequence to finish the task."
        # ============================================================
        if len(atomic_actions) >= 3:
            # Start from step 1 (skip 0=full traj), cap at 3 samples
            start_indices = list(range(1, len(atomic_actions)))
            if len(start_indices) > 3:
                # Take evenly spaced: early, middle, late
                n = len(start_indices)
                start_indices = [start_indices[0], start_indices[n//2], start_indices[-1]]

            for step_idx in start_indices:
                prefix = atomic_actions[:step_idx]
                remaining = atomic_actions[step_idx:]

                # Build: system + user + prefix (as context) + remaining (as target)
                step_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                # Add prefix (tools already called — this is context)
                for prev in prefix:
                    prev_name = prev.get("skill_name") if prev["type"] == "skill" else prev["tool_name"]
                    try:
                        prev_args = json.loads(prev["args"]) if isinstance(prev["args"], str) else prev["args"]
                    except:
                        prev_args = {}
                    if not isinstance(prev_args, dict):
                        prev_args = {}
                    tc_json = json.dumps({"name": prev_name, "arguments": prev_args}, ensure_ascii=False)
                    step_messages.append({
                        "role": "assistant",
                        "content": f"<tool_call>\n{tc_json}\n</tool_call>",
                    })
                    prev_output = prev.get("output", "")
                    prev_output = prev_output[:1500] if isinstance(prev_output, str) else ""
                    step_messages.append({
                        "role": "user",
                        "content": f"<tool_response name=\"{prev_name}\">\n{prev_output}\n</tool_response>",
                    })

                # Add remaining sequence (target for learning)
                for rem in remaining:
                    rem_name = rem.get("skill_name") if rem["type"] == "skill" else rem["tool_name"]
                    try:
                        rem_args = json.loads(rem["args"]) if isinstance(rem["args"], str) else rem["args"]
                    except:
                        rem_args = {}
                    if not isinstance(rem_args, dict):
                        rem_args = {}
                    tc_json = json.dumps({"name": rem_name, "arguments": rem_args}, ensure_ascii=False)
                    step_messages.append({
                        "role": "assistant",
                        "content": f"<tool_call>\n{tc_json}\n</tool_call>",
                    })
                    rem_output = rem.get("output", "")
                    rem_output = rem_output[:1500] if isinstance(rem_output, str) else ""
                    step_messages.append({
                        "role": "user",
                        "content": f"<tool_response name=\"{rem_name}\">\n{rem_output}\n</tool_response>",
                    })

                # End with done
                step_messages.append({"role": "assistant", "content": "Task completed successfully."})

                sft_examples.append({
                    "messages": step_messages,
                    "task_name": ep.get("task_name", ""),
                    "variant": f"continuation_from_step{step_idx}",
                    "has_skill": False,
                    "num_actions": len(remaining),
                    "num_skill_actions": 0,
                    "num_atomic_actions": len(remaining),
                })
                stepwise_count += 1

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "total_examples": len(sft_examples),
                "atomic_only": atomic_only_count,
                "with_skills": skill_used_count,
                "mixed_skill_atomic": mixed_count,
                "stepwise_next_tool": stepwise_count,
                "num_skills_available": len(skill_matchers),
            },
            "examples": sft_examples,
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"Generated {len(sft_examples)} SFT examples")
    logger.info(f"  Atomic only: {atomic_only_count}")
    logger.info(f"  Full skill: {skill_used_count}")
    logger.info(f"  Mixed (partial skill): {mixed_count}")
    logger.info(f"  Stepwise next-tool: {stepwise_count}")

    return sft_examples


# ============================================================================
# SFT Training with LoRA
# ============================================================================

def _tokenize_with_assistant_mask(messages, tokenizer, max_length=4096):
    """Tokenize messages and mask loss to assistant tokens only.

    This is the same masking logic used in GRPO (step4), ensuring SFT and
    RL stages train on the exact same token positions.

    Returns dict with input_ids, attention_mask, labels (all tensors).
    Non-assistant tokens have labels=-100 (ignored by CrossEntropyLoss).
    """
    import re as _re
    import torch

    # Messages are already in pure system/user/assistant format
    formatted = [{"role": m["role"], "content": m.get("content", "") or ""} for m in messages]
    try:
        text = tokenizer.apply_chat_template(formatted, tokenize=False, add_generation_prompt=False)
    except Exception:
        parts = [f"<|{m['role']}|>\n{m.get('content', '')}" for m in formatted]
        text = "\n".join(parts)

    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = enc["input_ids"].squeeze(0)
    attention_mask = enc["attention_mask"].squeeze(0)
    labels = torch.full_like(input_ids, -100)

    # Find assistant regions in the decoded text
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    patterns = [
        # Qwen2 / ChatML format
        r'<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>)',
        # Llama-3 format
        r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)(?:<\|eot_id\|>)',
    ]
    regions = []
    for pat in patterns:
        for m in _re.finditer(pat, decoded, _re.DOTALL):
            regions.append((m.start(1), m.end(1)))

    if not regions:
        # Fallback: train on everything (better than training on nothing)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # Map character regions to token indices via offset_mapping
    enc2 = tokenizer(decoded, return_offsets_mapping=True, add_special_tokens=False,
                     truncation=True, max_length=max_length)
    offsets = enc2.get("offset_mapping", [])

    if offsets and len(offsets) >= len(input_ids):
        for cs, ce in regions:
            for ti in range(len(input_ids)):
                if ti < len(offsets):
                    ts, te = offsets[ti]
                    if te > cs and ts < ce:
                        labels[ti] = input_ids[ti]
    else:
        # Fallback: heuristic role-based detection
        toks = [tokenizer.decode([t]) for t in input_ids.tolist()]
        in_asst = False
        for idx, tok_text in enumerate(toks):
            if 'assistant' in tok_text.lower() and not in_asst:
                in_asst = True
                continue
            if in_asst and any(x in tok_text for x in ['im_end', 'eot_id', 'im_start', 'start_header']):
                in_asst = False
                continue
            if in_asst:
                labels[idx] = input_ids[idx]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_sft(
    model_name: str,
    sft_data_path: str,
    output_dir: str,
    sft_config: SFTConfig,
):
    """
    Fine-tune model with LoRA using SFT data.

    Uses standard HuggingFace Trainer with assistant-only loss masking.
    Only assistant tokens (tool calls + final summary) contribute to loss,
    matching the GRPO stage's tokenize_with_assistant_mask exactly.
    """
    import torch
    from torch.utils.data import Dataset as TorchDataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    model_path = get_model_path(model_name)
    logger.info(f"Loading model: {model_name} from {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=sft_config.lora_rank,
        lora_alpha=sft_config.lora_alpha,
        lora_dropout=sft_config.lora_dropout,
        target_modules=sft_config.lora_target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load SFT data
    with open(sft_data_path, "r", encoding="utf-8") as f:
        sft_data = json.load(f)

    examples = sft_data.get("examples", [])
    logger.info(f"Loaded {len(examples)} SFT examples")

    # ----------------------------------------------------------------
    # Pre-tokenize all examples with assistant-only mask
    # ----------------------------------------------------------------
    max_seq_len = sft_config.max_seq_length
    tokenized_examples = []
    n_skipped = 0
    n_asst_tokens_total = 0

    for ex in examples:
        item = _tokenize_with_assistant_mask(ex["messages"], tokenizer, max_length=max_seq_len)
        n_asst = (item["labels"] != -100).sum().item()
        if n_asst == 0:
            n_skipped += 1
            continue
        tokenized_examples.append(item)
        n_asst_tokens_total += n_asst

    logger.info(f"Tokenized {len(tokenized_examples)} examples "
                f"({n_skipped} skipped with 0 assistant tokens)")
    if tokenized_examples:
        avg_asst = n_asst_tokens_total / len(tokenized_examples)
        avg_total = sum(len(ex["input_ids"]) for ex in tokenized_examples) / len(tokenized_examples)
        logger.info(f"Avg tokens per example: {avg_total:.0f} total, {avg_asst:.0f} assistant-only "
                    f"({avg_asst/max(avg_total,1)*100:.1f}% supervised)")

    class PreTokenizedDataset(TorchDataset):
        """Dataset that returns pre-tokenized examples with labels mask."""
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            return self.items[idx]

    dataset = PreTokenizedDataset(tokenized_examples)

    # Data collator: pad to max length within batch
    def collate_fn(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for b in batch:
            pad_len = max_len - len(b["input_ids"])
            input_ids.append(torch.cat([b["input_ids"], torch.full((pad_len,), tokenizer.pad_token_id)]))
            attention_mask.append(torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            labels.append(torch.cat([b["labels"], torch.full((pad_len,), -100)]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

    logger.info(f"Dataset size: {len(dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=sft_config.num_epochs,
        per_device_train_batch_size=sft_config.per_device_batch_size,
        gradient_accumulation_steps=sft_config.gradient_accumulation_steps,
        learning_rate=sft_config.learning_rate,
        warmup_ratio=sft_config.warmup_ratio,
        weight_decay=sft_config.weight_decay,
        logging_steps=sft_config.logging_steps,
        save_steps=sft_config.save_steps,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        dataloader_num_workers=0,  # pre-tokenized, no need for workers
        remove_unused_columns=False,
    )

    # Enable gradient checkpointing with input requires_grad
    # (required for PEFT + gradient_checkpointing + standard Trainer)
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    # Train
    logger.info("Starting SFT training (assistant-only loss mask)...")
    trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"SFT training complete. Model saved to {output_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AdaMacro Step 3: SFT Training")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--rl-dataset", type=str, default=RL_DATASET_PATH)
    parser.add_argument("--augmented-tools", type=str, default=AUGMENTED_TOOLS_PATH)
    parser.add_argument("--skill-library", type=str, default=SKILL_LIBRARY_PATH)
    parser.add_argument("--sft-data", type=str, default=SFT_DATA_PATH)
    parser.add_argument("--output-dir", type=str, default=os.path.join(CHECKPOINT_DIR, "sft"))
    parser.add_argument("--generate-only", action="store_true",
                       help="Only generate SFT data, skip training")
    parser.add_argument("--train-only", action="store_true",
                       help="Only train, skip data generation")
    # Override SFT config
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    sft_config = SFTConfig()
    if args.epochs: sft_config.num_epochs = args.epochs
    if args.lr: sft_config.learning_rate = args.lr
    if args.batch_size: sft_config.per_device_batch_size = args.batch_size
    if args.lora_rank: sft_config.lora_rank = args.lora_rank
    
    logger.info("=" * 70)
    logger.info("AdaMacro Step 3: SFT Data Generation & Training")
    logger.info("=" * 70)
    
    # Step 1: Generate SFT data
    if not args.train_only:
        logger.info("\n[Phase 1] Generating SFT training data...")
        generate_sft_data(
            args.rl_dataset,
            args.augmented_tools,
            args.skill_library,
            args.sft_data,
        )
    
    # Step 2: Train
    if not args.generate_only:
        logger.info(f"\n[Phase 2] Training with model: {args.model}")
        train_sft(
            args.model,
            args.sft_data,
            args.output_dir,
            sft_config,
        )


if __name__ == "__main__":
    main()