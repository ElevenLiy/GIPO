"""
AdaMacro Step 4: GIPO-API Training (Granularity-Imagination Policy Optimization with API)
==========================================================================================

Same as GIPO training but uses an external LLM API (e.g., Qwen via DashScope)
to simulate tool outputs instead of the static tool_simulator_database.
This produces more realistic training signals: tools may return correct results,
partial results, errors, or timeouts — closer to real deployment.

GIPO extends GRPO with per-step counterfactual imagination:

At each tool-calling step, if the model's chosen action has an alternative
at a different granularity level (atomic ↔ skill), GIPO simulates the
counterfactual action in the tool environment and compares immediate
gt_tools coverage. This produces a dense **process-level reward** signal
(R_imagination) that teaches the model *when* to use skills vs atomic tools.

Key mechanism:
  - Model calls atomic tool X → check if a skill containing X exists
    → simulate skill → compare coverage → process reward for this step
  - Model calls skill S → check first atomic tool in S's chain
    → simulate atomic → compare coverage → process reward for this step
  - Process reward is accumulated across all eligible steps in a rollout

Training process overview:
==========================

Each "step" in the log = 1 optimizer update, which processes (grad_accum × prompts).

For EACH training prompt:
  Phase 1: Generate 2 base rollouts (base_0 with skill-biased prompt, base_1 with neutral prompt).
     Each rollout = model interacts with environment step-by-step:
       turn 0: model generates → parse tool_call → env.execute → get tool_response
       turn 1: model sees history + tool_response → generates next tool_call → env.execute
       ...
       turn N: model generates text (no tool_call) → episode done
     At each tool-calling step, if an alternative granularity exists (atomic↔skill),
     fork a counterfactual branch with parameter-mapped arguments.
     Branch count is 0-2 → total group size is dynamic (2-4 rollouts).
  Phase 1.5: Compute R_imagination for each (base, branch) pair:
     Δ = R_branch - R_base → symmetric reward bonus scaled by gipo_step_reward_scale,
     clipped by gipo_step_reward_cap and gipo_total_reward_cap.
  Phase 2: Compute reward, group-normalize → advantage (positive = better than group mean)
  Phase 3: Policy gradient: loss = advantage × cross_entropy(assistant_tokens_only)
  Accumulate gradients for grad_accum prompts → optimizer.step()  ← this is 1 "step"

Reward:
  R = R_task + skill_bonus + R_efficiency + R_imagination
  R_task:        gt tools coverage (exact + fuzzy name matching)
  skill_bonus:   additive reward when skills genuinely helped
  R_eff:         bonus for fewer decision steps
  R_imagination: process reward from counterfactual granularity comparison (GIPO)
"""

import json
import copy
import logging
import os
import math
import random
import re
import time
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    RL_DATASET_PATH, AUGMENTED_TOOLS_PATH, SKILL_LIBRARY_PATH,
    GRPO_DATA_PATH, CHECKPOINT_DIR,
    ADAMACRO_OUTPUT_DIR, GRPOConfig, GIPOAPIConfig, get_model_path, DEFAULT_MODEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logs from openai/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# ============================================================================
# Execution Logger: saves detailed per-step records to JSON
# ============================================================================

class ExecutionLogger:
    """
    Saves detailed training execution records to a JSON-lines file.
    Each line = one prompt's complete rollout group.
    """
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.records = []
        # Write header
        with open(log_path, "w") as f:
            f.write("")  # clear
        logger.info(f"Execution log → {log_path}")

    def log_prompt(self, epoch: int, prompt_idx: int, global_step: int,
                   user_prompt: str, gt_tools: List[str],
                   rollouts: List[Dict]):
        """Log one prompt's complete training record."""
        record = {
            "epoch": epoch,
            "prompt_idx": prompt_idx,
            "global_step": global_step,
            "user_prompt": user_prompt[:200],
            "gt_tools": gt_tools,
            "rollouts": [],
        }
        for g, r in enumerate(rollouts):
            rollout_record = {
                "group_idx": g,
                "temperature": r.get("temperature", 0),
                "num_steps": r["num_steps"],
                "num_skill_calls": r["num_skill_calls"],
                "total_atomic": r["total_atomic"],
                "completed": r["completed"],
                "reward": round(r["reward"], 4),
                "advantage": round(r["advantage"], 4),
                "reward_breakdown": r.get("reward_breakdown", {}),
                # Per-turn action log
                "turns": [],
            }
            # Extract turn-by-turn actions from messages
            for i, (tool_name, args) in enumerate(r["actions"]):
                turn_info = {
                    "turn": i,
                    "tool_name": tool_name,
                    "arguments": _truncate_args(args),
                    "is_skill": tool_name in r.get("skill_names_used", []),
                }
                rollout_record["turns"].append(turn_info)

            # If model just output text without any tool call
            if r["num_steps"] == 0:
                # Get the final text response
                msgs = r.get("messages", [])
                for m in reversed(msgs):
                    if m.get("role") == "assistant":
                        rollout_record["final_text"] = m.get("content", "")[:200]
                        break

            record["rollouts"].append(rollout_record)

        self.records.append(record)

        # Append to file (JSON-lines format)
        # Clean surrogates from model output before serialization
        def _clean_surrogates(obj):
            if isinstance(obj, str):
                return obj.encode("utf-8", errors="replace").decode("utf-8")
            if isinstance(obj, dict):
                return {k: _clean_surrogates(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean_surrogates(v) for v in obj]
            return obj

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_clean_surrogates(record), ensure_ascii=False) + "\n")

    def save_summary(self):
        """Save a human-readable summary at the end."""
        summary_path = self.log_path.replace(".jsonl", "_summary.json")
        total = len(self.records)
        if total == 0:
            return

        avg_reward = sum(
            sum(r["reward"] for r in rec["rollouts"]) / len(rec["rollouts"])
            for rec in self.records
        ) / total

        avg_steps = sum(
            sum(r["num_steps"] for r in rec["rollouts"]) / len(rec["rollouts"])
            for rec in self.records
        ) / total

        skill_used = sum(
            1 for rec in self.records
            if any(r["num_skill_calls"] > 0 for r in rec["rollouts"])
        )

        summary = {
            "total_prompts": total,
            "avg_reward": round(avg_reward, 4),
            "avg_steps_per_rollout": round(avg_steps, 2),
            "prompts_with_skill_usage": skill_used,
            "skill_usage_rate": round(skill_used / total, 4) if total > 0 else 0,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Execution summary → {summary_path}")


def _truncate_args(args: Dict, max_len: int = 100) -> Dict:
    """Truncate arg values for readable logging."""
    if not isinstance(args, dict):
        return {}
    out = {}
    for k, v in args.items():
        s = str(v)
        out[k] = s[:max_len] + "..." if len(s) > max_len else s
    return out


# ============================================================================
# Reward
# ============================================================================

class AdaMacroReward:
    """
    R(τ) = R_task + skill_bonus + R_efficiency

    Core insight:
      R_task (gt_tools coverage) is THE task-completion proxy.
      skill_bonus is a small ADDITIVE reward when skills helped cover gt_tools.
      If r_task=0, no amount of skill usage helps — this prevents collapse.

    Gradient:
      high_coverage + skill(~1.5) > high_coverage + atomic(1.0) > low_coverage + skill(0.3) > nothing(0)

    The multiplier ensures:
      - "right skill, right task" → r_task is high AND gets multiplied → best reward
      - "wrong skill, any task"  → r_task stays low, multiplier on low base → low reward  
      - "right atomic, no skill" → r_task is high, no multiplier → good but not best
    """
    def __init__(self, config: GRPOConfig, skill_definitions: Dict = None):
        self.lambda_skill = config.lambda_skill    # 0.3
        self.skill_chains = {}
        if skill_definitions:
            for sname, sdef in skill_definitions.items():
                chain = sdef.get("tool_chain", [])
                if not chain:
                    chain = [s.get("tool_name", "") for s in sdef.get("execution_plan", [])]
                # Normalize chain tool names
                self.skill_chains[sname] = [normalize_tool_name(t) for t in chain]

    def compute(
        self,
        used_tools: List[str],
        gt_tools: List[str],
        skill_traces: List[List[Tuple[str, str]]],
        skill_names: List[str],
        num_decision_steps: int,
        num_skill_calls: int,
        total_atomic_cost: int,
        completed: bool,
        max_steps: int,
    ) -> Tuple[float, Dict]:

        # ==============================================================
        # R_task: tool overlap with ground truth (0 ~ 1.0)
        #
        # Matching levels:
        #   1. Exact match → 1.0 credit
        #   2. Substring match (after normalization) → 0.8 credit
        #   3. Token Jaccard ≥ 0.5 → jaccard_score credit
        # ==============================================================
        # Normalize all tool names to strip version suffixes
        all_used = set(normalize_tool_name(t) for t in used_tools)
        for trace in skill_traces:
            for tool_name, _ in trace:
                all_used.add(normalize_tool_name(tool_name))
        gt_set = set(normalize_tool_name(t) for t in gt_tools) if gt_tools else set()
        # Preserve ordered lists for position-aware scoring
        gt_list = [normalize_tool_name(t) for t in gt_tools] if gt_tools else []
        used_list = [normalize_tool_name(t) for t in used_tools]
        # Also append tools from skill traces in order
        for trace in skill_traces:
            for tool_name, _ in trace:
                used_list.append(normalize_tool_name(tool_name))

        def _tokenize_tool(name: str) -> set:
            """Split tool name into tokens for Jaccard comparison."""
            n = name.lower().replace("-", "_").replace(".", "_")
            # Remove version suffixes like _v1, _v2
            import re as _re
            n = _re.sub(r'_v\d+$', '', n)
            return set(t for t in n.split("_") if len(t) >= 2)

        def _fuzzy_match_score(a: str, b: str) -> float:
            """Return match score between two tool names (0~1)."""
            a_n = a.lower().replace("-", "_").replace(".", "_")
            b_n = b.lower().replace("-", "_").replace(".", "_")
            if a_n == b_n or a == b:
                return 1.0
            if a_n in b_n or b_n in a_n:
                return 0.8
            a_tokens = _tokenize_tool(a)
            b_tokens = _tokenize_tool(b)
            if a_tokens and b_tokens:
                inter = len(a_tokens & b_tokens)
                union = len(a_tokens | b_tokens)
                jaccard = inter / union if union > 0 else 0
                if jaccard >= 0.5:
                    return jaccard
            return 0.0

        if gt_set and all_used:
            total_credit = 0.0

            for gt in gt_set:
                best_score = 0.0
                for ut in all_used:
                    score = _fuzzy_match_score(gt, ut)
                    if score > best_score:
                        best_score = score
                    if best_score == 1.0:
                        break
                total_credit += best_score

            base_coverage = min(total_credit / max(len(gt_set), 1), 1.0)

            # ---- Position-aware order bonus ----
            # Build matched position pairs: for each gt tool (in order),
            # find its best match position in used_list
            gt_match_positions = []  # (gt_index, used_index, score)
            for gi, gt in enumerate(gt_list):
                best_score = 0.0
                best_ui = -1
                for ui, ut in enumerate(used_list):
                    score = _fuzzy_match_score(gt, ut)
                    if score > best_score:
                        best_score = score
                        best_ui = ui
                    if best_score == 1.0:
                        break
                if best_score > 0:
                    gt_match_positions.append((gi, best_ui, best_score))

            # Compute LCS length on the used_list positions to measure order preservation
            if len(gt_match_positions) >= 2:
                # Extract the used_list positions of matched gt tools (in gt order)
                pos_seq = [ui for _, ui, _ in gt_match_positions]
                # LCS of pos_seq with its sorted version = longest increasing subsequence
                # Use patience sorting for O(n log n) LIS
                import bisect
                tails = []
                for p in pos_seq:
                    idx = bisect.bisect_left(tails, p)
                    if idx == len(tails):
                        tails.append(p)
                    else:
                        tails[idx] = p
                lis_len = len(tails)
                order_bonus = lis_len / len(gt_match_positions)
            elif len(gt_match_positions) == 1:
                order_bonus = 1.0
            else:
                order_bonus = 0.0

            # Final r_task: coverage weighted by order quality
            # 70% from coverage, 30% bonus for correct ordering
            r_task = base_coverage * (0.7 + 0.3 * order_bonus)

            if r_task == 0.0 and completed:
                r_task = 0.1
        elif completed and all_used:
            r_task = 0.15
        elif completed:
            r_task = 0.0
        else:
            r_task = 0.0

        # 0-step penalty
        if num_decision_steps == 0:
            return 0.0, {
                "r_task": 0.0, "skill_bonus": 0.0, "r_efficiency": 0.0,
                "n_skill_ok": 0, "n_skill_relevant": 0, "exact_match": 0,
                "tools_used": [], "note": "0-step penalty",
            }

        # ==============================================================
        # Skill bonus: small ADDITIVE reward when skills genuinely helped
        #
        # Design: skill bonus is additive (not multiplicative) to avoid
        # compounding bias that causes skill-only collapse.
        # Max skill bonus ≈ +0.1, compared to r_task range [0, 1].
        # Irrelevant skill calls receive a penalty.
        # ==============================================================
        skill_bonus = 0.0
        n_skill_ok = 0
        n_skill_relevant = 0
        n_skill_irrelevant = 0

        if skill_traces and skill_names:
            # Track unique skills already counted (no stacking from repeated calls)
            seen_skills = set()

            for i, trace in enumerate(skill_traces):
                sname = skill_names[i] if i < len(skill_names) else ""

                # Skip duplicate skill calls — same skill called again adds nothing
                if sname in seen_skills:
                    continue

                chain = self.skill_chains.get(sname, [])

                # How many gt_tools does this skill's chain cover? (with fuzzy matching)
                chain_coverage = 0.0
                if gt_set and chain:
                    chain_credit = 0.0
                    for gt in gt_set:
                        gt_tokens = _tokenize_tool(gt)
                        for ct in chain:
                            ct_n = ct.lower().replace("-", "_").replace(".", "_")
                            gt_n = gt.lower().replace("-", "_").replace(".", "_")
                            if ct_n == gt_n or ct == gt:
                                chain_credit += 1.0
                                break
                            if ct_n in gt_n or gt_n in ct_n:
                                chain_credit += 0.8
                                break
                            ct_tokens = _tokenize_tool(ct)
                            if gt_tokens and ct_tokens:
                                inter = len(gt_tokens & ct_tokens)
                                union = len(gt_tokens | ct_tokens)
                                if union > 0 and inter / union >= 0.5:
                                    chain_credit += inter / union
                                    break
                    chain_coverage = chain_credit / max(len(gt_set), 1)

                all_ok = all(s == "success" for _, s in trace)

                if chain_coverage > 0 and all_ok:
                    # Additive bonus: max 0.1 per relevant skill (capped below)
                    this_bonus = 0.05 + 0.05 * chain_coverage
                    skill_bonus += this_bonus
                    n_skill_ok += 1
                    n_skill_relevant += 1
                    seen_skills.add(sname)
                elif chain_coverage > 0 and not all_ok:
                    passed = sum(1 for _, s in trace if s == "success")
                    frac = passed / max(len(trace), 1)
                    this_bonus = (0.05 + 0.05 * chain_coverage) * frac
                    skill_bonus += this_bonus
                    n_skill_relevant += 1
                    seen_skills.add(sname)
                elif chain_coverage == 0 and chain:
                    # PENALTY: skill chain has zero overlap with gt_tools
                    # Calling an irrelevant skill wastes steps and should be discouraged
                    skill_bonus -= 0.05
                    n_skill_irrelevant += 1
                    seen_skills.add(sname)

            # Cap total skill bonus to prevent stacking
            skill_bonus = max(-0.15, min(skill_bonus, 0.1))

            # Penalize repeated skill calls (same skill called N>1 times)
            # Count total skill calls vs unique skills
            if num_skill_calls > len(seen_skills) and len(seen_skills) > 0:
                repeat_count = num_skill_calls - len(seen_skills)
                skill_bonus -= 0.03 * repeat_count  # -0.03 per repeated call
                skill_bonus = max(skill_bonus, -0.15)

        # ==============================================================
        # R_efficiency: fewer decision steps (gated by coverage quality)
        # ==============================================================
        r_eff = 0.0
        if completed and num_decision_steps > 0:
            # Gate: only grant efficiency bonus when task is reasonably solved
            if r_task >= 0.4:
                r_eff += max(0, 1 - num_decision_steps / max_steps) * 0.15 * r_task
                # Compression bonus: ONLY when skills were actually used
                # Without this gate, early stopping gets free efficiency points
                if gt_tools and num_decision_steps < len(gt_tools) and num_skill_calls > 0:
                    r_eff += (len(gt_tools) - num_decision_steps) * 0.05

        # Under-exploration penalty: punish very short rollouts with low coverage
        if num_decision_steps < 3 and r_task < 0.3:
            r_eff -= 0.05 * (3 - num_decision_steps)

        # ==============================================================
        # Total: r_task + skill_bonus + efficiency + imagination (GIPO)
        # In GIPO, imagination signal comes from counterfactual branches
        # added directly to the GRPO group, not as a reward modifier.
        # ==============================================================
        total = r_task + skill_bonus + r_eff
        # Clamp to non-negative: negative rewards destabilize GRPO advantage
        total = max(total, 0.0)

        # Safely get order_bonus (may not be defined if gt_set is empty)
        _order_bonus = order_bonus if (gt_set and all_used) else 0.0

        breakdown = {
            "r_task": round(r_task, 4),
            "skill_bonus": round(skill_bonus, 4),
            "r_efficiency": round(r_eff, 4),
            "r_imagination": 0.0,  # placeholder; imagination signal via GRPO group
            "order_bonus": round(_order_bonus, 4),
            "n_skill_ok": n_skill_ok,
            "n_skill_relevant": n_skill_relevant,
            "n_skill_irrelevant": n_skill_irrelevant,
            "exact_match": int(len(all_used & gt_set)) if gt_set else 0,
            "tools_used": list(all_used)[:10],
        }
        return total, breakdown


# ============================================================================
# API Tool Environment (replaces ToolEnvironment for API-based simulation)
# ============================================================================

class APIToolEnvironment:
    """
    Uses an external LLM API to simulate tool execution.

    Instead of looking up pre-recorded outputs from tool_simulator_database,
    this environment asks a large language model to generate realistic tool
    outputs based on the tool name, arguments, and conversation context.

    This produces more realistic training signals:
    - Tools can return incorrect results (noise)
    - Tools can timeout (API timeout)
    - Tools usually return correct results given good context
    """

    def __init__(self, augmented_tools_path: str, api_config):
        from openai import OpenAI

        with open(augmented_tools_path, "r") as f:
            augmented_tools = json.load(f)

        self.skills = {t["name"]: t for t in augmented_tools if t.get("is_skill")}
        self.atomic_tools = {t["name"]: t for t in augmented_tools if not t.get("is_skill")}
        self.all_tool_names = set(t["name"] for t in augmented_tools)

        # Tool definition index for API prompt context
        self.tool_defs = {}
        for t in augmented_tools:
            self.tool_defs[t["name"]] = t
            self.tool_defs[normalize_tool_name(t["name"])] = t

        # Tool description string for system prompts (same as original)
        lines = []
        for t in augmented_tools[:60]:
            tag = "[SKILL]" if t.get("is_skill") else "[TOOL]"
            lines.append(f"{tag} {t['name']}: {t.get('description','')[:100]}")
        if len(augmented_tools) > 60:
            lines.append(f"... and {len(augmented_tools) - 60} more tools.")
        self.tool_desc = "\n".join(lines)

        # API client — matching api.py, with internal retry disabled
        try:
            self.client = OpenAI(
                api_key=api_config.api_key,
                base_url=api_config.api_base_url,
                max_retries=0,  # disable openai lib's internal retry
            )
        except TypeError:
            # Fallback for older openai versions that don't support max_retries
            self.client = OpenAI(
                api_key=api_config.api_key,
                base_url=api_config.api_base_url,
            )
        self.api_model = api_config.api_model
        self.api_max_retries = api_config.api_max_retries
        self.api_timeout = api_config.api_timeout

        logger.info(f"APIToolEnvironment: model={self.api_model} "
                    f"timeout={self.api_timeout}s retries={self.api_max_retries}")

    def _call_api(self, system_msg: str, user_msg: str) -> str:
        """Call external LLM API with timeout and retry."""
        combined = system_msg + "\n\n" + user_msg
        for attempt in range(self.api_max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.api_model,
                    messages=[{"role": "user", "content": combined}],
                    timeout=self.api_timeout,
                )
                content = response.choices[0].message.content
                if content is None:
                    return "{}"
                return content.strip()
            except Exception as e:
                if attempt == self.api_max_retries:
                    logger.warning(f"API call failed after {self.api_max_retries + 1} attempts: {e}")
                    return json.dumps({"error": "API timeout/error", "message": str(e)[:100]})
                time.sleep(2)

    def _simulate_atomic(self, tool_name: str, arguments: dict,
                         context_history: List[Tuple[str, str]] = None) -> str:
        """Simulate a single atomic tool call via API."""
        tool_def = (self.tool_defs.get(tool_name)
                    or self.tool_defs.get(normalize_tool_name(tool_name)))
        desc = tool_def.get("description", "No description") if tool_def else "Unknown tool"
        params_schema = tool_def.get("parameters_schema", {}) if tool_def else {}

        system_msg = (
            "You are a tool execution simulator. Generate realistic output for the given tool call.\n\n"
            "Rules:\n"
            "- Return ONLY the raw tool response (JSON or plain text), no explanations\n"
            "- Most calls (~85%) should return correct, helpful results\n"
            "- Some calls (~10%) may return partially correct or incomplete results\n"
            "- Rarely (~5%) return an error response like {\"error\": \"...\"}\n"
            "- Keep response CONCISE: under 300 characters. No verbose explanations.\n"
            "- Match the expected output format of the tool"
        )

        # Build context from previous tool calls
        ctx_str = ""
        if context_history:
            ctx_lines = []
            for prev_tool, prev_output in context_history[-3:]:
                ctx_lines.append(f"  - {prev_tool} → {prev_output[:150]}")
            ctx_str = "Previous tool calls in this session:\n" + "\n".join(ctx_lines) + "\n\n"

        user_msg = (
            f"Tool: {tool_name}\n"
            f"Description: {desc[:200]}\n"
            f"Parameters schema: {json.dumps(params_schema, ensure_ascii=False)[:300]}\n\n"
            f"{ctx_str}"
            f"Called with arguments:\n{json.dumps(arguments, ensure_ascii=False, indent=2)[:500]}\n\n"
            f"Generate the tool's output (CONCISE, under 300 chars):"
        )

        output = self._call_api(system_msg, user_msg)
        if len(output) > 500:
            output = output[:500]
        return output

    def _simulate_skill(self, skill_name: str, skill_def: dict, arguments: dict,
                        context_history: List[Tuple[str, str]] = None) -> dict:
        """Simulate an entire skill (tool chain) via a single API call."""
        chain = skill_def.get("tool_chain", [])
        if not chain:
            chain = [s.get("tool_name", "") for s in skill_def.get("execution_plan", [])]

        chain_desc = []
        for i, tool in enumerate(chain):
            td = (self.tool_defs.get(tool)
                  or self.tool_defs.get(normalize_tool_name(tool)))
            d = td.get("description", "")[:80] if td else ""
            chain_desc.append(f"  {i+1}. {tool}: {d}")

        system_msg = (
            "You are a tool execution simulator. Generate realistic output for a composite skill.\n"
            "A skill chains multiple tools in sequence. Generate the FINAL output after all steps.\n\n"
            "Rules:\n"
            "- Return ONLY the final output (JSON or plain text), no explanations\n"
            "- Most calls (~85%) should succeed with correct results\n"
            "- Some calls (~10%) may partially fail (one step in the chain errors)\n"
            "- Rarely (~5%) the entire chain fails\n"
            "- Keep response CONCISE: under 300 characters"
        )

        ctx_str = ""
        if context_history:
            ctx_lines = []
            for prev_tool, prev_output in context_history[-3:]:
                ctx_lines.append(f"  - {prev_tool} → {prev_output[:150]}")
            ctx_str = "Previous tool calls in this session:\n" + "\n".join(ctx_lines) + "\n\n"

        user_msg = (
            f"Skill: {skill_name}\n"
            f"Description: {skill_def.get('description', '')[:200]}\n"
            f"Tool chain:\n" + "\n".join(chain_desc) + "\n\n"
            f"{ctx_str}"
            f"Called with arguments:\n{json.dumps(arguments, ensure_ascii=False, indent=2)[:500]}\n\n"
            f"Generate the final output (CONCISE, under 300 chars):"
        )

        output = self._call_api(system_msg, user_msg)
        if len(output) > 500:
            output = output[:500]

        has_error = "error" in output.lower()[:50]
        trace = [(tool, "error" if has_error else "success") for tool in chain]

        return {
            "output": output,
            "is_skill": True,
            "trace": trace,
            "success": not has_error,
            "interrupted": has_error,
            "atomic_cost": max(len(chain), 1),
        }

    def execute(self, tool_name: str, arguments: dict,
                context_history: List[Tuple[str, str]] = None) -> dict:
        """
        Unified entry point. Returns same format as original ToolEnvironment.execute().

        Args:
            context_history: list of (tool_name, output) from previous calls in this
                             rollout, so the API can generate contextually consistent outputs.
        """
        if not isinstance(arguments, dict):
            arguments = {}

        # Name resolution (same as original)
        resolved_name = tool_name
        if (tool_name not in self.skills and tool_name not in self.atomic_tools
                and tool_name not in self.all_tool_names):
            norm = normalize_tool_name(tool_name)
            for orig in self.all_tool_names:
                if normalize_tool_name(orig) == norm:
                    resolved_name = orig
                    break

        is_known = (resolved_name in self.skills or resolved_name in self.atomic_tools
                    or resolved_name in self.all_tool_names)

        if resolved_name in self.skills:
            return self._simulate_skill(resolved_name, self.skills[resolved_name],
                                        arguments, context_history)
        elif is_known:
            output = self._simulate_atomic(resolved_name, arguments, context_history)
            ok = output and "error" not in output.lower()[:50]
            return {
                "output": output or "",
                "is_skill": False,
                "trace": [(tool_name, "success" if ok else "error")],
                "success": ok,
                "interrupted": False,
                "atomic_cost": 1,
            }
        else:
            return {
                "output": json.dumps({"error": f"Unknown tool: {tool_name}"}),
                "is_skill": False,
                "trace": [(tool_name, "unknown_tool")],
                "success": False,
                "interrupted": False,
                "atomic_cost": 1,
            }


# ============================================================================
# Action parsing
# ============================================================================

def normalize_tool_name(name: str) -> str:
    """Strip version suffixes like _v1, _v13, _v2beta from tool names.
    
    The version suffix is only relevant for DB parameter lookup, not for
    the model's tool selection. Keeping it increases vocabulary and makes
    generalization harder.
    
    Examples:
        excel-format_range_v13 → excel-format_range
        google_maps_geocode_v1 → google_maps_geocode
        canvas-canvas_create_quiz → canvas-canvas_create_quiz (no change)
    """
    import re as _re
    return _re.sub(r'_v\d+\w*$', '', name)


def find_best_matching_skill(gt_tools: List[str], skills: Dict[str, Dict]) -> Optional[str]:
    """Find the skill whose tool_chain has maximum overlap with gt_tools.

    Returns the skill name with the highest coverage (at least 1 overlapping
    tool), or None if no skill matches any gt_tool.
    """
    if not gt_tools or not skills:
        return None

    gt_set = set(normalize_tool_name(t) for t in gt_tools)
    best_skill = None
    best_coverage = 0

    for sname, sdef in skills.items():
        chain = sdef.get("tool_chain", [])
        if not chain:
            chain = [s.get("tool_name", "") for s in sdef.get("execution_plan", [])]
        chain_norm = set(normalize_tool_name(t) for t in chain if t)
        overlap = len(chain_norm & gt_set)
        if overlap > best_coverage:
            best_coverage = overlap
            best_skill = sname

    return best_skill if best_coverage > 0 else None


# ============================================================================
# GIPO: Counterfactual Granularity Imagination
# ============================================================================

def find_counterfactual_action(
    chosen_tool: str,
    is_skill: bool,
    skills: Dict[str, Dict],
    skill_chains: Dict[str, List[str]],
    original_arguments: Dict = None,
) -> Optional[Dict]:
    """
    Given the model's chosen action, find the counterfactual at a different
    granularity level, with proper parameter mapping.

    - If model chose an atomic tool → find a skill whose chain contains it
    - If model chose a skill → return the first atomic tool in its chain

    Parameter mapping ensures the counterfactual gets meaningful arguments
    instead of empty {}, making the comparison fair.

    Returns: {"name": str, "arguments": dict, "is_skill": bool, "reason": str}
             or None if no counterfactual exists.
    """
    chosen_norm = normalize_tool_name(chosen_tool)
    orig_args = original_arguments or {}

    if is_skill:
        # Model chose a skill → counterfactual is its first atomic tool
        chain = skill_chains.get(chosen_tool, [])
        skill_key = chosen_tool
        if not chain:
            # Try normalized name lookup
            for sname, schain in skill_chains.items():
                if normalize_tool_name(sname) == chosen_norm and schain:
                    chain = schain
                    skill_key = sname
                    break
        if chain:
            # Map skill's exposed params → first atomic tool's params
            # execution_plan[i]["params_source"][param] = {"type": "exposed", "param_name": X}
            cf_args = {}
            skill_def = skills.get(skill_key)
            if skill_def and orig_args:
                exec_plan = skill_def.get("execution_plan", [])
                if exec_plan:
                    first_step_params = exec_plan[0].get("params_source", {})
                    for pname, pinfo in first_step_params.items():
                        if pinfo.get("type") == "exposed":
                            exposed_name = pinfo.get("param_name", pname)
                            if exposed_name in orig_args:
                                cf_args[pname] = orig_args[exposed_name]
                # Fallback: if exec_plan didn't help, try direct name match
                if not cf_args:
                    for k, v in orig_args.items():
                        cf_args[k] = v
            return {
                "name": chain[0],
                "arguments": cf_args,
                "is_skill": False,
                "reason": f"atomic_alternative_for_{chosen_norm}",
            }
    else:
        # Model chose atomic tool → find a skill containing it
        best_skill = None
        best_chain_len = float("inf")
        for sname, schain in skill_chains.items():
            chain_norm = [normalize_tool_name(t) for t in schain]
            if chosen_norm in chain_norm:
                # Prefer shorter chains (more specific skills)
                if len(schain) < best_chain_len:
                    best_chain_len = len(schain)
                    best_skill = sname
        if best_skill:
            # Map atomic tool's params → skill's exposed params
            # execution_plan[i]["params_source"][param] = {"type": "exposed", "param_name": X}
            cf_args = {}
            skill_def = skills.get(best_skill)
            if skill_def and orig_args:
                exec_plan = skill_def.get("execution_plan", [])
                chain = skill_chains.get(best_skill, [])
                # Find chosen_tool's position in the chain
                chosen_idx = None
                for ci, ct in enumerate(chain):
                    if normalize_tool_name(ct) == chosen_norm:
                        chosen_idx = ci
                        break
                if chosen_idx is not None and chosen_idx < len(exec_plan):
                    step_params = exec_plan[chosen_idx].get("params_source", {})
                    for pname, pinfo in step_params.items():
                        if pinfo.get("type") == "exposed":
                            exposed_name = pinfo.get("param_name", pname)
                            if pname in orig_args:
                                cf_args[exposed_name] = orig_args[pname]
                # Fallback: if mapping didn't help, try direct name match
                if not cf_args:
                    exposed = skill_def.get("exposed_params", [])
                    for k, v in orig_args.items():
                        if k in exposed:
                            cf_args[k] = v
            return {
                "name": best_skill,
                "arguments": cf_args,
                "is_skill": True,
                "reason": f"skill_alternative_for_{chosen_norm}",
            }

    return None


def run_imagination_branch(
    model, tokenizer, env: "APIToolEnvironment",
    prefix_messages: List[Dict],
    prefix_actions: List[Tuple[str, Dict]],
    cf_tool_name: str,
    cf_arguments: Dict,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    device,
    gt_tools_len: int = 0,
    prefix_total_atomic: int = 0,
    prefix_skill_traces: List[List[Tuple[str, str]]] = None,
    prefix_skill_names: List[str] = None,
) -> Dict:
    """
    Run a counterfactual imagination branch from a decision point.

    Takes the message history up to (but not including) the branching step,
    executes the counterfactual tool, then lets the model continue generating
    from the modified history to produce a complete alternative trajectory.

    This produces a full rollout whose reward can be compared with the original
    to determine if the model's granularity choice was correct.

    Returns: same format as run_rollout (with messages, actions, reward fields etc.)
    """
    import torch

    TOOL_CALL_PREFIX = '<tool_call>\n{"name": "'

    # Extract available tool names from system prompt (first message)
    _available_tools = []
    if prefix_messages and prefix_messages[0]["role"] == "system":
        for line in prefix_messages[0]["content"].split("\n"):
            line = line.strip()
            if line.startswith("- ") and ":" in line:
                tname = line[2:].split(":")[0].strip()
                if tname.startswith("[SKILL]"):
                    tname = tname[7:]
                if tname and len(tname) < 60:
                    _available_tools.append(tname)
    _available_tools_set = set(_available_tools)  # for O(1) lookup in salvage

    # Start from prefix (messages before the branching step)
    # Inherit skill info from prefix to ensure fair reward comparison
    messages = [dict(m) for m in prefix_messages]
    actions = list(prefix_actions)
    skill_traces = list(prefix_skill_traces) if prefix_skill_traces else []
    skill_names_used = list(prefix_skill_names) if prefix_skill_names else []
    total_atomic = prefix_total_atomic if prefix_total_atomic > 0 else len(actions)
    completed = False

    # Build call_history from prefix messages (extract previous tool responses)
    call_history = []
    for m in prefix_messages:
        if m.get("role") == "user" and "<tool_response" in m.get("content", ""):
            content = m["content"]
            name_match = re.search(r'name="([^"]+)"', content)
            if name_match:
                prev_name = name_match.group(1)
                resp_match = re.search(r'<tool_response[^>]*>\n?(.*?)\n?</tool_response>',
                                       content, re.DOTALL)
                prev_out = resp_match.group(1)[:200] if resp_match else ""
                call_history.append((prev_name, prev_out))

    # Execute the counterfactual tool as the branching action
    if len(cf_tool_name) > 60 or " " in cf_tool_name:
        return _empty_branch_result(messages, actions, temperature,
                                    prefix_skill_traces=skill_traces,
                                    prefix_skill_names=skill_names_used,
                                    prefix_total_atomic=total_atomic)

    result = env.execute(cf_tool_name, cf_arguments, context_history=call_history)
    actions.append((cf_tool_name, cf_arguments))
    total_atomic += result["atomic_cost"]

    if result["is_skill"]:
        skill_traces.append(result["trace"])
        skill_names_used.append(cf_tool_name)

    tc_json = json.dumps({"name": cf_tool_name, "arguments": cf_arguments}, ensure_ascii=False)
    messages.append({"role": "assistant", "content": f"<tool_call>\n{tc_json}\n</tool_call>"})
    obs = str(result["output"])[:1500]
    messages.append({"role": "user",
                     "content": f"<tool_response name=\"{cf_tool_name}\">\n{obs}\n</tool_response>"})
    call_history.append((cf_tool_name, str(result["output"])[:200]))

    # Continue generating from the modified history
    min_forced_steps = min(max(3, gt_tools_len // 2), 6) if gt_tools_len > 0 else 3
    remaining_turns = max_turns - len(actions)

    for turn in range(remaining_turns):
        formatted = [{"role": m["role"], "content": m.get("content", "") or ""} for m in messages]
        try:
            prompt = tokenizer.apply_chat_template(formatted, tokenize=False, add_generation_prompt=True)
        except Exception:
            parts = [f"<|{m['role']}|>\n{m.get('content', '')}" for m in formatted]
            prompt = "\n".join(parts) + "\n<|assistant|>\n"

        if not isinstance(prompt, str) or not prompt.strip():
            break
        if len(prompt) > 50000:
            prompt = prompt[:50000]

        # Use same forced/free logic as run_rollout
        if len(actions) < min_forced_steps:
            forced_prompt = prompt + TOOL_CALL_PREFIX
            try:
                inputs = tokenizer(forced_prompt, return_tensors="pt", truncation=True, max_length=3584)
            except (TypeError, ValueError):
                break
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True, temperature=temperature, top_p=0.95,
                )
            gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
            completion = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            if not completion and _available_tools:
                import random as _rnd
                completion = f'{_rnd.choice(_available_tools)}", "arguments": {{}}}}\n</tool_call>'

            full_response = TOOL_CALL_PREFIX + completion
            tool_call = parse_tool_call(full_response)

            if tool_call is None:
                name_match = re.match(r'^([a-zA-Z][\w\-\.]*)', completion)
                if name_match and _available_tools_set:
                    candidate = name_match.group(1)
                    if candidate in _available_tools_set:
                        tool_call = {"name": candidate, "arguments": {}}
                if tool_call is None and _available_tools:
                    import random as _rnd
                    tool_call = {"name": _rnd.choice(_available_tools), "arguments": {}}
                elif tool_call is None:
                    break
        else:
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3584)
            except (TypeError, ValueError):
                completed = True
                break
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True, temperature=temperature, top_p=0.95,
                )
            gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            tool_call = parse_tool_call(response)

            if tool_call is None:
                messages.append({"role": "assistant", "content": response})
                completed = True
                break

        # Execute tool
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]
        if not isinstance(arguments, dict):
            arguments = {}

        if len(tool_name) > 60 or " " in tool_name or tool_name.startswith("http"):
            completed = len(actions) > 0
            break

        result = env.execute(tool_name, arguments, context_history=call_history)
        actions.append((tool_name, arguments))
        total_atomic += result["atomic_cost"]

        if result["is_skill"]:
            skill_traces.append(result["trace"])
            skill_names_used.append(tool_name)

        tc_json = json.dumps({"name": tool_name, "arguments": arguments}, ensure_ascii=False)
        messages.append({"role": "assistant", "content": f"<tool_call>\n{tc_json}\n</tool_call>"})
        obs = str(result["output"])[:1500]
        messages.append({"role": "user",
                         "content": f"<tool_response name=\"{tool_name}\">\n{obs}\n</tool_response>"})
        call_history.append((tool_name, str(result["output"])[:200]))

        if total_atomic >= 30:
            break

    return {
        "messages": messages,
        "actions": actions,
        "skill_traces": skill_traces,
        "skill_names_used": skill_names_used,
        "num_steps": len(actions),
        "num_skill_calls": len(skill_traces),
        "total_atomic": total_atomic,
        "completed": completed,
        "temperature": temperature,
        "is_imagination_branch": True,
    }


def _empty_branch_result(messages, actions, temperature,
                         prefix_skill_traces=None, prefix_skill_names=None,
                         prefix_total_atomic=0):
    """Return a minimal result when branch cannot be executed."""
    _traces = list(prefix_skill_traces) if prefix_skill_traces else []
    _names = list(prefix_skill_names) if prefix_skill_names else []
    return {
        "messages": messages,
        "actions": actions,
        "skill_traces": _traces,
        "skill_names_used": _names,
        "num_steps": len(actions),
        "num_skill_calls": len(_traces),
        "total_atomic": max(prefix_total_atomic, len(actions)),
        "completed": False,
        "temperature": temperature,
        "is_imagination_branch": True,
    }


def _extract_balanced_json(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from text (handles nested braces)."""
    start = text.find('{')
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        if depth == 0:
            return text[start:i + 1]
    return None


def parse_tool_call(response: str) -> Optional[Dict]:
    # Priority 1: <tool_call> tags (handles nested JSON correctly)
    tc = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', response, re.DOTALL)
    if tc:
        try:
            obj = json.loads(tc.group(1))
            if isinstance(obj, dict) and "name" in obj:
                return {"name": str(obj["name"]), "arguments": obj.get("arguments", {})}
        except (json.JSONDecodeError, TypeError, AttributeError):
            balanced = _extract_balanced_json(tc.group(1))
            if balanced:
                try:
                    obj = json.loads(balanced)
                    if isinstance(obj, dict) and "name" in obj:
                        return {"name": str(obj["name"]), "arguments": obj.get("arguments", {})}
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

    # Priority 2: find JSON with "name" key using balanced brace matching
    name_pos = re.search(r'\{\s*"name"\s*:', response)
    if name_pos:
        balanced = _extract_balanced_json(response[name_pos.start():])
        if balanced:
            try:
                obj = json.loads(balanced)
                if isinstance(obj, dict) and "name" in obj:
                    return {"name": str(obj["name"]), "arguments": obj.get("arguments", obj.get("parameters", {}))}
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

    # Priority 3: entire response is JSON
    try:
        obj = json.loads(response.strip())
        if isinstance(obj, dict) and "name" in obj:
            return {"name": str(obj["name"]), "arguments": obj.get("arguments", {})}
    except:
        pass

    # Priority 4: tool_name\n{args} pattern
    fn = re.search(r'(\w[\w\-\.]+)\s*\n\s*(\{.*?\})', response, re.DOTALL)
    if fn:
        try:
            args = json.loads(fn.group(2))
            if isinstance(args, dict):
                return {"name": fn.group(1), "arguments": args}
        except:
            pass
    return None


# ============================================================================
# Single Rollout
# ============================================================================

def run_rollout(
    model, tokenizer, env: APIToolEnvironment,
    system_prompt: str, user_prompt: str,
    max_turns: int, max_new_tokens: int,
    temperature: float, device,
    oracle_first_tool: str = None,
    gt_tools_len: int = 0,
) -> Dict:
    """
    Run a single rollout episode with GIPO counterfactual imagination.

    Strategy: "free first, force if stuck"
    - Every turn: let model generate freely (it may output tool_call or text)
    - If model outputs tool_call → execute it, continue
    - If model outputs text AND has ≥1 prior action → treat as task done (natural stop)
    - If model outputs text AND has 0 actions → use forced prefix to guarantee ≥1 tool call

    GIPO addition: after each tool execution, check if a counterfactual action
    at a different granularity exists. If so, simulate it and compute a per-step
    process reward based on gt_tools coverage difference.
    """
    import torch
    if not hasattr(run_rollout, '_debug_count'):
        run_rollout._debug_count = 0

    TOOL_CALL_PREFIX = '<tool_call>\n{"name": "'

    # Extract available tool names from system prompt for ultimate fallback
    _available_tools = []
    for line in system_prompt.split("\n"):
        line = line.strip()
        if line.startswith("- ") and ":" in line:
            tname = line[2:].split(":")[0].strip()
            if tname.startswith("[SKILL]"):
                tname = tname[7:]
            if tname and len(tname) < 60:
                _available_tools.append(tname)
    _available_tools_set = set(_available_tools)  # for O(1) lookup in salvage

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    actions = []
    skill_traces = []
    skill_names_used = []
    call_history = []  # API context: (tool_name, output) for each previous call
    continued_this_rollout = False  # at most 1 continuation nudge per rollout
    total_atomic = 0
    completed = False

    # GIPO: track message count before each action for branching
    action_msg_offsets = []  # len(messages) right before each tool execution

    for turn in range(max_turns):
        formatted = [{"role": m["role"], "content": m.get("content", "") or ""} for m in messages]
        try:
            prompt = tokenizer.apply_chat_template(formatted, tokenize=False, add_generation_prompt=True)
        except Exception:
            parts = [f"<|{m['role']}|>\n{m.get('content', '')}" for m in formatted]
            prompt = "\n".join(parts) + "\n<|assistant|>\n"

        if not isinstance(prompt, str) or not prompt.strip():
            completed = False
            break

        if len(prompt) > 50000:
            prompt = prompt[:50000]

        # ==============================================================
        # Turn routing:
        #   actions < 2: forced prefix → must output tool
        #   actions ≥ 2: free generation → can choose to stop
        #
        # This ensures every rollout tries at least 2 tools before
        # deciding to stop, preventing the "1-tool-then-text" collapse.
        # ==============================================================
        min_forced_steps = min(max(3, gt_tools_len // 2), 6) if gt_tools_len > 0 else 3

        if len(actions) < min_forced_steps:
            # --- FORCED TURN: must output a tool ---

            # Oracle mode on first turn only
            if oracle_first_tool is not None and len(actions) == 0:
                tool_call = {"name": oracle_first_tool, "arguments": {}}
                if run_rollout._debug_count < 20:
                    logger.info(f"    [oracle_seed] forced first tool: {oracle_first_tool}")
            else:
                # Forced prefix generation
                forced_prompt = prompt + TOOL_CALL_PREFIX
                try:
                    inputs = tokenizer(forced_prompt, return_tensors="pt", truncation=True, max_length=3584)
                except (TypeError, ValueError):
                    completed = False
                    break
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.95,
                    )
                gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
                completion = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                # Fallback if empty
                if not completion and _available_tools:
                    import random as _rnd
                    completion = f'{_rnd.choice(_available_tools)}", "arguments": {{}}}}\n</tool_call>'

                if run_rollout._debug_count < 20:
                    logger.info(f"    [t{turn}_forced] completion({len(completion)}): {repr(completion[:120])}")
                    run_rollout._debug_count += 1

                full_response = TOOL_CALL_PREFIX + completion
                tool_call = parse_tool_call(full_response)

                # Salvage: only accept if the extracted name is a known tool
                if tool_call is None:
                    name_match = re.match(r'^([a-zA-Z][\w\-\.]*)', completion)
                    if name_match and _available_tools_set:
                        candidate = name_match.group(1)
                        # Only use the name if it's actually a known tool
                        if candidate in _available_tools_set:
                            tool_call = {"name": candidate, "arguments": {}}
                    if tool_call is None and _available_tools:
                        import random as _rnd
                        tool_call = {"name": _rnd.choice(_available_tools), "arguments": {}}
                    elif tool_call is None:
                        completed = False
                        break

        else:
            # --- SUBSEQUENT TURNS: free generation ---
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3584)
            except (TypeError, ValueError):
                completed = True
                break
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                )
            gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            tool_call = parse_tool_call(response)

            if tool_call is None:
                # Model wants to stop — but check if it stopped too early
                suggested_min = min_forced_steps + 1
                if (len(actions) < suggested_min
                        and turn < max_turns - 1
                        and not continued_this_rollout):
                    # Continuation nudge: tell the model to keep going
                    continued_this_rollout = True
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": "You have not completed the task yet. Continue calling tools to finish."
                    })
                    continue  # next iteration; forced/free branch will handle it
                else:
                    # Natural completion
                    messages.append({"role": "assistant", "content": response})
                    completed = True
                    break

        # ==============================================================
        # Step 3: Execute the tool call
        # ==============================================================
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]
        if not isinstance(arguments, dict):
            arguments = {}

        # Filter nonsense tool names
        if len(tool_name) > 60 or " " in tool_name or tool_name.startswith("http"):
            if len(actions) == 0:
                completed = False
            else:
                completed = True
            break

        # GIPO: record message offset before this action (for branching)
        action_msg_offsets.append(len(messages))

        result = env.execute(tool_name, arguments, context_history=call_history)
        actions.append((tool_name, arguments))
        total_atomic += result["atomic_cost"]

        if result["is_skill"]:
            skill_traces.append(result["trace"])
            skill_names_used.append(tool_name)

        tc_json = json.dumps({"name": tool_name, "arguments": arguments}, ensure_ascii=False)
        messages.append({"role": "assistant", "content": f"<tool_call>\n{tc_json}\n</tool_call>"})
        obs = str(result["output"])[:1500]
        messages.append({"role": "user",
                         "content": f"<tool_response name=\"{tool_name}\">\n{obs}\n</tool_response>"})
        call_history.append((tool_name, str(result["output"])[:200]))

        if total_atomic >= 30:
            break

    return {
        "messages": messages,
        "actions": actions,
        "skill_traces": skill_traces,
        "skill_names_used": skill_names_used,
        "num_steps": len(actions),
        "num_skill_calls": len(skill_traces),
        "total_atomic": total_atomic,
        "completed": completed,
        "temperature": temperature,
        # GIPO: message offsets for each action (used by imagination branching)
        "action_msg_offsets": action_msg_offsets,
    }


# ============================================================================
# Tokenize with assistant-only label mask
# ============================================================================

def tokenize_with_assistant_mask(messages, tokenizer, max_length=4096):
    import torch
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

    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    patterns = [
        r'<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>)',
        r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)(?:<\|eot_id\|>)',
    ]
    regions = []
    for pat in patterns:
        for m in re.finditer(pat, decoded, re.DOTALL):
            regions.append((m.start(1), m.end(1)))

    if not regions:
        labels = input_ids.clone()
        return input_ids, attention_mask, labels

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

    return input_ids, attention_mask, labels


# ============================================================================
# Online GRPO Training
# ============================================================================

def train_grpo(
    model_name: str,
    sft_checkpoint_dir: str,
    grpo_data_path: str,
    output_dir: str,
    grpo_config: "GIPOAPIConfig",
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType

    # ---------------------------------------------------------------- model
    model_path = get_model_path(model_name)
    logger.info(f"Loading base model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if os.path.exists(os.path.join(sft_checkpoint_dir, "adapter_config.json")):
        logger.info(f"Loading SFT LoRA from {sft_checkpoint_dir}")
        base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                    device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, sft_checkpoint_dir)
        model = model.merge_and_unload()
    else:
        logger.info("No SFT checkpoint; starting from base model")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                     device_map="auto", trust_remote_code=True)
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=grpo_config.lora_rank, lora_alpha=grpo_config.lora_alpha,
        lora_dropout=grpo_config.lora_dropout,
        target_modules=grpo_config.lora_target_modules, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    device = next(model.parameters()).device

    # ---------------------------------------------------------------- env
    env = APIToolEnvironment(AUGMENTED_TOOLS_PATH, grpo_config)
    reward_fn = AdaMacroReward(grpo_config, skill_definitions=env.skills)

    # Build tool description matching SFT format (with parameters)
    with open(AUGMENTED_TOOLS_PATH, "r") as f:
        all_augmented_tools = json.load(f)

    tool_desc_map = {}  # tool_name → description string
    # Also build a mapping from normalized name → original name for DB lookup
    norm_to_orig = {}  # normalized_name → original_name (for execute)
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

        # For skills: show the tool chain so model knows what's inside
        chain_str = ""
        if t.get("is_skill"):
            chain = t.get("tool_chain", [])
            if not chain:
                chain = [s.get("tool_name", "") for s in t.get("execution_plan", [])]
            if chain:
                norm_chain = [normalize_tool_name(c) for c in chain[:5]]
                chain_str = f" [chain: {' → '.join(norm_chain)}]"

        # Use normalized name in prompt (model sees clean names)
        tool_desc_map[norm_name] = f"- {tag}{norm_name}: {t.get('description','')[:100]}{chain_str}{param_str}"
        # Keep mapping to original for DB lookup
        if norm_name not in norm_to_orig:
            norm_to_orig[norm_name] = orig_name

    def build_system_prompt(task_name: str) -> str:
        """
        Build per-prompt system prompt.
        
        NO gt_tools leak: we use task_name to pick a relevant tool CATEGORY
        (e.g., "filesystem", "web") rather than the exact gt tools.
        This avoids train/eval distribution mismatch.
        """
        lines = []
        seen = set()

        # Infer tool category from task_name (e.g., "filesystem_read_task" → "filesystem")
        task_lower = task_name.lower().replace("-", "_")
        category_keywords = set()
        for part in task_lower.split("_"):
            if len(part) >= 3:
                category_keywords.add(part)

        # Add skills first (always visible, core to AdaMacro)
        # env.skills keys are original names, tool_desc_map keys are normalized
        norm_skill_names = set(normalize_tool_name(s) for s in env.skills)
        for tn, desc in tool_desc_map.items():
            if tn in norm_skill_names:
                lines.append(desc)
                seen.add(tn)
            if len(seen) >= 15:
                break

        # Add tools whose name matches task category keywords
        for tn, desc in tool_desc_map.items():
            if tn in seen:
                continue
            tn_lower = tn.lower().replace("-", "_")
            if any(kw in tn_lower for kw in category_keywords):
                lines.append(desc)
                seen.add(tn)
            if len(seen) >= 40:
                break

        # Fill remaining with random other tools
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
            # Normalize tool names (strip _v13 etc.) and deduplicate
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

    logger.info(f"GIPO-API Training  G={G}  epochs={num_epochs}  lr={lr}  "
                f"grad_accum={grad_accum}  total_steps≈{total_steps}  "
                f"(API-based tool simulation)")
    logger.info(f"GIPO params: img_scale={grpo_config.gipo_step_reward_scale} "
                f"step_cap={grpo_config.gipo_step_reward_cap} "
                f"total_cap={grpo_config.gipo_total_reward_cap}")
    logger.info(f"API config: model={grpo_config.api_model} "
                f"timeout={grpo_config.api_timeout}s "
                f"base_url={grpo_config.api_base_url[:50]}")

    # ---------------------------------------------------------------- training loop
    global_step = 0
    acc_loss = acc_reward = acc_steps_ep = acc_skill_r = 0.0
    acc_img_steps = 0.0  # GIPO: count of imagination branches
    acc_cnt = 0
    best_reward = -1e9

    for epoch in range(num_epochs):
        random.shuffle(train_prompts)
        logger.info(f"\n{'='*60}\nEpoch {epoch+1}/{num_epochs}\n{'='*60}")

        for pidx, pdata in enumerate(train_prompts):
            user_prompt = pdata["user_prompt"]
            gt_tools = pdata["gt_tools"]

            # ==========================================================
            # GIPO Phase 1: Generate 2 base rollouts + 0-2 branches
            #
            #   base_0 (skill-biased)   → branch_0 (flipped granularity)?
            #   base_1 (oracle-seeded)  → branch_1 (flipped granularity)?
            #
            # Each (base, branch) pair shares the same prefix and only
            # differs at one granularity decision → controlled comparison.
            # If no counterfactual exists for a base, skip — no fallback.
            # Group size is 2-4; advantage is computed over whatever exists.
            # ==========================================================
            model.eval()
            rollouts = []

            system_prompt = build_system_prompt(pdata["task_name"])

            # Build skill-biased variant for base_0 diversity
            # Append directive instead of fragile string replace
            skill_biased_prompt = system_prompt + (
                "\n\nIMPORTANT: You SHOULD prefer [SKILL] tools over atomic tools when possible. "
                "Skills chain multiple steps and are more efficient. "
                "Check the [SKILL] entries in the tool list first."
            )

            # Build skill chains for counterfactual lookup
            _skill_chains = {}
            for sname, sdef in env.skills.items():
                chain = sdef.get("tool_chain", [])
                if not chain:
                    chain = [s.get("tool_name", "") for s in sdef.get("execution_plan", [])]
                _skill_chains[sname] = [normalize_tool_name(t) for t in chain]

            # Helper: compute reward for a rollout
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

            # --- Base rollout 0: skill-biased prompt for diversity ---
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

            # --- Base rollout 1: oracle-seeded or higher temperature ---
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

            # --- 0-step resample: if both bases are 0-step, retry ---
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
                    if pidx < 5:
                        logger.info(f"    [resample] attempt {resample_attempts}: "
                                    f"steps={retry_ro['num_steps']} R={retry_ro['reward']:.3f}")

            # --- GIPO: generate counterfactual branches for each base ---
            # For each base rollout, find the first step with an alternative
            # granularity. Fork from that point and run to completion.
            # If no counterfactual exists, skip — don't pad with fallbacks
            # that would dilute the imagination signal.
            n_branches = 0
            for ri, base_ro in enumerate(list(rollouts)):  # iterate over copy
                if base_ro["num_steps"] == 0:
                    # Can't branch a 0-step rollout → skip
                    continue

                # Find first eligible branching point
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
                    # Fork from branch_step with counterfactual action
                    msg_offset = offsets[branch_step]
                    prefix_messages = base_ro["messages"][:msg_offset]
                    prefix_actions = actions[:branch_step]

                    # Compute prefix atomic cost and collect prefix skill info
                    _prefix_atomic = 0
                    _prefix_skill_traces = []
                    _prefix_skill_names = []
                    for _pa_name, _ in prefix_actions:
                        _pa_norm = normalize_tool_name(_pa_name)
                        # Check if this prefix action was a skill (try original + normalized name)
                        _is_prefix_skill = (_pa_name in env.skills or any(
                            normalize_tool_name(s) == _pa_norm for s in env.skills
                        ))
                        if _is_prefix_skill:
                            # Find matching skill def for chain length
                            _sk_def = env.skills.get(_pa_name)
                            if not _sk_def:
                                for _sn, _sd in env.skills.items():
                                    if normalize_tool_name(_sn) == _pa_norm:
                                        _sk_def = _sd
                                        break
                            if _sk_def:
                                # Use actual trace length from base rollout (may be shorter if interrupted)
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
                                    # Fallback to definition chain length
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
                # else: no counterfactual found → skip, don't pad with fallback

            # rollouts has 2-4 entries: [base_0, base_1, (branch_0)?, (branch_1)?]
            # Only base-branch pairs carry imagination signal.

            # ========== GIPO Phase 1.5: Imagination Reward ==========
            # For each (base, branch) pair, compute a symmetric reward bonus
            # that directly encodes the granularity comparison signal.
            # Δ > 0 means branch (alternative granularity) was better → base penalized, branch rewarded.
            # Δ < 0 means base (original choice) was better → base rewarded, branch penalized.
            img_scale = grpo_config.gipo_step_reward_scale  # 0.15
            img_step_cap = grpo_config.gipo_step_reward_cap  # 0.1  per-branch cap
            img_total_cap = grpo_config.gipo_total_reward_cap  # 0.3  per-rollout cap

            for ro in rollouts:
                if ro.get("rollout_type") != "branch":
                    continue
                binfo = ro.get("branch_info", {})
                parent_idx = binfo.get("parent_idx")
                if parent_idx is None or parent_idx >= len(rollouts):
                    continue
                base_ro = rollouts[parent_idx]

                # Δ > 0 means branch (alternative granularity) was better
                delta = ro["reward"] - base_ro["reward"]
                r_img = delta * img_scale
                # Per-branch clip, then total clip
                r_img = max(-img_step_cap, min(img_step_cap, r_img))
                r_img = max(-img_total_cap, min(img_total_cap, r_img))

                # Symmetric: branch gets +r_img, base gets -r_img
                ro["reward"] += r_img
                ro["reward_breakdown"]["r_imagination"] = round(r_img, 4)
                base_ro["reward"] -= r_img
                base_ro["reward_breakdown"]["r_imagination"] = round(-r_img, 4)

            # Re-clamp after imagination adjustment
            for ro in rollouts:
                ro["reward"] = max(ro["reward"], 0.0)

            # ========== Phase 2: group-relative advantage ==========
            rewards = [r["reward"] for r in rollouts]

            mu = sum(rewards) / len(rewards)
            raw_std = math.sqrt(sum((r - mu)**2 for r in rewards) / len(rewards))

            # Robust advantage: if reward variance is too low, the gradient
            # is mostly noise. Use a minimum std threshold to avoid
            # amplifying noise, and skip the update entirely when rewards
            # are truly identical.
            MIN_ADV_STD = 0.05
            if raw_std < 1e-6:
                # All rewards identical → zero advantage, skip gradient
                for r in rollouts:
                    r["advantage"] = 0.0
            else:
                std = max(raw_std, MIN_ADV_STD)
                for i, r in enumerate(rollouts):
                    r["advantage"] = (rewards[i] - mu) / std
                    # Clip advantage to prevent extreme updates
                    r["advantage"] = max(-3.0, min(3.0, r["advantage"]))

            # ========== Logging ==========
            # Detailed: first 5 prompts per epoch + every 50th prompt
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
            # File: every prompt
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
                # GIPO loss: advantage * cross_entropy
                # Normalize by actual group size (2-4 rollouts)
                #   adv > 0: minimize CE → reinforce this trajectory
                #   adv < 0: maximize CE → suppress this trajectory
                n_rollouts = len(rollouts)
                pg_loss = adv * out.loss / (n_rollouts * grad_accum)

                if torch.isfinite(pg_loss):
                    pg_loss.backward()
                    prompt_loss += pg_loss.item() * n_rollouts * grad_accum

            # Stats
            n_rollouts = len(rollouts)
            acc_loss += prompt_loss
            acc_reward += sum(rewards) / len(rewards)  # track group mean reward
            skill_ratio = (sum(r["num_skill_calls"] for r in rollouts)
                           / max(sum(r["num_steps"] for r in rollouts), 1))
            acc_skill_r += skill_ratio
            acc_steps_ep += sum(r["num_steps"] for r in rollouts) / n_rollouts
            # GIPO: track imagination stats
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
                    logger.info(
                        f"step {global_step}/{total_steps} | "
                        f"loss={al:.4f}  reward={ar:.3f}  "
                        f"skill_ratio={sr:.2f}  avg_steps={se:.1f}  "
                        f"branches={br:.1f}/2  "
                        f"lr={clr:.2e}  epoch={epoch + (pidx+1)/len(train_prompts):.2f}"
                    )
                    if ar > best_reward:
                        best_reward = ar
                    acc_loss = acc_reward = acc_skill_r = acc_steps_ep = 0.0
                    acc_img_steps = 0.0
                    acc_cnt = 0

                # Periodic save (inside grad_accum block to avoid redundant saves)
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
    logger.info(f"GIPO complete → {output_dir}  best_reward={best_reward:.3f}")


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
    parser = argparse.ArgumentParser(description="AdaMacro Step 4: GIPO-API Training")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--sft-checkpoint", type=str, default=os.path.join(CHECKPOINT_DIR, "sft"))
    parser.add_argument("--rl-dataset", type=str, default=RL_DATASET_PATH)
    parser.add_argument("--augmented-tools", type=str, default=AUGMENTED_TOOLS_PATH)
    parser.add_argument("--skill-library", type=str, default=SKILL_LIBRARY_PATH)
    parser.add_argument("--grpo-data", type=str, default=GRPO_DATA_PATH)
    parser.add_argument("--output-dir", type=str, default=os.path.join(CHECKPOINT_DIR, "gipo_api"))
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    # API-specific arguments
    parser.add_argument("--api-model", type=str, default=None, help="LLM model for tool simulation")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--api-base-url", type=str, default=None, help="API base URL")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    grpo_config = GIPOAPIConfig()
    if args.group_size: grpo_config.group_size = args.group_size
    if args.epochs: grpo_config.num_epochs = args.epochs
    if args.lr: grpo_config.learning_rate = args.lr
    if args.api_model: grpo_config.api_model = args.api_model
    if args.api_key: grpo_config.api_key = args.api_key
    if args.api_base_url: grpo_config.api_base_url = args.api_base_url

    logger.info("=" * 70)
    logger.info("AdaMacro Step 4: GIPO-API Training (API-based tool simulation)")
    logger.info("=" * 70)

    if args.generate_only:
        logger.info("Online GRPO — no offline generation needed.")
        return

    train_grpo(args.model, args.sft_checkpoint, args.grpo_data,
               args.output_dir, grpo_config)


if __name__ == "__main__":
    main()