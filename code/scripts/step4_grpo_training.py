"""
AdaMacro Step 4: Online GRPO Training
=======================================

Training process overview:
==========================

Each "step" in the log = 1 optimizer update, which processes (grad_accum × prompts).

For EACH training prompt:
  1. Generate G=4 rollouts with different temperatures
     Each rollout = model interacts with environment step-by-step:
       turn 0: model generates → parse tool_call → env.execute → get tool_response
       turn 1: model sees history + tool_response → generates next tool_call → env.execute
       ...
       turn N: model generates text (no tool_call) → episode done
  2. Compute reward for each rollout
  3. Group-normalize → advantage  (positive = better than group mean)
  4. Policy gradient: loss = advantage × cross_entropy(assistant_tokens_only)
  5. Accumulate gradients for grad_accum prompts → optimizer.step()  ← this is 1 "step"

Reward:
  R = R_task + λ·R_skill + R_efficiency
  R_task:  how many gt tools were covered (exact + fuzzy name matching)
  R_skill: bonus for using skills successfully (THE key AdaMacro incentive)
  R_eff:   bonus for fewer decision steps (skills compress steps)
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
    TOOL_SIMULATOR_DB_PATH, GRPO_DATA_PATH, CHECKPOINT_DIR,
    ADAMACRO_OUTPUT_DIR, GRPOConfig, get_model_path, DEFAULT_MODEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
        # Total: r_task + skill_bonus + efficiency
        # Additive design prevents skill bias from dominating r_task.
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
            "order_bonus": round(_order_bonus, 4),
            "n_skill_ok": n_skill_ok,
            "n_skill_relevant": n_skill_relevant,
            "n_skill_irrelevant": n_skill_irrelevant,
            "exact_match": int(len(all_used & gt_set)) if gt_set else 0,
            "tools_used": list(all_used)[:10],
        }
        return total, breakdown


# ============================================================================
# Tool Environment
# ============================================================================

class ToolEnvironment:
    """
    Wraps tool_simulator_database + SkillInterpreter.
    
    Parameter matching: 4-level fallback
      1. Exact match in DB
      2. Key-overlap match
      3. Most-frequent call from rl_dataset
      4. Generic response
    """
    def __init__(self, augmented_tools_path: str, tool_sim_db_path: str,
                 rl_dataset_path: str = None):
        with open(augmented_tools_path, "r") as f:
            augmented_tools = json.load(f)
        with open(tool_sim_db_path, "r") as f:
            self.tool_sim_db = json.load(f)

        self.skills = {t["name"]: t for t in augmented_tools if t.get("is_skill")}
        self.atomic_tools = {t["name"]: t for t in augmented_tools if not t.get("is_skill")}
        self.all_tool_names = set(t["name"] for t in augmented_tools)

        lines = []
        for t in augmented_tools[:60]:
            tag = "[SKILL]" if t.get("is_skill") else "[TOOL]"
            lines.append(f"{tag} {t['name']}: {t.get('description','')[:100]}")
        if len(augmented_tools) > 60:
            lines.append(f"... and {len(augmented_tools) - 60} more tools.")
        self.tool_desc = "\n".join(lines)

        try:
            from step2_skill_instantiation import SkillInterpreter
        except ImportError:
            from step2_skill_instantiation import SkillInterpreter
        self.interpreter = SkillInterpreter(self.tool_sim_db)

        # Build indexes
        self.tool_freq_index = {}
        self.tool_calls_index = {}
        if rl_dataset_path and os.path.exists(rl_dataset_path):
            self._build_freq_index(rl_dataset_path)
        self._build_calls_index()

    def _build_freq_index(self, path):
        from collections import Counter
        with open(path) as f:
            data = json.load(f)
        tool_arg_counts = defaultdict(Counter)
        tool_arg_data = defaultdict(dict)
        for ep in data.get("episodes", []):
            if ep.get("success", 0) != 1:
                continue
            names = ep.get("tool_names", [])
            args_list = ep.get("tool_args", [])
            outs = ep.get("output_texts", [])
            for i, tn in enumerate(names):
                a = args_list[i] if i < len(args_list) else "{}"
                o = outs[i] if i < len(outs) else ""
                try:
                    ad = json.loads(a) if isinstance(a, str) else a
                except:
                    ad = {}
                if not isinstance(ad, dict):
                    ad = {}
                sig = str(sorted(ad.keys()))
                tool_arg_counts[tn][sig] += 1
                if sig not in tool_arg_data[tn]:
                    tool_arg_data[tn][sig] = (ad, o)
        for tn, ctr in tool_arg_counts.items():
            best_sig = ctr.most_common(1)[0][0]
            args, out = tool_arg_data[tn][best_sig]
            self.tool_freq_index[tn] = {"args": args, "output": out}
        logger.info(f"Built freq index for {len(self.tool_freq_index)} tools")

    def _build_calls_index(self):
        for tn, td in self.tool_sim_db.get("tools", {}).items():
            cl = []
            for sd in td.get("schemas", {}).values():
                for c in sd.get("calls", []):
                    cl.append({"args": c.get("args", {}), "output": c.get("output", "")})
            if cl:
                self.tool_calls_index[tn] = cl

    def execute_tool(self, tool_name, model_args):
        if not isinstance(model_args, dict):
            model_args = {}
        calls = self.tool_calls_index.get(tool_name, [])
        if calls:
            model_keys = set(model_args.keys())
            for c in calls:
                da = c["args"]
                if (isinstance(da, dict) and model_keys == set(da.keys()) and
                        all(str(model_args.get(k)) == str(da.get(k)) for k in model_args)):
                    return c["output"]
            if model_keys:
                best_s, best_o = 0, calls[0]["output"]
                for c in calls:
                    dk = set(c["args"].keys()) if isinstance(c["args"], dict) else set()
                    if dk:
                        s = len(model_keys & dk) / max(len(model_keys | dk), 1)
                        if s > best_s:
                            best_s, best_o = s, c["output"]
                if best_s > 0:
                    return best_o
            return calls[0]["output"]
        if tool_name in self.tool_freq_index:
            return self.tool_freq_index[tool_name]["output"]
        return json.dumps({"status": "ok", "tool": tool_name, "result": "executed"})

    def execute(self, tool_name, arguments):
        if not isinstance(arguments, dict):
            arguments = {}

        # Try exact match first, then normalized name fallback
        # Model may generate "excel-format_range" but DB has "excel-format_range_v13"
        resolved_name = tool_name
        if (tool_name not in self.skills and tool_name not in self.atomic_tools
                and tool_name not in self.tool_calls_index):
            norm = normalize_tool_name(tool_name)
            # Search for a DB entry whose normalized name matches
            for orig in list(self.all_tool_names) + list(self.tool_calls_index.keys()):
                if normalize_tool_name(orig) == norm:
                    resolved_name = orig
                    break

        # Check if resolved tool_name is known AT ALL
        is_known = (resolved_name in self.skills or resolved_name in self.atomic_tools
                    or resolved_name in self.all_tool_names
                    or resolved_name in self.tool_calls_index
                    or resolved_name in self.tool_freq_index)

        if resolved_name in self.skills:
            skill_def = self.skills[resolved_name]
            result = self.interpreter.execute_skill(skill_def, arguments)
            chain_len = len(result.get("trace", []))
            return {
                "output": result.get("final_output", "") or "",
                "is_skill": True,
                "trace": result.get("trace", []),
                "success": result.get("success", False),
                "interrupted": result.get("interrupted", False),
                "atomic_cost": max(chain_len, 1),
            }
        elif is_known:
            output = self.execute_tool(resolved_name, arguments)
            ok = output is not None and output != "Not found" and output != ""
            return {
                "output": output or "",
                "is_skill": False,
                "trace": [(tool_name, "success" if ok else "not_found")],
                "success": ok,
                "interrupted": False,
                "atomic_cost": 1,
            }
        else:
            # UNKNOWN tool: return error, mark as failure
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


def parse_tool_call(response: str) -> Optional[Dict]:
    tc = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', response, re.DOTALL)
    if tc:
        try:
            obj = json.loads(tc.group(1))
            if isinstance(obj, dict) and "name" in obj:
                return {"name": str(obj["name"]), "arguments": obj.get("arguments", {})}
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
    m = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*\}', response)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "name" in obj:
                return {"name": str(obj["name"]), "arguments": obj.get("arguments", obj.get("parameters", {}))}
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
    try:
        obj = json.loads(response.strip())
        if isinstance(obj, dict) and "name" in obj:
            return {"name": str(obj["name"]), "arguments": obj.get("arguments", {})}
    except:
        pass
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
    model, tokenizer, env: ToolEnvironment,
    system_prompt: str, user_prompt: str,
    max_turns: int, max_new_tokens: int,
    temperature: float, device,
    oracle_first_tool: str = None,
    gt_tools_len: int = 0,
) -> Dict:
    """
    Run a single rollout episode.
    
    Strategy: "free first, force if stuck"
    - Every turn: let model generate freely (it may output tool_call or text)
    - If model outputs tool_call → execute it, continue
    - If model outputs text AND has ≥1 prior action → treat as task done (natural stop)
    - If model outputs text AND has 0 actions → use forced prefix to guarantee ≥1 tool call
    
    This gives the model freedom to decide when to stop, while ensuring
    every trajectory has at least 1 action for GRPO to learn from.
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
    continued_this_rollout = False  # at most 1 continuation nudge per rollout
    total_atomic = 0
    completed = False

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

                # Salvage
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

        result = env.execute(tool_name, arguments)
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
    grpo_config: GRPOConfig,
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
    env = ToolEnvironment(AUGMENTED_TOOLS_PATH, TOOL_SIMULATOR_DB_PATH, RL_DATASET_PATH)
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

    logger.info(f"Online GRPO  G={G}  epochs={num_epochs}  lr={lr}  "
                f"grad_accum={grad_accum}  total_steps≈{total_steps}")

    # ---------------------------------------------------------------- training loop
    global_step = 0
    acc_loss = acc_reward = acc_steps_ep = acc_skill_r = 0.0
    acc_cnt = 0
    best_reward = -1e9

    for epoch in range(num_epochs):
        random.shuffle(train_prompts)
        logger.info(f"\n{'='*60}\nEpoch {epoch+1}/{num_epochs}\n{'='*60}")

        for pidx, pdata in enumerate(train_prompts):
            user_prompt = pdata["user_prompt"]
            gt_tools = pdata["gt_tools"]

            # ========== Phase 1: generate G rollouts ==========
            model.eval()
            rollouts = []

            # Per-prompt system prompt (uses task category, NOT gt_tools to avoid leak)
            system_prompt = build_system_prompt(pdata["task_name"])

            # Build a skill-biased variant for g=0:
            # Append directive instead of fragile string replace
            skill_biased_prompt = system_prompt + (
                "\n\nIMPORTANT: You SHOULD prefer [SKILL] tools over atomic tools when possible. "
                "Skills chain multiple steps and are more efficient. "
                "Check the [SKILL] entries in the tool list first."
            )

            for g in range(G):
                t = base_temp * (0.6 + g * 0.3)
                t = max(0.1, min(t, 1.5))

                # Anti-collapse strategy with 4 diverse rollout modes:
                #   g=0: skill-biased prompt (encourage skill usage)
                #   g=1: normal prompt
                #   g=2: high temperature (t × 1.5) for exploration
                #   g=3: oracle-seeded — pick a random gt_tool as forced first action
                #
                # g=3 is the KEY anti-collapse mechanism:
                # Instead of letting the model choose (it'll pick the collapsed tool),
                # we FORCE a random gt_tool as the first tool call. The model then
                # continues from there. This rollout almost always gets a different
                # (and often better) reward, breaking the advantage=0 deadlock.
                
                oracle_first_tool = None
                up = user_prompt  # per-group user prompt (may be augmented)
                if g == 0:
                    # g=0: low temperature exploitation with normal prompt
                    sp = system_prompt
                elif g == 2:
                    t = min(base_temp * 1.8, 2.0)  # much higher temp
                    sp = system_prompt
                elif g == G - 1:
                    # Oracle-seeded: use normal prompt (not skill-biased).
                    # 50/50 chance of seeding with skill vs atomic gt_tool
                    # to ensure model learns BOTH pathways.
                    sp = system_prompt
                    if gt_tools:
                        best_skill = find_best_matching_skill(gt_tools, env.skills)
                        if best_skill and random.random() < 0.5:
                            oracle_first_tool = best_skill
                        else:
                            oracle_first_tool = random.choice(gt_tools)
                else:
                    sp = system_prompt

                rollout = run_rollout(
                    model, tokenizer, env,
                    sp, up,
                    max_turns=max_turns, max_new_tokens=max_gen_tok,
                    temperature=t, device=device,
                    oracle_first_tool=oracle_first_tool,
                    gt_tools_len=len(gt_tools),
                )
                used_tools = [name for name, _ in rollout["actions"]]
                skill_names_list = list(rollout["skill_names_used"])
                reward, breakdown = reward_fn.compute(
                    used_tools=used_tools,
                    gt_tools=gt_tools,
                    skill_traces=rollout["skill_traces"],
                    skill_names=skill_names_list,
                    num_decision_steps=rollout["num_steps"],
                    num_skill_calls=rollout["num_skill_calls"],
                    total_atomic_cost=rollout["total_atomic"],
                    completed=rollout["completed"],
                    max_steps=max_turns,
                )
                rollout["reward"] = reward
                rollout["reward_breakdown"] = breakdown
                rollouts.append(rollout)

            # --- 0-step resample: if ALL rollouts are 0-step, resample ---
            # GRPO needs at least 1 trajectory with actions for gradient signal.
            # Retry up to 3 times with increasing temperature.
            has_any_action = any(r["num_steps"] > 0 for r in rollouts)
            resample_attempts = 0
            while not has_any_action and resample_attempts < 3:
                resample_attempts += 1
                # Higher temperature → more exploration
                t = base_temp * (1.5 + resample_attempts * 0.5)
                t = min(t, 2.0)
                # Try skill-biased prompt (more likely to produce tool calls)
                rollout = run_rollout(
                    model, tokenizer, env,
                    skill_biased_prompt, user_prompt,
                    max_turns=max_turns, max_new_tokens=max_gen_tok,
                    temperature=t, device=device,
                    gt_tools_len=len(gt_tools),
                )
                used_tools = [name for name, _ in rollout["actions"]]
                skill_names_list_r = list(rollout["skill_names_used"])
                reward, breakdown = reward_fn.compute(
                    used_tools=used_tools, gt_tools=gt_tools,
                    skill_traces=rollout["skill_traces"],
                    skill_names=skill_names_list_r,
                    num_decision_steps=rollout["num_steps"],
                    num_skill_calls=rollout["num_skill_calls"],
                    total_atomic_cost=rollout["total_atomic"],
                    completed=rollout["completed"], max_steps=max_turns,
                )
                rollout["reward"] = reward
                rollout["reward_breakdown"] = breakdown

                if rollout["num_steps"] > 0:
                    # Replace the worst (first 0-step) rollout
                    for ri in range(len(rollouts)):
                        if rollouts[ri]["num_steps"] == 0:
                            rollouts[ri] = rollout
                            break
                    has_any_action = True
                    if pidx < 5:
                        logger.info(f"    [resample] attempt {resample_attempts}: "
                                    f"got {rollout['num_steps']} steps, "
                                    f"reward={reward:.3f}, t={t:.1f}")

            # --- Skill resample: if no rollout used a skill, try up to 2 more ---
            has_skill = any(r["num_skill_calls"] > 0 for r in rollouts)
            has_any_action = any(r["num_steps"] > 0 for r in rollouts)
            if not has_skill and has_any_action and env.skills:
                # Find which skills COULD apply (their constituent tools overlap with gt)
                gt_set = set(gt_tools)
                candidate_skills = []
                for sname, sdef in env.skills.items():
                    chain = sdef.get("tool_chain", [])
                    if any(t in gt_set for t in chain):
                        candidate_skills.append(sname)

                # Build skill-hint prompt: explicitly mention candidate skills
                if candidate_skills:
                    skill_hint = (
                        f"\n\nHint: consider using one of these skills which may help: "
                        f"{', '.join(candidate_skills[:5])}"
                    )
                else:
                    skill_hint = (
                        "\n\nHint: check if any [SKILL] in the tool list can handle "
                        "multiple steps at once."
                    )

                for retry in range(2):
                    t = base_temp * 1.3
                    rollout = run_rollout(
                        model, tokenizer, env,
                        skill_biased_prompt, user_prompt + skill_hint,
                        max_turns=max_turns, max_new_tokens=max_gen_tok,
                        temperature=t, device=device,
                        gt_tools_len=len(gt_tools),
                    )
                    used_tools = [name for name, _ in rollout["actions"]]
                    skill_names_list2 = list(rollout["skill_names_used"])
                    reward, breakdown = reward_fn.compute(
                        used_tools=used_tools, gt_tools=gt_tools,
                        skill_traces=rollout["skill_traces"],
                        skill_names=skill_names_list2,
                        num_decision_steps=rollout["num_steps"],
                        num_skill_calls=rollout["num_skill_calls"],
                        total_atomic_cost=rollout["total_atomic"],
                        completed=rollout["completed"], max_steps=max_turns,
                    )
                    rollout["reward"] = reward
                    rollout["reward_breakdown"] = breakdown
                    if rollout["num_skill_calls"] > 0:
                        worst_idx = min(range(len(rollouts)), key=lambda i: rollouts[i]["reward"])
                        rollouts[worst_idx] = rollout
                        break

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
                logger.info(f"  [prompt {pidx}] gt_tools={gt_tools[:4]}...")
                for g, r in enumerate(rollouts):
                    used = [name for name, _ in r["actions"]]
                    bd = r["reward_breakdown"]
                    logger.info(
                        f"    g={g} steps={r['num_steps']} skills={r['num_skill_calls']} "
                        f"relevant={bd.get('n_skill_relevant',0)} "
                        f"completed={r['completed']} reward={r['reward']:.3f} "
                        f"(task={bd['r_task']:.2f} order={bd.get('order_bonus',0):.2f} "
                        f"sk_bonus={bd.get('skill_bonus',0):+.2f} irrel={bd.get('n_skill_irrelevant',0)} "
                        f"eff={bd['r_efficiency']:.2f}) "
                        f"adv={r['advantage']:+.2f} tools={used[:5]}"
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
                # GRPO loss: advantage * cross_entropy
                #   adv > 0: minimize CE → reinforce this trajectory
                #   adv < 0: maximize CE → suppress this trajectory
                pg_loss = adv * out.loss / (G * grad_accum)

                if torch.isfinite(pg_loss):
                    pg_loss.backward()
                    prompt_loss += pg_loss.item() * G * grad_accum

            # Stats
            acc_loss += prompt_loss
            acc_reward += sum(rewards) / len(rewards)  # track group mean reward
            skill_ratio = (sum(r["num_skill_calls"] for r in rollouts)
                           / max(sum(r["num_steps"] for r in rollouts), 1))
            acc_skill_r += skill_ratio
            acc_steps_ep += sum(r["num_steps"] for r in rollouts) / G
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
                    clr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"step {global_step}/{total_steps} | "
                        f"loss={al:.4f}  reward={ar:.3f}  "
                        f"skill_ratio={sr:.2f}  avg_steps={se:.1f}  "
                        f"lr={clr:.2e}  epoch={epoch + (pidx+1)/len(train_prompts):.2f}"
                    )
                    if ar > best_reward:
                        best_reward = ar
                    acc_loss = acc_reward = acc_skill_r = acc_steps_ep = 0.0
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
    logger.info(f"GRPO complete → {output_dir}  best_reward={best_reward:.3f}")


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
    parser = argparse.ArgumentParser(description="AdaMacro Step 4: Online GRPO")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--sft-checkpoint", type=str, default=os.path.join(CHECKPOINT_DIR, "sft"))
    parser.add_argument("--rl-dataset", type=str, default=RL_DATASET_PATH)
    parser.add_argument("--augmented-tools", type=str, default=AUGMENTED_TOOLS_PATH)
    parser.add_argument("--skill-library", type=str, default=SKILL_LIBRARY_PATH)
    parser.add_argument("--tool-simulator-db", type=str, default=TOOL_SIMULATOR_DB_PATH)
    parser.add_argument("--grpo-data", type=str, default=GRPO_DATA_PATH)
    parser.add_argument("--output-dir", type=str, default=os.path.join(CHECKPOINT_DIR, "grpo"))
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    grpo_config = GRPOConfig()
    if args.group_size: grpo_config.group_size = args.group_size
    if args.epochs: grpo_config.num_epochs = args.epochs
    if args.lr: grpo_config.learning_rate = args.lr

    logger.info("=" * 70)
    logger.info("AdaMacro Step 4: Online GRPO Training")
    logger.info("=" * 70)

    if args.generate_only:
        logger.info("Online GRPO — no offline generation needed.")
        return

    train_grpo(args.model, args.sft_checkpoint, args.grpo_data,
               args.output_dir, grpo_config)


if __name__ == "__main__":
    main()