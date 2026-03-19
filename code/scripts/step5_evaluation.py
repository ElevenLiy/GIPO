"""
AdaMacro Step 5: Evaluation
============================

Evaluates the trained model on test trajectories.

Metrics (Section 3.3):
- Task success rate
- Decision steps (number of policy decisions)
- Atomic tool calls (true budget measure)
- Skill usage ratio, coverage, success rate
- Skill interrupt ratio & position distribution
- select_strategy hit rate
"""

import json
import logging
import os
import re
import time
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict, Counter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    RL_DATASET_PATH, AUGMENTED_TOOLS_PATH, SKILL_LIBRARY_PATH,
    TOOL_SIMULATOR_DB_PATH, EVAL_RESULTS_DIR, CHECKPOINT_DIR,
    EvalConfig, get_model_path, DEFAULT_MODEL,
)
from step2_skill_instantiation import SkillInterpreter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Agent with Skill Support
# ============================================================================

class AdaMacroAgent:
    """
    Agent that can use both atomic tools and skills.
    Handles skill execution with trace and soft-interrupt.
    """

    def __init__(
        self,
        model_path: str,
        augmented_tools: List[Dict],
        tool_simulator_db: Optional[Dict] = None,
        eval_config: EvalConfig = EvalConfig(),
        lora_path: Optional[str] = None,
    ):
        self.eval_config = eval_config
        self.augmented_tools = augmented_tools

        # Separate skills from atomic tools
        self.skills = {t["name"]: t for t in augmented_tools if t.get("is_skill")}
        self.atomic_tools = {t["name"]: t for t in augmented_tools if not t.get("is_skill")}

        # Build normalized → original name mapping (matches step4 ToolEnvironment.execute)
        self._norm_to_orig = {}  # normalized_name → original_name
        self._known_tool_names = set()
        for t in augmented_tools:
            name = t["name"]
            self._known_tool_names.add(name)
            norm = re.sub(r'_v\d+\w*$', '', name.lower().replace("-", "_").replace(".", "_"))
            self._known_tool_names.add(norm)
            if norm not in self._norm_to_orig:
                self._norm_to_orig[norm] = name

        # Skill interpreter
        self.interpreter = SkillInterpreter(tool_simulator_db)

        # Load model
        self._load_model(model_path, lora_path)

    def _load_model(self, model_path: str, lora_path: Optional[str] = None):
        """Load the model with optional LoRA adapter."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)

        if lora_path and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            logger.info(f"Loaded LoRA adapter from {lora_path}")

        self.model.eval()

    def generate_action(self, messages: List[Dict], has_prior_actions: bool = False) -> Dict:
        """Generate the next action.
        
        Turn 0 (no prior actions): forced prefix, single generation
        Turn 1+ (has prior actions): free generation, natural stop if text
        """
        import torch

        TOOL_CALL_PREFIX = '<tool_call>\n{"name": "'

        formatted = [{"role": m["role"], "content": m.get("content", "") or ""} for m in messages]
        try:
            prompt = self.tokenizer.apply_chat_template(
                formatted, tokenize=False, add_generation_prompt=True)
        except:
            parts = [f"<|{m['role']}|>\n{m.get('content','')}" for m in formatted]
            prompt = "\n".join(parts) + "\n<|assistant|>\n"

        if not isinstance(prompt, str) or not prompt.strip():
            return {"type": "text", "content": "", "raw_response": ""}

        gen_kwargs = {
            "max_new_tokens": 512,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.eval_config.temperature <= 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.eval_config.temperature
            gen_kwargs["top_p"] = self.eval_config.top_p

        if not has_prior_actions:
            # --- FIRST TURN: forced prefix ---
            forced_prompt = prompt + TOOL_CALL_PREFIX
            try:
                inputs = self.tokenizer(forced_prompt, return_tensors="pt", truncation=True, max_length=4096)
            except (TypeError, ValueError):
                return {"type": "text", "content": "", "raw_response": ""}
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            completion = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

            if not completion:
                return {"type": "text", "content": "", "raw_response": ""}

            full_response = TOOL_CALL_PREFIX + completion
            result = self._parse_action(full_response)
            if result["type"] == "tool_call":
                return result

            # Salvage: only accept if it matches a known tool name
            name_match = re.match(r'^([a-zA-Z][\w\-\.]*)', completion)
            if name_match and len(name_match.group(1)) <= 60:
                salvaged_name = name_match.group(1)
                norm_salvaged = salvaged_name.lower().replace("-", "_").replace(".", "_")
                if salvaged_name in self._known_tool_names or norm_salvaged in self._known_tool_names:
                    return {
                        "type": "tool_call",
                        "name": salvaged_name,
                        "arguments": {},
                        "raw_response": full_response,
                    }
            return {"type": "text", "content": "", "raw_response": completion}

        else:
            # --- SUBSEQUENT TURNS: free generation ---
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            except (TypeError, ValueError):
                return {"type": "text", "content": "", "raw_response": ""}
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

            result = self._parse_action(response)
            if result["type"] == "tool_call":
                return result

            # Text output → natural stop
            return {"type": "text", "content": response, "raw_response": response}

    def _parse_action(self, response: str) -> Dict:
        """Parse assistant response to extract tool call.
        Handles multiple formats: <tool_call>, raw JSON, Qwen function_call, etc.
        """
        import re

        # Format 1: <tool_call>...</tool_call>
        tc_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', response, re.DOTALL)
        if tc_match:
            try:
                tc = json.loads(tc_match.group(1))
                if isinstance(tc, dict) and "name" in tc:
                    return {
                        "type": "tool_call",
                        "name": str(tc["name"]),
                        "arguments": tc.get("arguments", {}),
                        "raw_response": response,
                    }
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        # Format 2: {"name": "...", "arguments": {...}} anywhere in response
        json_match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*\}', response)
        if json_match:
            try:
                tc = json.loads(json_match.group(0))
                if "name" in tc:
                    return {
                        "type": "tool_call",
                        "name": tc["name"],
                        "arguments": tc.get("arguments", tc.get("parameters", {})),
                        "raw_response": response,
                    }
            except json.JSONDecodeError:
                pass

        # Format 3: Try full response as JSON
        try:
            tc = json.loads(response.strip())
            if isinstance(tc, dict) and "name" in tc:
                return {
                    "type": "tool_call",
                    "name": tc["name"],
                    "arguments": tc.get("arguments", tc.get("parameters", {})),
                    "raw_response": response,
                }
        except:
            pass

        # Format 4: Qwen-style function call (function_name\n{args})
        fn_match = re.search(r'(\w[\w\-\.]+)\s*\n\s*(\{.*?\})', response, re.DOTALL)
        if fn_match:
            try:
                args = json.loads(fn_match.group(2))
                return {
                    "type": "tool_call",
                    "name": fn_match.group(1),
                    "arguments": args,
                    "raw_response": response,
                }
            except:
                pass

        return {
            "type": "text",
            "content": response,
            "raw_response": response,
        }

    def _resolve_skill_name(self, tool_name: str) -> Optional[str]:
        """Resolve a tool name to a skill key, handling normalization/aliases."""
        if tool_name in self.skills:
            return tool_name
        # Try case-insensitive / normalized matching
        norm = tool_name.lower().replace("-", "_").replace(".", "_")
        norm = re.sub(r'_v\d+\w*$', '', norm)
        for sname in self.skills:
            sn = sname.lower().replace("-", "_").replace(".", "_")
            sn = re.sub(r'_v\d+\w*$', '', sn)
            if norm == sn:
                return sname
        return None

    def execute_action(self, action: Dict) -> Dict:
        """Execute an action (either skill or atomic tool)."""
        if action["type"] != "tool_call":
            return {"output": action.get("content", ""), "trace": [], "is_skill": False}

        tool_name = action["name"]
        arguments = action.get("arguments", {})

        resolved_skill = self._resolve_skill_name(tool_name)
        if resolved_skill is not None:
            # Execute skill with trace
            skill_def = self.skills[resolved_skill]
            result = self.interpreter.execute_skill(skill_def, arguments)
            return {
                "output": result.get("final_output", ""),
                "trace": result.get("trace", []),
                "is_skill": True,
                "skill_success": result.get("success", False),
                "interrupted": result.get("interrupted", False),
                "interrupt_step": result.get("interrupt_step"),
            }
        else:
            # Execute atomic tool via simulator
            # Resolve name: exact → normalized → full-library scan (same as step4)
            resolved = tool_name
            output = self.interpreter._execute_tool(resolved, arguments)
            if output is None or output == "Not found":
                norm = re.sub(r'_v\d+\w*$', '', tool_name.lower().replace("-", "_").replace(".", "_"))
                orig = self._norm_to_orig.get(norm)
                if orig and orig != tool_name:
                    resolved = orig
                    output = self.interpreter._execute_tool(resolved, arguments)
            is_found = output is not None and output != "Not found"
            return {
                "output": output if is_found else "",
                "trace": [(resolved, "success" if is_found else "not_found")],
                "is_skill": False,
            }

    def run_episode(self, user_prompt: str, available_tools_desc: str, episode_idx: int = 0, gt_tools: List[str] = None) -> Dict:
        """Run a complete episode with the agent."""
        # System prompt MUST match training format (step4) exactly
        # Count tools actually in the provided description (not full library)
        n_tool_lines = sum(1 for line in available_tools_desc.split("\n") if line.strip().startswith("- "))
        n_skill_lines = sum(1 for line in available_tools_desc.split("\n") if "[SKILL]" in line)
        system_prompt = (
            "You are a tool-calling agent. You MUST use tools to complete tasks. "
            "Do NOT answer directly — always call at least one tool first.\n\n"
            "You have access to both atomic tools and composite skills. "
            "Skills are pre-composed tool chains that execute multiple tools in sequence. "
            "Choose whichever tools (atomic or skill) best fit the task.\n\n"
            f"Available tools ({n_tool_lines} total, including {n_skill_lines} skills):\n"
            f"{available_tools_desc}\n\n"
            "To call a tool, respond ONLY with:\n"
            "<tool_call>\n"
            '{"name": "tool_name", "arguments": {"param": "value"}}\n'
            "</tool_call>\n\n"
            "After receiving all tool responses, provide a brief text summary to finish."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        episode_data = {
            "actions": [],
            "traces": [],
            "decision_steps": 0,
            "atomic_calls": 0,
            "skill_calls": 0,
            "skill_successes": 0,
            "skill_interrupts": 0,
            "interrupt_positions": [],
        }

        # Dynamic forced steps: match GRPO training logic
        gt_tools_len = len(gt_tools) if gt_tools else 0
        min_forced = min(max(3, gt_tools_len // 2), 6) if gt_tools_len > 0 else 3
        continued = False  # at most 1 continuation nudge per episode

        for turn in range(self.eval_config.max_turns):
            num_actions = len(episode_data["actions"])
            has_prior = num_actions >= min_forced
            action = self.generate_action(messages, has_prior_actions=has_prior)

            # Log raw output for first 3 episodes to help debug
            if episode_idx < 3 and turn < 3:
                logger.info(f"  [Debug] Episode {episode_idx} Turn {turn}: "
                           f"type={action['type']}, "
                           f"raw={action.get('raw_response', '')[:200]}")

            if action["type"] == "text":
                # Check if agent stopped too early — continuation nudge
                if (num_actions < min_forced + 1
                        and turn < self.eval_config.max_turns - 1
                        and not continued):
                    continued = True
                    messages.append({"role": "assistant", "content": action.get("content", "")})
                    messages.append({
                        "role": "user",
                        "content": "You have not completed the task yet. Continue calling tools to finish."
                    })
                    continue
                # Agent finished
                episode_data["final_response"] = action.get("content", "")
                break

            # Only count actual tool calls as decision steps (not text/nudge)
            episode_data["decision_steps"] += 1

            # Execute
            result = self.execute_action(action)
            episode_data["actions"].append(action)

            if result["is_skill"]:
                episode_data["skill_calls"] += 1
                # Use actual trace length (handles interrupted skills correctly)
                actual_trace = result.get("trace", [])
                episode_data["atomic_calls"] += max(len(actual_trace), 1)

                if result.get("skill_success"):
                    episode_data["skill_successes"] += 1
                if result.get("interrupted"):
                    episode_data["skill_interrupts"] += 1
                    episode_data["interrupt_positions"].append(result.get("interrupt_step", 0))

                episode_data["traces"].append(result.get("trace", []))
            else:
                episode_data["atomic_calls"] += 1

            # Budget check
            if episode_data["atomic_calls"] >= self.eval_config.max_atomic_calls:
                logger.info(f"Budget exhausted at turn {turn}")
                break

            # Add to conversation
            messages.append({
                "role": "assistant",
                "content": f"<tool_call>\n{{\"name\": \"{action['name']}\", \"arguments\": {json.dumps(action.get('arguments', {}), ensure_ascii=False)}}}\n</tool_call>"
            })
            messages.append({
                "role": "user",
                "content": f"<tool_response name=\"{action['name']}\">\n{str(result['output'])[:1500]}\n</tool_response>"
            })

        return episode_data


# ============================================================================
# Evaluation Runner
# ============================================================================

def evaluate(
    model_name: str,
    lora_path: Optional[str],
    rl_dataset_path: str,
    augmented_tools_path: str,
    tool_simulator_db_path: str,
    output_path: str,
    eval_config: EvalConfig,
    max_episodes: int = 100,
):
    """Run evaluation on test episodes."""
    # Load data
    with open(augmented_tools_path, "r") as f:
        augmented_tools = json.load(f)

    tool_sim_db = None
    if os.path.exists(tool_simulator_db_path):
        with open(tool_simulator_db_path, "r") as f:
            tool_sim_db = json.load(f)

    with open(rl_dataset_path, "r") as f:
        rl_data = json.load(f)

    episodes = rl_data.get("episodes", [])

    # --- Fix issue 6: train/test split ---
    # Use last 30% as test set (first 70% used for SFT/GRPO training)
    total_eps = len(episodes)
    test_start = int(total_eps * 0.7)
    test_episodes = episodes[test_start:]
    logger.info(f"Train/test split: using episodes [{test_start}:{total_eps}] "
                f"({len(test_episodes)} episodes) for evaluation")

    # Build tool description map — matches step4 format
    # (normalized names, with [SKILL] tag and chain info)
    def _normalize_tn(name: str) -> str:
        import re as _re
        n = name.lower().replace("-", "_").replace(".", "_")
        return _re.sub(r'_v\d+\w*$', '', n)

    tool_desc_map = {}  # norm_name → description string
    norm_skill_names = set()
    for t in augmented_tools:
        orig_name = t["name"]
        norm_name = _normalize_tn(orig_name)
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
            norm_skill_names.add(norm_name)
            chain = t.get("tool_chain", [])
            if not chain:
                chain = [s.get("tool_name", "") for s in t.get("execution_plan", [])]
            if chain:
                norm_chain = [_normalize_tn(c) for c in chain[:5]]
                chain_str = f" [chain: {' → '.join(norm_chain)}]"
        tool_desc_map[norm_name] = f"- {tag}{norm_name}: {t.get('description','')[:100]}{chain_str}{param_str}"

    def _build_tools_desc(task_name: str) -> str:
        """Build per-task tool list matching step4 logic:
        skills (≤15) → category-related atomic (≤40) → random fill (≤50)."""
        lines = []
        seen = set()

        # Infer tool category from task_name
        task_lower = task_name.lower().replace("-", "_")
        category_keywords = set()
        for part in task_lower.split("_"):
            if len(part) >= 3:
                category_keywords.add(part)

        # 1) Skills first (always visible, core to AdaMacro)
        for tn, desc in tool_desc_map.items():
            if tn in norm_skill_names:
                lines.append(desc)
                seen.add(tn)
            if len(seen) >= 15:
                break

        # 2) Tools whose name matches task category keywords
        for tn, desc in tool_desc_map.items():
            if tn in seen:
                continue
            if any(kw in tn for kw in category_keywords):
                lines.append(desc)
                seen.add(tn)
            if len(seen) >= 40:
                break

        # 3) Fill remaining with random other tools
        import random as _rand
        other_tools = [tn for tn in tool_desc_map if tn not in seen]
        _rand.shuffle(other_tools)
        for tn in other_tools:
            lines.append(tool_desc_map[tn])
            seen.add(tn)
            if len(seen) >= 50:
                break

        return "\n".join(lines), len(seen), sum(1 for tn in seen if tn in norm_skill_names)

    # Initialize agent
    model_path = get_model_path(model_name)
    agent = AdaMacroAgent(
        model_path=model_path,
        augmented_tools=augmented_tools,
        tool_simulator_db=tool_sim_db,
        eval_config=eval_config,
        lora_path=lora_path,
    )

    results = []
    for i, ep in enumerate(test_episodes[:max_episodes]):
        user_prompt = ep.get("user_prompt", "")
        if not user_prompt:
            continue

        task_name = ep.get("task_name", "")
        logger.info(f"Episode {i+1}/{min(len(test_episodes), max_episodes)}: {task_name}")
        start_time = time.time()

        # Build per-task tool list (matching step4 training format)
        tools_desc, n_tools, n_skills = _build_tools_desc(task_name)
        episode_data = agent.run_episode(user_prompt, tools_desc, episode_idx=i, gt_tools=ep.get("tool_names", []))
        elapsed = time.time() - start_time

        episode_data["task_name"] = ep.get("task_name", "")
        episode_data["elapsed_time"] = elapsed
        episode_data["ground_truth_success"] = ep.get("success", 0)
        episode_data["gt_tools"] = ep.get("tool_names", [])
        results.append(episode_data)

    # Compute aggregate metrics
    metrics = compute_metrics(results, augmented_tools)

    # Save
    output = {
        "meta": {
            "model": model_name,
            "lora_path": lora_path,
            "num_episodes": len(results),
            "eval_config": {
                "max_turns": eval_config.max_turns,
                "max_atomic_calls": eval_config.max_atomic_calls,
            },
        },
        "metrics": metrics,
        "episodes": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"\nEvaluation complete. Results saved to {output_path}")
    logger.info("=" * 60)
    logger.info("  KEY METRICS")
    logger.info(f"  1. Tool F1                       : {metrics['tool_f1']:.2%}  (P={metrics['tool_precision']:.2%}, R={metrics['tool_recall']:.2%})")
    logger.info(f"  2. Next-tool Prediction Acc      : {metrics['next_tool_acc']:.2%}  ({metrics['next_tool_correct']}/{metrics['next_tool_total']})")
    logger.info(f"  3. Skill Usage Ratio             : {metrics['skill_usage_ratio']:.2%}")
    logger.info(f"  4. Avg Decision Steps            : {metrics['avg_decision_steps']:.1f}  (atomic calls: {metrics['avg_atomic_calls']:.1f})")
    logger.info("=" * 60)
    logger.info("  AUXILIARY")
    logger.info(f"  Progress Rate (F1>0)             : {metrics['progress_rate']:.2%}  ({metrics['progress_episodes']}/{metrics['num_episodes_valid']})")
    logger.info(f"  Skill success rate               : {metrics['skill_success_rate']:.2%}")
    logger.info(f"  Skill coverage (used/available)  : {metrics['skill_coverage']:.2%}  ({metrics['skills_used']}/{metrics['skills_available']})")
    logger.info(f"All metrics: {json.dumps(metrics, indent=2)}")
    return metrics


def compute_metrics(results: List[Dict], augmented_tools: List[Dict]) -> Dict:
    """Compute evaluation metrics from episode results.

    KEY METRICS (reported in paper):
      1. Tool F1       — harmonic mean of Precision and Recall (trajectory-level)
      2. Next-tool Acc — per-step next-tool prediction accuracy
      3. Skill Usage Ratio — fraction of decision steps that invoke a skill
      4. Avg Decision Steps / Avg Atomic Calls — efficiency comparison
    """
    n = len(results)
    if n == 0:
        return {}

    skills_available = {t["name"] for t in augmented_tools if t.get("is_skill")}
    # Build skill → atomic chain mapping (normalized keys for consistent lookup)
    skill_chains = {}
    for t in augmented_tools:
        if t.get("is_skill"):
            chain = t.get("tool_chain", [])
            if not chain:
                chain = [s.get("tool_name", "") for s in t.get("execution_plan", [])]
            # Index by both original and normalized name
            skill_chains[t["name"]] = chain
            norm_key = t["name"].lower().replace("-", "_").replace(".", "_")
            import re as _re
            norm_key = _re.sub(r'_v\d+\w*$', '', norm_key)
            if norm_key != t["name"]:
                skill_chains[norm_key] = chain

    total_decisions = sum(r["decision_steps"] for r in results)
    total_atomic = sum(r["atomic_calls"] for r in results)
    total_skill_calls = sum(r["skill_calls"] for r in results)
    total_skill_success = sum(r["skill_successes"] for r in results)
    total_skill_interrupts = sum(r["skill_interrupts"] for r in results)

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _normalize_tool_name(name: str) -> str:
        """Strip version suffixes like _v1, _v13."""
        import re as _re
        return _re.sub(r'_v\d+\w*$', '', name)

    def _tokenize_tool(name: str) -> set:
        import re as _re
        n = name.lower().replace("-", "_").replace(".", "_")
        n = _re.sub(r'_v\d+$', '', n)
        return set(t for t in n.split("_") if len(t) >= 2)

    def _fuzzy_match_score(a: str, b: str) -> float:
        """Three-level fuzzy tool matching (same as step4 reward)."""
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

    # ------------------------------------------------------------------
    # KEY METRIC 1: Tool F1 (Precision, Recall, F1 per episode)
    #
    #   Recall    = Σ best_match(g, used) / |gt|   for g in gt
    #   Precision = Σ best_match(u, gt)   / |used|  for u in used
    #   F1        = 2 * P * R / (P + R)
    # ------------------------------------------------------------------
    per_episode_recall = []
    per_episode_precision = []
    per_episode_f1 = []
    per_episode_tool_counts = []

    for r in results:
        gt = set(_normalize_tool_name(t) for t in r.get("gt_tools", []))

        # Collect all atomic tools used (expanding skills via traces)
        all_atomic_used = set()
        for a in r.get("actions", []):
            aname = _normalize_tool_name(a.get("name", ""))
            if aname in skill_chains:
                for atomic_t in skill_chains[aname]:
                    all_atomic_used.add(_normalize_tool_name(atomic_t))
            else:
                all_atomic_used.add(aname)
        for trace in r.get("traces", []):
            for tool_name, status in trace:
                all_atomic_used.add(_normalize_tool_name(tool_name))

        if gt and all_atomic_used:
            # Recall: for each gt tool, find best match in used
            recall_credit = 0.0
            for g in gt:
                best = max(_fuzzy_match_score(g, ut) for ut in all_atomic_used)
                recall_credit += best
            recall = min(recall_credit / len(gt), 1.0)

            # Precision: for each used tool, find best match in gt
            precision_credit = 0.0
            for ut in all_atomic_used:
                best = max(_fuzzy_match_score(ut, g) for g in gt)
                precision_credit += best
            precision = min(precision_credit / len(all_atomic_used), 1.0)

            # F1
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            # --- [Optional] Order-aware coverage (commented out for future use) ---
            # To re-enable: uncomment and use order_coverage instead of / alongside f1
            #
            # gt_list = [_normalize_tool_name(t) for t in r.get("gt_tools", [])]
            # used_list = [_normalize_tool_name(a.get("name", "")) for a in r.get("actions", [])]
            # gt_match_positions = []
            # for gi, gtool in enumerate(gt_list):
            #     best_score = 0.0
            #     best_ui = -1
            #     for ui, ut in enumerate(used_list):
            #         score = _fuzzy_match_score(gtool, ut)
            #         if score > best_score:
            #             best_score = score
            #             best_ui = ui
            #         if best_score == 1.0:
            #             break
            #     if best_score > 0:
            #         gt_match_positions.append((gi, best_ui, best_score))
            #
            # if len(gt_match_positions) >= 2:
            #     import bisect
            #     pos_seq = [ui for _, ui, _ in gt_match_positions]
            #     tails = []
            #     for p in pos_seq:
            #         idx = bisect.bisect_left(tails, p)
            #         if idx == len(tails):
            #             tails.append(p)
            #         else:
            #             tails[idx] = p
            #     order_bonus = len(tails) / len(gt_match_positions)
            # elif len(gt_match_positions) == 1:
            #     order_bonus = 1.0
            # else:
            #     order_bonus = 0.0
            #
            # order_coverage = recall * (0.7 + 0.3 * order_bonus)
            # --- End optional order-aware coverage ---

        elif not gt:
            # No ground truth — skip this episode (don't inflate scores)
            continue
        else:
            recall = 0.0
            precision = 0.0
            f1 = 0.0

        per_episode_recall.append(recall)
        per_episode_precision.append(precision)
        per_episode_f1.append(f1)
        per_episode_tool_counts.append(r.get("decision_steps", 0))

    n_valid = len(per_episode_f1)  # episodes with valid gt (excludes empty gt)
    avg_recall = sum(per_episode_recall) / n_valid if n_valid > 0 else 0
    avg_precision = sum(per_episode_precision) / n_valid if n_valid > 0 else 0
    avg_f1 = sum(per_episode_f1) / n_valid if n_valid > 0 else 0
    avg_tool_calls = sum(per_episode_tool_counts) / n_valid if n_valid > 0 else 0

    # Progress rate: fraction of episodes with any correct tool match
    progress_episodes = sum(1 for f in per_episode_f1 if f > 0)
    progress_rate = progress_episodes / n_valid if n_valid > 0 else 0

    # ------------------------------------------------------------------
    # KEY METRIC 2: Next-tool Prediction Accuracy
    #
    # For each step i in the episode, check if the tool called at step i
    # matches gt_tools[i] (positional). This measures whether the model
    # predicts the correct *next* tool at each decision point.
    #   next_tool_acc = correct_steps / total_steps  (micro-averaged)
    # ------------------------------------------------------------------
    next_tool_correct = 0
    next_tool_total = 0

    for r in results:
        gt_list = [_normalize_tool_name(t) for t in r.get("gt_tools", [])]
        actions = r.get("actions", [])

        for step_i, action in enumerate(actions):
            if step_i >= len(gt_list):
                break  # no more gt to compare

            # Expand skill to its first atomic tool for comparison
            aname = _normalize_tool_name(action.get("name", ""))
            # If the action is a skill, also consider its constituent atomic tools
            candidates = {aname}
            if aname in skill_chains:
                for atomic_t in skill_chains[aname]:
                    candidates.add(_normalize_tool_name(atomic_t))

            gt_tool = gt_list[step_i]
            # Check if any candidate fuzzy-matches the gt tool at this position
            best_score = max(_fuzzy_match_score(c, gt_tool) for c in candidates)
            if best_score >= 0.5:
                next_tool_correct += 1
            next_tool_total += 1

    next_tool_acc = next_tool_correct / next_tool_total if next_tool_total > 0 else 0.0

    # ------------------------------------------------------------------
    # Auxiliary metrics
    # ------------------------------------------------------------------
    # Skills actually used (normalize for consistent matching)
    norm_skills_available = {}
    for s in skills_available:
        ns = s.lower().replace("-", "_").replace(".", "_")
        ns = re.sub(r'_v\d+\w*$', '', ns)
        norm_skills_available[ns] = s
    skills_used = set()
    for r in results:
        for a in r.get("actions", []):
            aname = a.get("name", "")
            if aname in skills_available:
                skills_used.add(aname)
            else:
                na = aname.lower().replace("-", "_").replace(".", "_")
                na = re.sub(r'_v\d+\w*$', '', na)
                if na in norm_skills_available:
                    skills_used.add(norm_skills_available[na])

    # Interrupt position distribution
    all_interrupt_pos = []
    for r in results:
        all_interrupt_pos.extend(r.get("interrupt_positions", []))
    interrupt_dist = Counter(all_interrupt_pos)

    metrics = {
        # === KEY METRICS (reported in paper) ===
        "tool_f1": round(avg_f1, 4),
        "tool_recall": round(avg_recall, 4),
        "tool_precision": round(avg_precision, 4),
        "next_tool_acc": round(next_tool_acc, 4),
        "skill_usage_ratio": round(total_skill_calls / max(total_decisions, 1), 4),
        "avg_decision_steps": round(total_decisions / n, 2) if n > 0 else 0,
        "avg_atomic_calls": round(total_atomic / n, 2) if n > 0 else 0,
        # === Auxiliary (for reference) ===
        "num_episodes": n,
        "num_episodes_valid": n_valid,
        "avg_tool_calls": round(avg_tool_calls, 2),
        "progress_rate": round(progress_rate, 4),
        "progress_episodes": progress_episodes,
        "skill_coverage": round(len(skills_used) / max(len(skills_available), 1), 4),
        "skill_success_rate": round(total_skill_success / max(total_skill_calls, 1), 4),
        "skill_interrupt_ratio": round(total_skill_interrupts / max(total_skill_calls, 1), 4),
        "interrupt_position_distribution": dict(interrupt_dist),
        "total_skill_calls": total_skill_calls,
        "total_atomic_calls": total_atomic,
        "next_tool_correct": next_tool_correct,
        "next_tool_total": next_tool_total,
        "skills_used": len(skills_used),
        "skills_available": len(skills_available),
    }

    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AdaMacro Step 5: Evaluation")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--rl-dataset", type=str, default=RL_DATASET_PATH)
    parser.add_argument("--augmented-tools", type=str, default=AUGMENTED_TOOLS_PATH)
    parser.add_argument("--tool-simulator-db", type=str, default=TOOL_SIMULATOR_DB_PATH)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-atomic-calls", type=int, default=50)
    parser.add_argument("--stage", choices=["base", "sft", "grpo"], default="grpo",
                       help="Which checkpoint to evaluate")
    args = parser.parse_args()

    eval_config = EvalConfig(max_turns=args.max_turns, max_atomic_calls=args.max_atomic_calls)

    # Determine LoRA path based on stage
    if args.lora_path is None:
        if args.stage == "sft":
            args.lora_path = os.path.join(CHECKPOINT_DIR, "sft", args.model)
        elif args.stage == "grpo":
            args.lora_path = os.path.join(CHECKPOINT_DIR, "grpo", args.model)
        else:
            args.lora_path = None

    # Output path
    if args.output is None:
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
        args.output = os.path.join(
            EVAL_RESULTS_DIR,
            f"eval_{args.model}_{args.stage}_{int(time.time())}.json"
        )

    logger.info("=" * 70)
    logger.info("AdaMacro Step 5: Evaluation")
    logger.info(f"  Model: {args.model}, Stage: {args.stage}")
    logger.info("=" * 70)

    evaluate(
        model_name=args.model,
        lora_path=args.lora_path,
        rl_dataset_path=args.rl_dataset,
        augmented_tools_path=args.augmented_tools,
        tool_simulator_db_path=args.tool_simulator_db,
        output_path=args.output,
        eval_config=eval_config,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()