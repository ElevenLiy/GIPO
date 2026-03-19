"""
AdaMacro Step 2: Automatic Skill Instantiation & Trace-Aware Execution
=======================================================================

Implements Section 2.2 of the paper:
- Convert BPE macro tokens into executable skills (templates)
- Auto-determine exposed parameters vs internally-piped parameters
- Expose select_strategy as a discrete parameter for LLM control
- Inject trace recording and soft-interrupt mechanism
- Register skills into augmented tool library alongside atomic tools

Input:  skill_library.json (from Step 1), all_tools_v2.json
Output: augmented_tools.json (atomic + skill tools)
"""

import json
import copy
import logging
from typing import List, Dict, Tuple, Optional, Any, Set
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    SKILL_LIBRARY_PATH, ALL_TOOLS_PATH, AUGMENTED_TOOLS_PATH,
    TOOL_SIMULATOR_DB_PATH, ADAMACRO_OUTPUT_DIR, SkillConfig
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Skill Template Definitions
# ============================================================================

class SkillTemplate:
    """Base class for skill execution templates."""
    
    SEQUENTIAL = "sequential"
    SELECT = "select"
    CONDITIONAL = "conditional"
    
    @staticmethod
    def detect_template(tool_details: List[Dict]) -> str:
        """
        Auto-detect the appropriate template based on tool chain analysis.
        
        Rules:
        - If any tool returns list/array outputs and next tool needs an ID/index
          -> SELECT template
        - Otherwise -> SEQUENTIAL template
        """
        list_output_keywords = [
            "search", "list", "query", "find", "get_all", "fetch",
            "list_directory", "search_files"
        ]
        select_input_keywords = [
            "id", "index", "item", "target", "selected", "choice"
        ]
        
        for i in range(len(tool_details) - 1):
            curr_name = tool_details[i].get("name", "").lower()
            next_params = [p.lower() for p in tool_details[i + 1].get("params", [])]
            
            # Check if current tool likely returns a list
            is_list_output = any(kw in curr_name for kw in list_output_keywords)
            # Check if next tool needs selection
            needs_select = any(kw in p for p in next_params for kw in select_input_keywords)
            
            if is_list_output and needs_select:
                return SkillTemplate.SELECT
        
        return SkillTemplate.SEQUENTIAL


# ============================================================================
# Parameter Analysis
# ============================================================================

def analyze_skill_parameters(
    tool_details: List[Dict],
    tool_schemas: Dict[str, Dict],
) -> Tuple[List[Dict], List[Dict], Dict[int, Dict[str, str]]]:
    """
    Analyze parameter flow within a macro to determine:
    1. exposed_params: Parameters the LLM must provide (skill's formal parameters)
    2. internal_params: Parameters auto-piped from previous step outputs
    3. pipe_map: Mapping of which output fields feed into which input params
    
    Strategy:
    - The first tool's required params are always exposed
    - For subsequent tools, if a param name matches an output field of any
      previous tool, it's internal (auto-piped)
    - Remaining params are exposed
    
    Returns:
        exposed_params, internal_params, pipe_map
    """
    exposed_params = []
    internal_params = []
    pipe_map = {}  # {step_idx: {param_name: source_description}}
    
    # Track what output fields are available from previous steps
    available_outputs: Set[str] = set()
    
    for step_idx, tool in enumerate(tool_details):
        tool_name = tool.get("name", "")
        schema = tool_schemas.get(tool_name, {})
        properties = schema.get("parameters_schema", {}).get("properties", {})
        required = set(schema.get("parameters_schema", {}).get("required", []))
        actual_keys = set(tool.get("params", []))
        
        all_params = actual_keys | set(properties.keys())
        step_pipes = {}
        
        for param in sorted(all_params):
            param_info = {
                "name": param,
                "tool_name": tool_name,
                "step_index": step_idx,
                "type": properties.get(param, {}).get("type", "string"),
                "description": properties.get(param, {}).get("description", ""),
                "required": param in required,
            }
            
            if step_idx == 0:
                # First tool: all params are exposed
                exposed_params.append(param_info)
            else:
                # Check if this param can be piped from previous outputs
                # Heuristic: common naming patterns
                pipeable = False
                for common_pipe in _get_pipe_candidates(param):
                    if common_pipe in available_outputs:
                        step_pipes[param] = f"output_of_step_{step_idx - 1}.{common_pipe}"
                        pipeable = True
                        break
                
                if pipeable:
                    internal_params.append(param_info)
                else:
                    exposed_params.append(param_info)
        
        if step_pipes:
            pipe_map[step_idx] = step_pipes
        
        # Add this tool's output fields to available outputs
        # Use actual_keys as proxy for potential output fields
        available_outputs.update(actual_keys)
        # Also add common output field names
        available_outputs.update(["result", "output", "data", "content", "text", "id", "path"])
    
    return exposed_params, internal_params, pipe_map


def _get_pipe_candidates(param_name: str) -> List[str]:
    """Generate candidate field names that might pipe into this param."""
    candidates = [param_name]
    # Common mappings
    pipe_mappings = {
        "path": ["path", "file_path", "filepath", "output_path"],
        "content": ["content", "text", "data", "output", "result"],
        "id": ["id", "item_id", "target_id", "selected_id"],
        "query": ["query", "search_query", "q"],
        "url": ["url", "link", "href"],
    }
    for key, vals in pipe_mappings.items():
        if param_name.lower() in [v.lower() for v in vals] or key in param_name.lower():
            candidates.extend(vals)
    return list(set(candidates))


# ============================================================================
# Skill Instantiation
# ============================================================================

def instantiate_skill(
    macro: Dict,
    tool_schemas: Dict[str, Dict],
    skill_config: SkillConfig,
) -> Dict:
    """
    Convert a BPE macro into an executable skill definition.
    
    The skill:
    - Has a name like "skill_macro_001"
    - Has a natural language description
    - Has formal parameters (exposed + select_strategy)
    - Has internal execution template
    - Supports trace recording and soft interrupt
    """
    macro_id = macro["macro_id"]
    tool_names = macro["tool_names"]
    tool_details = macro.get("tool_details", [])
    
    # Detect template type
    template_type = SkillTemplate.detect_template(tool_details)
    
    # Analyze parameters
    exposed_params, internal_params, pipe_map = analyze_skill_parameters(
        tool_details, tool_schemas
    )
    
    # Build skill name and description
    skill_name = f"skill_{macro_id}"
    
    # Generate description from tool chain
    tool_desc_parts = []
    for td in tool_details:
        desc = td.get("description", "")
        name = td.get("name", "")
        if desc:
            tool_desc_parts.append(f"{name}: {desc[:100]}")
        else:
            tool_desc_parts.append(name)
    
    skill_description = (
        f"[Composite Skill] Executes a {len(tool_names)}-step workflow: "
        f"{' -> '.join(tool_names)}. "
        f"Template: {template_type}. "
        f"Steps: {'; '.join(tool_desc_parts[:3])}"
    )
    if len(tool_desc_parts) > 3:
        skill_description += f" ... and {len(tool_desc_parts) - 3} more steps."
    
    # Build parameter schema for the skill
    skill_params = {}
    for p in exposed_params:
        skill_params[p["name"]] = {
            "type": p["type"],
            "description": p["description"],
            "required": p["required"],
            "source_tool": p["tool_name"],
            "source_step": p["step_index"],
        }
    
    # Add select_strategy parameter for SELECT templates
    if template_type == SkillTemplate.SELECT:
        skill_params["select_strategy"] = {
            "type": "string",
            "description": (
                "Strategy for selecting from candidate list. "
                "Options: rank-0 (select first/top), rank-1 (select second), "
                "random (random choice), filter (filter by condition)"
            ),
            "required": False,
            "enum": skill_config.select_strategies,
            "default": "rank-0",
        }
    
    # Build execution plan
    execution_plan = []
    for step_idx, tname in enumerate(tool_names):
        step = {
            "step_index": step_idx,
            "tool_name": tname,
            "params_source": {},
        }
        
        # Determine parameter sources
        td = tool_details[step_idx] if step_idx < len(tool_details) else {}
        for param in td.get("params", []):
            if step_idx in pipe_map and param in pipe_map[step_idx]:
                step["params_source"][param] = {
                    "type": "pipe",
                    "source": pipe_map[step_idx][param],
                }
            else:
                step["params_source"][param] = {
                    "type": "exposed",
                    "param_name": param,
                }
        
        execution_plan.append(step)
    
    skill = {
        "skill_id": skill_name,
        "macro_id": macro_id,
        "name": skill_name,
        "description": skill_description,
        "type": "function",
        "template_type": template_type,
        "tool_chain": tool_names,
        "tool_chain_length": len(tool_names),
        "parameters_schema": {
            "type": "object",
            "properties": skill_params,
            "required": [p["name"] for p in exposed_params if p["required"]],
        },
        "exposed_params": [p["name"] for p in exposed_params],
        "internal_params": [p["name"] for p in internal_params],
        "pipe_map": {str(k): v for k, v in pipe_map.items()},
        "execution_plan": execution_plan,
        "trace_enabled": skill_config.enable_trace,
        "soft_interrupt_enabled": skill_config.enable_soft_interrupt,
        "frequency": macro.get("current_frequency", 0),
        "compression_gain": macro.get("compression_gain", 0),
    }
    
    return skill


# ============================================================================
# Trace-Aware Skill Interpreter
# ============================================================================

class SkillInterpreter:
    """
    Runtime interpreter for executing skills with trace recording.
    
    Trace format:
        Trace(S_m) = [(tool_1, status_1), ..., (tool_k, status_k)]
    
    On hard failure (timeout, error) or logic block (empty candidates, assertion):
        - Triggers soft interrupt
        - Returns partial trace + intermediate observation
        - Returns control to the policy (which can fall back to atomic tools)
    """
    
    def __init__(self, tool_simulator_db: Optional[Dict] = None):
        """
        Args:
            tool_simulator_db: Tool simulator database for offline execution
        """
        self.tool_simulator = tool_simulator_db
    
    def execute_skill(
        self,
        skill: Dict,
        input_params: Dict[str, Any],
        max_retries: int = 1,
    ) -> Dict:
        """
        Execute a skill and return trace.
        
        Args:
            skill: Skill definition
            input_params: Input parameters from the policy
            max_retries: Maximum retries per step
            
        Returns:
            {
                "success": bool,
                "trace": [(tool_name, status), ...],
                "outputs": [output_per_step, ...],
                "interrupted": bool,
                "interrupt_step": int or None,
                "interrupt_reason": str or None,
                "final_output": str,
            }
        """
        trace = []
        outputs = []
        execution_plan = skill.get("execution_plan", [])
        pipe_map = skill.get("pipe_map", {})
        select_strategy = input_params.get("select_strategy", "rank-0")
        
        # Running context for parameter piping
        context = {"input_params": input_params, "step_outputs": []}
        
        for step_idx, step in enumerate(execution_plan):
            tool_name = step["tool_name"]
            
            # Resolve parameters for this step
            try:
                step_params = self._resolve_params(step, context, select_strategy)
            except Exception as e:
                trace.append((tool_name, "param_error"))
                return {
                    "success": False,
                    "trace": trace,
                    "outputs": outputs,
                    "interrupted": True,
                    "interrupt_step": step_idx,
                    "interrupt_reason": f"Parameter resolution failed: {str(e)}",
                    "final_output": "",
                }
            
            # Execute the tool (via simulator)
            try:
                output = self._execute_tool(tool_name, step_params)
                
                if output is None or output == "Not found":
                    # Soft failure - tool returned empty
                    trace.append((tool_name, "empty_output"))
                    outputs.append("")
                    
                    if skill.get("soft_interrupt_enabled", True):
                        return {
                            "success": False,
                            "trace": trace,
                            "outputs": outputs,
                            "interrupted": True,
                            "interrupt_step": step_idx,
                            "interrupt_reason": "Empty tool output",
                            "final_output": "",
                        }
                else:
                    trace.append((tool_name, "success"))
                    outputs.append(output)
                    context["step_outputs"].append(output)
                    
            except Exception as e:
                trace.append((tool_name, "error"))
                outputs.append("")
                
                if skill.get("soft_interrupt_enabled", True):
                    return {
                        "success": False,
                        "trace": trace,
                        "outputs": outputs,
                        "interrupted": True,
                        "interrupt_step": step_idx,
                        "interrupt_reason": f"Execution error: {str(e)}",
                        "final_output": "",
                    }
        
        # All steps completed successfully
        final_output = outputs[-1] if outputs else ""
        return {
            "success": True,
            "trace": trace,
            "outputs": outputs,
            "interrupted": False,
            "interrupt_step": None,
            "interrupt_reason": None,
            "final_output": final_output,
        }
    
    def _resolve_params(
        self, step: Dict, context: Dict, select_strategy: str
    ) -> Dict:
        """Resolve parameters for a step using input params and piped outputs."""
        params = {}
        for param_name, source in step.get("params_source", {}).items():
            if source["type"] == "exposed":
                # Get from input parameters
                exposed_name = source.get("param_name", param_name)
                if exposed_name in context["input_params"]:
                    params[param_name] = context["input_params"][exposed_name]
            elif source["type"] == "pipe":
                # Get from previous step output
                step_outputs = context["step_outputs"]
                if step_outputs:
                    last_output = step_outputs[-1]
                    # Try to parse and extract field
                    params[param_name] = self._extract_from_output(
                        last_output, param_name, select_strategy
                    )
        return params
    
    def _extract_from_output(
        self, output: str, field_name: str, select_strategy: str
    ) -> Any:
        """Extract a field from a tool output, applying select_strategy if needed."""
        try:
            parsed = json.loads(output) if isinstance(output, str) else output
        except (json.JSONDecodeError, TypeError):
            return output
        
        # If parsed is a list, apply select_strategy
        if isinstance(parsed, list) and parsed:
            if select_strategy == "rank-0":
                item = parsed[0]
            elif select_strategy == "rank-1":
                item = parsed[1] if len(parsed) > 1 else parsed[0]
            elif select_strategy == "random":
                import random
                item = random.choice(parsed)
            elif select_strategy == "filter":
                item = parsed[0]  # Default filter selects first
            else:
                item = parsed[0]
            
            # Try to extract field_name from item
            if isinstance(item, dict) and field_name in item:
                return item[field_name]
            return item
        
        # If parsed is a dict, try to get field
        if isinstance(parsed, dict):
            if field_name in parsed:
                return parsed[field_name]
            # Try common variants
            for key in parsed:
                if field_name.lower() in key.lower():
                    return parsed[key]
        
        return output
    
    def _execute_tool(self, tool_name: str, params: Dict) -> Optional[str]:
        """Execute a tool using the simulator database."""
        if self.tool_simulator is None:
            return json.dumps({"status": "simulated", "tool": tool_name})
        
        tools = self.tool_simulator.get("tools", {})
        tool_data = tools.get(tool_name, {})
        
        if not tool_data:
            return "Not found"
        
        # Find matching call by parameter keys
        import hashlib
        param_keys = sorted(params.keys())
        key_hash = hashlib.md5(str(param_keys).encode()).hexdigest()[:12]
        
        schemas = tool_data.get("schemas", {})
        for schema_hash, schema_data in schemas.items():
            calls = schema_data.get("calls", [])
            if calls:
                # Find closest match by comparing parameter values
                best_match = calls[0]  # Default to first call
                for call in calls:
                    call_args = call.get("args", {})
                    if self._params_match(params, call_args):
                        return call.get("output", "")
                return best_match.get("output", "")
        
        return "Not found"
    
    def _params_match(self, query_params: Dict, db_params: Dict) -> bool:
        """Check if query parameters match database parameters."""
        if not query_params or not db_params:
            return False
        # Exact key match
        return set(query_params.keys()) == set(db_params.keys()) and \
               all(str(query_params.get(k)) == str(db_params.get(k)) for k in query_params)


# ============================================================================
# Build Augmented Tool Library
# ============================================================================

def build_augmented_tools(
    all_tools_path: str,
    skill_library_path: str,
    tool_schemas: Dict[str, Dict],
    skill_config: SkillConfig,
) -> List[Dict]:
    """
    Build the augmented tool library: A = A_atom ∪ A_skill
    
    Atomic tools are preserved as-is; skills are added with proper schemas.
    """
    # Load atomic tools
    with open(all_tools_path, "r", encoding="utf-8") as f:
        atomic_tools = json.load(f)
    
    # Load skill library
    with open(skill_library_path, "r", encoding="utf-8") as f:
        skill_lib = json.load(f)
    
    macros = skill_lib.get("macros", {})
    
    # Instantiate skills
    skills = []
    for macro_id, macro in macros.items():
        skill = instantiate_skill(macro, tool_schemas, skill_config)
        skills.append(skill)
    
    logger.info(f"Instantiated {len(skills)} skills from {len(macros)} macros")
    
    # Build augmented list
    augmented = []
    
    # Add atomic tools (preserve original format)
    max_atomic_id = 0
    for tool in atomic_tools:
        augmented.append({
            "id": tool.get("id", len(augmented)),
            "name": tool.get("name", ""),
            "original_name": tool.get("original_name", tool.get("name", "")),
            "type": "function",
            "is_skill": False,
            "description": tool.get("description", ""),
            "parameters_schema": tool.get("parameters_schema", {}),
            "actual_keys": tool.get("actual_keys", []),
        })
        max_atomic_id = max(max_atomic_id, tool.get("id", 0))
    
    # Add skills
    for i, skill in enumerate(skills):
        skill_entry = {
            "id": max_atomic_id + 1 + i,
            "name": skill["name"],
            "original_name": skill["name"],
            "type": "function",
            "is_skill": True,
            "skill_id": skill["skill_id"],
            "description": skill["description"],
            "parameters_schema": skill["parameters_schema"],
            "actual_keys": skill["exposed_params"],
            "template_type": skill["template_type"],
            "tool_chain": skill["tool_chain"],
            "tool_chain_length": skill["tool_chain_length"],
            "execution_plan": skill["execution_plan"],
            "pipe_map": skill["pipe_map"],
            "trace_enabled": skill["trace_enabled"],
            "soft_interrupt_enabled": skill["soft_interrupt_enabled"],
        }
        augmented.append(skill_entry)
    
    logger.info(f"Augmented tool library: {len(atomic_tools)} atomic + {len(skills)} skills "
                f"= {len(augmented)} total")
    
    return augmented


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AdaMacro Step 2: Skill Instantiation")
    parser.add_argument("--skill-library", type=str, default=SKILL_LIBRARY_PATH)
    parser.add_argument("--all-tools", type=str, default=ALL_TOOLS_PATH)
    parser.add_argument("--output", type=str, default=AUGMENTED_TOOLS_PATH)
    args = parser.parse_args()
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("AdaMacro Step 2: Skill Instantiation & Trace-Aware Execution")
    logger.info("=" * 70)
    
    # Load tool schemas
    tool_schemas = {}
    with open(args.all_tools, "r", encoding="utf-8") as f:
        tools = json.load(f)
    for t in tools:
        name = t.get("name", "")
        tool_schemas[name] = t
    
    skill_config = SkillConfig()
    
    # Build augmented tools
    augmented = build_augmented_tools(
        args.all_tools, args.skill_library, tool_schemas, skill_config
    )
    
    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved augmented tools to {args.output}")
    
    # Print summary
    n_atomic = sum(1 for t in augmented if not t.get("is_skill", False))
    n_skill = sum(1 for t in augmented if t.get("is_skill", False))
    logger.info(f"Atomic tools: {n_atomic}, Skills: {n_skill}, Total: {len(augmented)}")
    
    # Print skill details
    logger.info("\nSkill details:")
    for t in augmented:
        if t.get("is_skill"):
            logger.info(f"  {t['name']}: {' -> '.join(t['tool_chain'])} "
                       f"(template={t['template_type']}, params={t['actual_keys']})")


if __name__ == "__main__":
    main()
