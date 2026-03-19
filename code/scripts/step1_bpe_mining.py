"""
AdaMacro Step 1: BPE-style Macro Mining with Budget Constraint
===============================================================

Implements the budgeted BPE algorithm from Section 2.1 of the paper:
- Extract successful trajectories and map them to tool token sequences
- Abstract each tool call to "tool_name[key_signature]" tokens
- Iteratively merge the most frequent adjacent pairs under a budget K
- Prune low-utility macros to control vocabulary size

Input:  rl_dataset (with tool sequences from successful episodes)
Output: skill_library.json (discovered macros with metadata)
"""

import re
import json
import copy
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import (
    RL_DATASET_PATH, ALL_TOOLS_PATH, SKILL_LIBRARY_PATH,
    ADAMACRO_OUTPUT_DIR, BPEConfig
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Token Abstraction
# ============================================================================

def abstract_tool_token(tool_name: str, param_keys: List[str]) -> str:
    """
    Abstract a tool call into a token: tool_name[key1,key2,...]
    This reduces parameter value noise and improves generalization.
    
    E.g., "filesystem-read_file" with keys ["path"] -> "filesystem-read_file[path]"
    """
    if param_keys:
        sig = ",".join(sorted(param_keys))
        return f"{tool_name}[{sig}]"
    return f"{tool_name}[]"


def extract_param_keys_from_args(args_str: str) -> List[str]:
    """Extract parameter keys from a tool_args JSON string."""
    if not args_str:
        return []
    try:
        parsed = json.loads(args_str) if isinstance(args_str, str) else args_str
        if isinstance(parsed, dict):
            return sorted(parsed.keys())
    except (json.JSONDecodeError, TypeError):
        pass
    return []


# ============================================================================
# Corpus Preparation
# ============================================================================

def load_successful_sequences(rl_dataset_path: str, success_only: bool = True) -> List[List[str]]:
    """
    Load trajectory sequences from RL dataset.
    Each sequence is a list of abstracted tool tokens.
    
    Returns:
        List of token sequences (one per successful episode)
    """
    with open(rl_dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    episodes = data.get("episodes", [])
    sequences = []
    
    for ep in episodes:
        if success_only and ep.get("success", 0) != 1:
            continue
        
        tool_names = ep.get("tool_names", [])
        tool_args = ep.get("tool_args", [])
        
        if not tool_names:
            continue
        
        tokens = []
        for i, name in enumerate(tool_names):
            args_str = tool_args[i] if i < len(tool_args) else "{}"
            keys = extract_param_keys_from_args(args_str)
            token = abstract_tool_token(name, keys)
            tokens.append(token)
        
        if len(tokens) >= 2:
            sequences.append(tokens)
    
    logger.info(f"Loaded {len(sequences)} sequences from {len(episodes)} episodes "
                f"(success_only={success_only})")
    return sequences


# ============================================================================
# BPE Algorithm
# ============================================================================

class BPEMacroMiner:
    """
    BPE-style iterative merging for macro discovery.
    
    Given corpus C, the weighted count for adjacent pair (a, b):
        f(a, b) = Σ_{τ∈C} Σ_{i=1}^{|τ|-1} 1[τ_i = a, τ_{i+1} = b]
    
    Each round selects:
        (a*, b*) = argmax_{(a,b)} f(a, b)
    
    and merges all occurrences of (a*, b*) into a new token.
    """
    
    def __init__(self, config: BPEConfig):
        self.config = config
        # Merge history: list of (pair, new_token, frequency, round)
        self.merge_history: List[Dict] = []
        # Final macro vocabulary
        self.macros: Dict[str, Dict] = {}
        # Original atomic vocabulary
        self.atomic_vocab: Set[str] = set()
    
    def _count_pairs(self, sequences: List[List[str]]) -> Counter:
        """Count all adjacent token pairs in the corpus."""
        pair_counts = Counter()
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] += 1
        return pair_counts
    
    def _merge_pair(
        self, sequences: List[List[str]], pair: Tuple[str, str], new_token: str
    ) -> List[List[str]]:
        """Replace all occurrences of `pair` with `new_token` in sequences."""
        a, b = pair
        new_sequences = []
        for seq in sequences:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_sequences.append(new_seq)
        return new_sequences
    
    def _get_macro_length(self, token: str) -> int:
        """Get the number of atomic tools in a macro token."""
        # A merged token is "tok_a ⊕ tok_b", recursively composed
        # We track this via merge history
        if token in self.atomic_vocab:
            return 1
        for record in self.merge_history:
            if record["new_token"] == token:
                len_a = self._get_macro_length(record["pair"][0])
                len_b = self._get_macro_length(record["pair"][1])
                return len_a + len_b
        return 1  # fallback
    
    def _decompose_macro(self, token: str) -> List[str]:
        """Recursively decompose a macro token into atomic tool tokens."""
        if token in self.atomic_vocab:
            return [token]
        for record in self.merge_history:
            if record["new_token"] == token:
                left = self._decompose_macro(record["pair"][0])
                right = self._decompose_macro(record["pair"][1])
                return left + right
        return [token]  # fallback
    
    def mine(self, sequences: List[List[str]]) -> Dict[str, Dict]:
        """
        Run BPE macro mining.
        
        Args:
            sequences: List of token sequences
            
        Returns:
            Dictionary of discovered macros:
            {
                macro_id: {
                    "token": merged_token_string,
                    "atomic_sequence": [tool_token1, tool_token2, ...],
                    "tool_names": [tool_name1, tool_name2, ...],
                    "frequency": int,
                    "length": int,
                    "merge_round": int,
                    "compression_gain": float,
                }
            }
        """
        # Collect atomic vocabulary
        for seq in sequences:
            self.atomic_vocab.update(seq)
        
        logger.info(f"Atomic vocabulary size: {len(self.atomic_vocab)}")
        logger.info(f"Corpus size: {len(sequences)} sequences, "
                    f"{sum(len(s) for s in sequences)} total tokens")
        
        current_sequences = [list(seq) for seq in sequences]
        
        for round_idx in range(self.config.max_merges):
            # Count pairs
            pair_counts = self._count_pairs(current_sequences)
            
            if not pair_counts:
                logger.info(f"No more pairs to merge at round {round_idx}")
                break
            
            # Find the most frequent pair
            best_pair, best_freq = pair_counts.most_common(1)[0]
            
            if best_freq < self.config.min_freq:
                logger.info(f"Best pair frequency {best_freq} < min_freq {self.config.min_freq}, stopping")
                break
            
            # Check max macro length constraint
            new_token = f"{best_pair[0]} ⊕ {best_pair[1]}"
            new_length = self._get_macro_length(best_pair[0]) + self._get_macro_length(best_pair[1])
            
            if new_length > self.config.max_macro_len:
                # Skip this pair and try next
                # Remove from counter and retry
                del pair_counts[best_pair]
                found = False
                for pair, freq in pair_counts.most_common():
                    if freq < self.config.min_freq:
                        break
                    cand_len = self._get_macro_length(pair[0]) + self._get_macro_length(pair[1])
                    if cand_len <= self.config.max_macro_len:
                        best_pair = pair
                        best_freq = freq
                        new_token = f"{pair[0]} ⊕ {pair[1]}"
                        new_length = cand_len
                        found = True
                        break
                if not found:
                    logger.info(f"No valid pair within max_macro_len at round {round_idx}")
                    break
            
            # Compute compression gain
            total_tokens_before = sum(len(s) for s in current_sequences)
            
            # Merge
            current_sequences = self._merge_pair(current_sequences, best_pair, new_token)
            
            total_tokens_after = sum(len(s) for s in current_sequences)
            compression_gain = total_tokens_before - total_tokens_after
            
            self.merge_history.append({
                "pair": best_pair,
                "new_token": new_token,
                "frequency": best_freq,
                "round": round_idx,
                "length": new_length,
                "compression_gain": compression_gain,
            })
            
            logger.info(
                f"Round {round_idx:3d}: merge ({best_pair[0]}, {best_pair[1]}) "
                f"freq={best_freq}, len={new_length}, gain={compression_gain}"
            )
        
        # Build macro dictionary with pruning
        self.macros = {}
        total_tokens = sum(len(s) for s in current_sequences)
        
        for i, record in enumerate(self.merge_history):
            macro_token = record["new_token"]
            
            # Count current frequency (after all merges)
            current_freq = sum(
                seq.count(macro_token) for seq in current_sequences
            )
            
            # Pruning: check minimum usage ratio
            usage_ratio = current_freq / max(total_tokens, 1)
            if usage_ratio < self.config.min_usage_ratio and current_freq < self.config.min_freq:
                continue
            
            # Check minimum length
            atomic_seq = self._decompose_macro(macro_token)
            if len(atomic_seq) < self.config.min_macro_len:
                continue
            
            # Extract tool names from atomic sequence
            tool_names = []
            for atom in atomic_seq:
                # Parse "tool_name[keys]" -> "tool_name"
                if "[" in atom:
                    tool_names.append(atom.split("[")[0])
                else:
                    tool_names.append(atom)
            
            macro_id = generate_semantic_macro_id(tool_names, set(self.macros.keys()))
            self.macros[macro_id] = {
                "macro_id": macro_id,
                "token": macro_token,
                "atomic_sequence": atomic_seq,
                "tool_names": tool_names,
                "length": len(atomic_seq),
                "original_frequency": record["frequency"],
                "current_frequency": current_freq,
                "usage_ratio": usage_ratio,
                "merge_round": record["round"],
                "compression_gain": record["compression_gain"],
            }
        
        logger.info(f"Discovered {len(self.macros)} macros after pruning "
                    f"(from {len(self.merge_history)} merges)")
        
        return self.macros


# ============================================================================
# Semantic Macro Naming
# ============================================================================

def _extract_function_name(tool_name: str) -> str:
    """Extract the function part from a tool name like 'server-func_name'.

    The function part is the last hyphen-separated segment that contains
    underscores, which distinguishes it from server-prefix segments.

    Examples:
        'pdf-tools-get_pdf_info' -> 'get_pdf_info'
        'filesystem-read_file' -> 'read_file'
        'playwright_with_chunk-browser_click' -> 'browser_click'
        'canvas-canvas_list_courses' -> 'canvas_list_courses'
    """
    segments = tool_name.split("-")
    # Find the LAST segment containing '_' (the function name)
    for i in range(len(segments) - 1, -1, -1):
        if "_" in segments[i]:
            return segments[i]
    # Fallback: return last segment
    return segments[-1]


def _abbreviate_function(func_name: str) -> str:
    """Abbreviate a function name to its core action.

    Examples:
        'get_pdf_info' -> 'get_pdf'
        'list_directory' -> 'list_dir'
        'browser_snapshot_navigate_to_next_span' -> 'snapshot_navigate'
        'canvas_list_courses' -> 'list_courses'
    """
    # Remove redundant server-name prefixes that leak into function names
    # e.g., 'canvas_list_courses' -> 'list_courses' (remove 'canvas_' prefix)
    _redundant_prefixes = [
        "canvas_canvas_", "canvas_", "woo_", "browser_",
    ]
    for prefix in _redundant_prefixes:
        if func_name.startswith(prefix) and len(func_name) > len(prefix):
            func_name = func_name[len(prefix):]
            break

    # Known abbreviations
    _abbrev = {
        "list_directory": "list_dir",
        "directory_tree": "dir_tree",
        "read_file": "read_file",
        "write_file": "write_file",
        "search_files": "search_files",
        "list_allowed_directories": "list_dirs",
        "get_pdf_info": "get_pdf",
        "read_pdf_pages": "read_pdf",
        "run_command": "run_cmd",
        "send_email": "send_email",
        "snapshot_navigate_to_next_span": "snapshot_next",
        "snapshot_search": "snapshot_search",
    }
    if func_name in _abbrev:
        return _abbrev[func_name]

    # General: keep at most first 3 underscore tokens
    tokens = func_name.split("_")
    if len(tokens) > 3:
        return "_".join(tokens[:3])
    return func_name


def generate_semantic_macro_id(tool_names: List[str], existing_ids: Set[str]) -> str:
    """Generate a descriptive macro ID from the tool chain.

    Examples:
        ['pdf-tools-get_pdf_info', 'pdf-tools-read_pdf_pages']
            -> 'get_pdf_and_read_pdf'
        ['filesystem-list_directory', 'filesystem-read_file']
            -> 'list_dir_and_read_file'
        ['playwright_with_chunk-browser_type', 'playwright_with_chunk-browser_click']
            -> 'type_and_click'
    """
    MAX_LEN = 70  # leave room for 'skill_' prefix (total < 77)

    parts = []
    for tool_name in tool_names:
        func = _extract_function_name(tool_name)
        short = _abbreviate_function(func)
        parts.append(short)

    base_id = "_and_".join(parts)

    # If still too long, progressively shorten each part
    if len(base_id) > MAX_LEN:
        # Strategy: keep only first 2 underscore tokens per part
        short_parts = []
        for p in parts:
            tokens = p.split("_")
            short_parts.append("_".join(tokens[:2]) if len(tokens) > 2 else p)
        base_id = "_and_".join(short_parts)

    # If STILL too long (many tools), keep first 3 and last, with count
    if len(base_id) > MAX_LEN:
        if len(parts) > 4:
            base_id = "_and_".join(parts[:3]) + f"_etc{len(parts)}"
        else:
            base_id = base_id[:MAX_LEN].rstrip("_")

    # Ensure uniqueness
    candidate = base_id
    counter = 2
    while candidate in existing_ids:
        suffix = f"_{counter}"
        candidate = base_id[:MAX_LEN - len(suffix)] + suffix
        counter += 1

    return candidate


# ============================================================================
# Utility: Load tool schema info for skill instantiation
# ============================================================================

def load_tool_schemas(all_tools_path: str) -> Dict[str, Dict]:
    """Load tool schemas from all_tools_v2.json."""
    with open(all_tools_path, "r", encoding="utf-8") as f:
        tools = json.load(f)
    
    schema_map = {}
    for tool in tools:
        name = tool.get("name") or tool.get("original_name", "")
        schema_map[name] = {
            "name": name,
            "original_name": tool.get("original_name", name),
            "description": tool.get("description", ""),
            "parameters_schema": tool.get("parameters_schema", {}),
            "actual_keys": tool.get("actual_keys", []),
        }
    return schema_map


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AdaMacro Step 1: BPE Macro Mining")
    parser.add_argument("--rl-dataset", type=str, default=RL_DATASET_PATH)
    parser.add_argument("--all-tools", type=str, default=ALL_TOOLS_PATH)
    parser.add_argument("--output", type=str, default=SKILL_LIBRARY_PATH)
    parser.add_argument("--max-merges", type=int, default=50)
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--max-macro-len", type=int, default=6)
    parser.add_argument("--success-only", action="store_true", default=True)
    args = parser.parse_args()
    
    # Ensure output dir
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure
    bpe_config = BPEConfig(
        max_merges=args.max_merges,
        min_freq=args.min_freq,
        max_macro_len=args.max_macro_len,
        success_only=args.success_only,
    )
    
    logger.info("=" * 70)
    logger.info("AdaMacro Step 1: BPE Macro Mining")
    logger.info("=" * 70)
    
    # Step 1: Load sequences
    sequences = load_successful_sequences(args.rl_dataset, bpe_config.success_only)
    
    if not sequences:
        logger.error("No sequences found. Check RL dataset path and success_only flag.")
        return
    
    # Step 2: Run BPE mining
    miner = BPEMacroMiner(bpe_config)
    macros = miner.mine(sequences)
    
    # Step 3: Load tool schemas for enrichment
    tool_schemas = load_tool_schemas(args.all_tools)
    
    # Enrich macros with tool descriptions
    for macro_id, macro in macros.items():
        enriched_tools = []
        for tname in macro["tool_names"]:
            if tname in tool_schemas:
                enriched_tools.append({
                    "name": tname,
                    "description": tool_schemas[tname]["description"][:200],
                    "params": tool_schemas[tname]["actual_keys"],
                })
            else:
                enriched_tools.append({"name": tname, "description": "", "params": []})
        macro["tool_details"] = enriched_tools
    
    # Step 4: Save
    output_data = {
        "meta": {
            "algorithm": "BPE-style iterative merging",
            "max_merges": bpe_config.max_merges,
            "min_freq": bpe_config.min_freq,
            "max_macro_len": bpe_config.max_macro_len,
            "success_only": bpe_config.success_only,
            "num_input_sequences": len(sequences),
            "num_macros_discovered": len(macros),
            "atomic_vocab_size": len(miner.atomic_vocab),
            "total_merge_rounds": len(miner.merge_history),
        },
        "macros": macros,
        "merge_history": miner.merge_history,
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved skill library to {args.output}")
    logger.info(f"Total macros: {len(macros)}")
    
    # Print top macros
    sorted_macros = sorted(macros.values(), key=lambda m: m["current_frequency"], reverse=True)
    logger.info("\nTop 10 macros by frequency:")
    for m in sorted_macros[:10]:
        logger.info(f"  {m['macro_id']}: {' -> '.join(m['tool_names'])} "
                    f"(freq={m['current_frequency']}, len={m['length']})")


if __name__ == "__main__":
    main()
