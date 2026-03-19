# generate_rl_dataset_v3.py
"""
生成新版 RL Dataset (v3)

基于 v2 版本，新增功能：
- 去除连续重复的工具调用（如连续6次 get-tickets 只保留1次）

V3 更新：
- 添加 deduplicate_consecutive 函数去除连续重复
- 同步处理 tool_ids, tool_names, tool_args, output_texts, param_hashes
- 记录去重统计信息

使用方法：
1. 从原始轨迹重新生成（推荐）:
   python generate_rl_dataset_v3.py
   
2. 或者基于已有的 v2 JSON 进行转换:
   python generate_rl_dataset_v3.py --from-v2 path/to/rl_dataset_llm_v2.json
"""

import json
import hashlib
import os
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm


def compute_param_hash(args: Any) -> str:
    """计算参数的 schema hash（基于参数键）"""
    keys = set()
    if isinstance(args, dict):
        keys = set(args.keys())
    elif isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                keys = set(parsed.keys())
        except:
            pass
    
    if not keys:
        return ""
    return hashlib.md5(str(sorted(keys)).encode()).hexdigest()[:12]


def deduplicate_consecutive(
    tool_ids: List[int],
    tool_names: List[str],
    tool_args: List[str],
    output_texts: List[str],
    param_hashes: List[str],
) -> Tuple[List[int], List[str], List[str], List[str], List[str], int]:
    """
    去除连续重复的工具调用
    
    例如: [A, A, A, B, C, C] -> [A, B, C]
    
    Args:
        tool_ids: 工具ID列表
        tool_names: 工具名列表
        tool_args: 工具参数列表
        output_texts: 输出文本列表
        param_hashes: 参数哈希列表
    
    Returns:
        去重后的各列表 + 被移除的数量
    """
    if not tool_ids:
        return [], [], [], [], [], 0
    
    new_ids = [tool_ids[0]]
    new_names = [tool_names[0]]
    new_args = [tool_args[0] if tool_args else "{}"]
    new_outputs = [output_texts[0] if output_texts else ""]
    new_hashes = [param_hashes[0] if param_hashes else ""]
    
    removed_count = 0
    
    for i in range(1, len(tool_ids)):
        # 只有当 tool_id 与前一个不同时才添加
        if tool_ids[i] != tool_ids[i - 1]:
            new_ids.append(tool_ids[i])
            new_names.append(tool_names[i] if i < len(tool_names) else "")
            new_args.append(tool_args[i] if i < len(tool_args) else "{}")
            new_outputs.append(output_texts[i] if i < len(output_texts) else "")
            new_hashes.append(param_hashes[i] if i < len(param_hashes) else "")
        else:
            removed_count += 1
    
    return new_ids, new_names, new_args, new_outputs, new_hashes, removed_count


def load_mcp_graph_v2(path: str) -> Tuple[Dict[str, Dict[str, int]], Dict[int, str], int]:
    """
    加载 mcp_rl_graph_v2.json
    
    返回:
        tool_variant_map: {original_name: {param_hash: tool_id}}
        id_to_name: {tool_id: name}
        num_tools: 工具总数
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    tools = data.get("tools", [])
    
    # 构建映射: original_name -> {param_hash -> tool_id}
    tool_variant_map: Dict[str, Dict[str, int]] = defaultdict(dict)
    id_to_name: Dict[int, str] = {}
    
    for tool in tools:
        tool_id = tool.get("id")
        name = tool.get("name")
        original_name = tool.get("original_name", name)
        key_hash = tool.get("key_hash", "")
        
        if tool_id is None or name is None:
            continue
        
        id_to_name[tool_id] = name
        
        # 映射: original_name + param_hash -> tool_id
        if key_hash:
            tool_variant_map[original_name][key_hash] = tool_id
        else:
            # 没有 key_hash 的工具，用空字符串作为默认
            tool_variant_map[original_name][""] = tool_id
    
    print(f"[load_mcp_graph_v2] Loaded {len(tools)} tools")
    print(f"[load_mcp_graph_v2] Unique original names: {len(tool_variant_map)}")
    
    # 统计有多少工具有多个变体
    multi_variant = sum(1 for v in tool_variant_map.values() if len(v) > 1)
    print(f"[load_mcp_graph_v2] Tools with multiple variants: {multi_variant}")
    
    return dict(tool_variant_map), id_to_name, len(tools)


def resolve_tool_id(
    tool_name: str,
    arguments: Any,
    tool_variant_map: Dict[str, Dict[str, int]],
) -> Optional[int]:
    """
    根据工具名和参数解析 tool_id
    
    如果有多个变体，根据参数的 key_hash 匹配
    如果没有匹配的变体，返回默认变体（如果存在）
    """
    if tool_name not in tool_variant_map:
        return None
    
    variants = tool_variant_map[tool_name]
    
    # 计算参数的 key_hash
    param_hash = compute_param_hash(arguments)
    
    # 精确匹配
    if param_hash in variants:
        return variants[param_hash]
    
    # 没有精确匹配，尝试找最佳匹配
    # 1. 尝试空 hash（默认变体）
    if "" in variants:
        return variants[""]
    
    # 2. 返回第一个变体
    return next(iter(variants.values()))


def parse_arguments(args_str: str) -> Dict:
    """解析参数字符串为字典"""
    if not args_str:
        return {}
    
    try:
        parsed = json.loads(args_str)
        if isinstance(parsed, dict):
            return parsed
    except:
        pass
    
    return {}


def process_trajectory_file(
    filepath: str,
    tool_variant_map: Dict[str, Dict[str, int]],
    enable_dedup: bool = True,
) -> Tuple[List[Dict], int, int]:
    """
    处理单个轨迹文件
    
    Args:
        filepath: 轨迹文件路径
        tool_variant_map: 工具变体映射
        enable_dedup: 是否启用去重
    
    Returns:
        episodes: episode 列表
        total_removed: 总共移除的重复调用数
        total_original: 原始总调用数
    """
    episodes = []
    total_removed = 0
    total_original = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            try:
                traj = json.loads(line.strip())
            except:
                continue
            
            # 提取基本信息
            task_name = traj.get("task_name", "")
            
            # task_status 是 JSON 字符串，需要解析
            task_status_str = traj.get("task_status", "{}")
            try:
                task_status = json.loads(task_status_str) if isinstance(task_status_str, str) else task_status_str
                
                # 【V2.1 修复】正确的成功判断：
                # running == "done" AND evaluation == true
                running_done = task_status.get("running") == "done"
                evaluation_passed = task_status.get("evaluation") == True
                success = 1 if (running_done and evaluation_passed) else 0
                
            except:
                success = 0
            
            # messages 是 JSON 字符串，需要解析
            messages_str = traj.get("messages", "[]")
            try:
                messages = json.loads(messages_str) if isinstance(messages_str, str) else messages_str
            except:
                messages = []
            
            if not isinstance(messages, list):
                continue
            
            # 提取 user_prompt（第一条 user 消息）
            user_prompt = ""
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        user_prompt = content
                    elif isinstance(content, list):
                        # content 可能是 list of {type, text}
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                user_prompt = item.get("text", "")
                                break
                    break
            
            # 提取工具调用序列
            tool_ids = []
            tool_names = []
            tool_args = []
            output_texts = []
            param_hashes = []
            
            # 收集所有工具调用和输出
            tool_call_map = {}  # tool_call_id -> (name, args)
            
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                
                role = msg.get("role", "")
                
                if role == "assistant":
                    tool_calls = msg.get("tool_calls", [])
                    if not isinstance(tool_calls, list):
                        continue
                    
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        
                        tc_id = tc.get("id", "")
                        func = tc.get("function", {})
                        if not isinstance(func, dict):
                            continue
                        
                        name = func.get("name", "")
                        args_str = func.get("arguments", "")
                        
                        if not name:
                            continue
                        
                        # 解析参数
                        args = parse_arguments(args_str)
                        
                        # 解析 tool_id（考虑参数变体）
                        tool_id = resolve_tool_id(name, args, tool_variant_map)
                        
                        if tool_id is None:
                            # 工具名不在图中，跳过
                            continue
                        
                        tool_ids.append(tool_id)
                        tool_names.append(name)
                        tool_args.append(args_str if args_str else "{}")
                        param_hashes.append(compute_param_hash(args))
                        
                        # 记录 tool_call_id 用于匹配输出
                        tool_call_map[tc_id] = len(tool_ids) - 1
                
                elif role == "tool":
                    # 工具输出
                    tc_id = msg.get("tool_call_id", "")
                    content = msg.get("content", "")
                    
                    if isinstance(content, list):
                        # 可能是 [{type: text, text: ...}]
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                        content = " ".join(text_parts)
                    
                    output_texts.append(str(content)[:500])  # 截断
            
            # 确保输出数量匹配
            while len(output_texts) < len(tool_ids):
                output_texts.append("")
            output_texts = output_texts[:len(tool_ids)]
            
            if not tool_ids:
                continue
            
            # 记录原始数量
            original_len = len(tool_ids)
            total_original += original_len
            
            # 【V3 核心】去除连续重复
            if enable_dedup:
                (tool_ids, tool_names, tool_args, output_texts, param_hashes, removed) = \
                    deduplicate_consecutive(tool_ids, tool_names, tool_args, output_texts, param_hashes)
                total_removed += removed
            
            episodes.append({
                "task_name": task_name,
                "user_prompt": user_prompt[:2000],  # 截断
                "success": int(success),
                "tool_ids": tool_ids,
                "tool_names": tool_names,
                "tool_args": tool_args,
                "output_texts": output_texts,
                "param_hashes": param_hashes,
                "original_length": original_len,  # 记录原始长度（用于分析）
                "dedup_length": len(tool_ids),    # 去重后长度
            })
    
    return episodes, total_removed, total_original


def convert_from_v2(v2_path: str, output_path: str):
    """
    从已有的 v2 JSON 文件转换为 v3
    
    这是一个快速转换方法，不需要重新处理原始轨迹
    """
    print("=" * 70)
    print("Converting RL Dataset V2 -> V3")
    print("=" * 70)
    
    with open(v2_path, 'r') as f:
        v2_data = json.load(f)
    
    v2_episodes = v2_data.get("episodes", [])
    print(f"[Input] Loaded {len(v2_episodes)} episodes from V2")
    
    v3_episodes = []
    total_removed = 0
    total_original = 0
    
    for ep in tqdm(v2_episodes, desc="Deduplicating"):
        tool_ids = ep.get("tool_ids", [])
        tool_names = ep.get("tool_names", [])
        tool_args = ep.get("tool_args", [])
        output_texts = ep.get("output_texts", [])
        param_hashes = ep.get("param_hashes", [])
        
        original_len = len(tool_ids)
        total_original += original_len
        
        # 去重
        (new_ids, new_names, new_args, new_outputs, new_hashes, removed) = \
            deduplicate_consecutive(tool_ids, tool_names, tool_args, output_texts, param_hashes)
        
        total_removed += removed
        
        v3_episodes.append({
            "task_name": ep.get("task_name", ""),
            "user_prompt": ep.get("user_prompt", ""),
            "success": ep.get("success", 0),
            "tool_ids": new_ids,
            "tool_names": new_names,
            "tool_args": new_args,
            "output_texts": new_outputs,
            "param_hashes": new_hashes,
            "original_length": original_len,
            "dedup_length": len(new_ids),
        })
    
    # 统计
    print(f"\n[Dedup Stats]")
    print(f"  Original tool calls: {total_original}")
    print(f"  After dedup:         {total_original - total_removed}")
    print(f"  Removed:             {total_removed} ({100*total_removed/total_original:.1f}%)")
    
    # 计算平均序列长度变化
    avg_original = sum(ep["original_length"] for ep in v3_episodes) / len(v3_episodes)
    avg_dedup = sum(ep["dedup_length"] for ep in v3_episodes) / len(v3_episodes)
    print(f"  Avg length before:   {avg_original:.2f}")
    print(f"  Avg length after:    {avg_dedup:.2f}")
    
    # 保存
    success_count = sum(1 for ep in v3_episodes if ep["success"])
    
    output_data = {
        "meta": {
            "version": "v3",
            "description": "Deduplicated consecutive tool calls",
            "num_episodes": len(v3_episodes),
            "num_tools": v2_data.get("meta", {}).get("num_tools", 0),
            "success_episodes": success_count,
            "source": "Converted from V2",
            "success_criteria": "running=='done' AND evaluation==true",
            "features": [
                "tool_variants",
                "param_hashes",
                "consecutive_dedup",  # V3 新特性
            ],
            "dedup_stats": {
                "original_tool_calls": total_original,
                "after_dedup": total_original - total_removed,
                "removed": total_removed,
                "removal_rate": f"{100*total_removed/total_original:.1f}%",
                "avg_length_before": round(avg_original, 2),
                "avg_length_after": round(avg_dedup, 2),
            },
        },
        "episodes": v3_episodes,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n[Saved] {output_path}")
    print("=" * 70)


def generate_rl_dataset_v3(
    trajectories_dir: str,
    mcp_graph_path: str,
    output_path: str,
):
    """
    从原始轨迹生成 RL Dataset V3
    """
    print("=" * 70)
    print("Generating RL Dataset V3")
    print("Feature: Deduplicate consecutive tool calls")
    print("=" * 70)
    
    # 1. 加载工具图
    tool_variant_map, id_to_name, num_tools = load_mcp_graph_v2(mcp_graph_path)
    
    # 2. 处理轨迹文件
    trajectories_dir = Path(trajectories_dir)
    all_episodes = []
    total_removed = 0
    total_original = 0
    
    jsonl_files = list(trajectories_dir.glob("*.jsonl"))
    print(f"\n[Processing] Found {len(jsonl_files)} trajectory files")
    
    for filepath in tqdm(jsonl_files, desc="Processing trajectories"):
        episodes, removed, original = process_trajectory_file(
            str(filepath), tool_variant_map, enable_dedup=True
        )
        all_episodes.extend(episodes)
        total_removed += removed
        total_original += original
    
    print(f"\n[Result] Generated {len(all_episodes)} episodes")
    
    # 3. 去重统计
    print(f"\n[Dedup Stats]")
    print(f"  Original tool calls: {total_original}")
    print(f"  After dedup:         {total_original - total_removed}")
    print(f"  Removed:             {total_removed} ({100*total_removed/max(1,total_original):.1f}%)")
    
    if all_episodes:
        avg_original = sum(ep["original_length"] for ep in all_episodes) / len(all_episodes)
        avg_dedup = sum(ep["dedup_length"] for ep in all_episodes) / len(all_episodes)
        print(f"  Avg length before:   {avg_original:.2f}")
        print(f"  Avg length after:    {avg_dedup:.2f}")
    else:
        avg_original = avg_dedup = 0
    
    # 4. 其他统计
    success_count = sum(1 for ep in all_episodes if ep["success"])
    total_tools_after = sum(len(ep["tool_ids"]) for ep in all_episodes)
    unique_tools = set()
    for ep in all_episodes:
        unique_tools.update(ep["tool_ids"])
    
    print(f"\n[General Stats]")
    print(f"  Success episodes: {success_count}/{len(all_episodes)} ({100*success_count/max(1,len(all_episodes)):.1f}%)")
    print(f"  Total tool calls (after dedup): {total_tools_after}")
    print(f"  Unique tools used: {len(unique_tools)}")
    if unique_tools:
        print(f"  Tool ID range: {min(unique_tools)} - {max(unique_tools)}")
    
    # 5. 保存
    output_data = {
        "meta": {
            "version": "v3",
            "description": "Deduplicated consecutive tool calls",
            "num_episodes": len(all_episodes),
            "num_tools": num_tools,
            "success_episodes": success_count,
            "source": "Toolathlon-Trajectories",
            "success_criteria": "running=='done' AND evaluation==true",
            "features": [
                "tool_variants",
                "param_hashes",
                "consecutive_dedup",  # V3 新特性
            ],
            "dedup_stats": {
                "original_tool_calls": total_original,
                "after_dedup": total_original - total_removed,
                "removed": total_removed,
                "removal_rate": f"{100*total_removed/max(1,total_original):.1f}%",
                "avg_length_before": round(avg_original, 2),
                "avg_length_after": round(avg_dedup, 2),
            },
        },
        "episodes": all_episodes,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n[Saved] {output_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate RL Dataset V3 (with consecutive dedup)")
    parser.add_argument(
        "--from-v2",
        type=str,
        default=None,
        help="Convert from existing V2 JSON file instead of regenerating from trajectories"
    )
    parser.add_argument(
        "--trajectories-dir",
        type=str,
        default=None,
        help="Directory containing trajectory JSONL files"
    )
    parser.add_argument(
        "--mcp-graph",
        type=str,
        default=None,
        help="Path to mcp_rl_graph_v2.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for V3 JSON file"
    )
    
    args = parser.parse_args()
    
    # 默认路径
    PROJECT_ROOT = Path("/seu_share2/home/fenglei/230250004/Agent_Tool/tool-use/tool-use")
    
    if args.from_v2:
        # 从 V2 转换
        output_path = args.output or str(PROJECT_ROOT / "GRPO-ACO" / "data" / "rl_dataset_llm_v3.json")
        convert_from_v2(args.from_v2, output_path)
    else:
        # 从原始轨迹生成
        trajectories_dir = args.trajectories_dir or str(PROJECT_ROOT / "Toolathlon-Trajectories")
        mcp_graph_path = args.mcp_graph or str(PROJECT_ROOT / "json_file" / "mcp_rl_graph_v2.json")
        output_path = args.output or str(PROJECT_ROOT / "GRPO-ACO" / "data" / "rl_dataset_llm_v3.json")
        
        generate_rl_dataset_v3(trajectories_dir, mcp_graph_path, output_path)


if __name__ == "__main__":
    main()