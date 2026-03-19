# extract_tools_v2.py
"""
提取工具定义 V2

功能：
1. 从轨迹文件提取工具及其参数结构变体
2. 从 tool_sequences_cleaned.json 提取工具转移统计 (next_tools)
3. 生成 mcp_rl_graph_v2.json 供 GRPO-ACO 训练使用

输入：
- Toolathlon-Trajectories-merge/*.jsonl: 轨迹文件（提取参数结构）
- tool_sequences_cleaned.json: 工具序列（提取转移统计）

输出：
- all_tools_v2.json: 工具列表（含参数结构变体）
- mcp_rl_graph_v2.json: 工具图（含 next_tools 转移统计）
"""

import json
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple


def normalize_key_pattern(args_raw: Any) -> Tuple[str, Set[str]]:
    """
    提取参数的 key 结构
    
    返回: (pattern_string, key_set)
    """
    if args_raw is None:
        return "<NONE>", set()
    
    # 已经是 dict
    if isinstance(args_raw, dict):
        keys = set(args_raw.keys())
        pattern = "keys:" + ",".join(sorted(keys))
        return pattern, keys
    
    # 字符串：尝试 JSON 解析
    if isinstance(args_raw, str):
        text = args_raw.strip()
        if not text:
            return "<EMPTY_STR>", set()
        try:
            parsed = json.loads(text)
        except Exception:
            return "<NON_JSON_STR>", set()
        
        if isinstance(parsed, dict):
            keys = set(parsed.keys())
            pattern = "keys:" + ",".join(sorted(keys))
            return pattern, keys
        else:
            return "<NON_DICT_JSON>", set()
    
    # 其他类型
    if isinstance(args_raw, list):
        return "<LIST>", set()
    return "<OTHER>", set()


def hash_key_set(keys: Set[str]) -> str:
    """对参数 key 集合进行哈希"""
    sorted_keys = sorted(keys)
    return hashlib.md5(str(sorted_keys).encode()).hexdigest()[:12]


def extract_tools_from_trajectories(input_dir: str) -> Tuple[Dict, Dict]:
    """
    从轨迹文件中提取工具定义和参数模式
    
    返回:
        tool_base_info: {tool_name: {description, parameters_schema}}
        tool_key_patterns: {tool_name: {key_pattern: {keys, count, example_args}}}
    """
    input_dir = Path(input_dir)
    
    # 工具基本信息
    tool_base_info: Dict[str, Dict] = {}
    
    # 参数模式统计
    tool_key_patterns: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    
    file_count = 0
    call_count = 0
    
    for file in sorted(input_dir.glob("*.jsonl")):
        file_count += 1
        print(f"扫描文件: {file.name}")
        
        with file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # ========== 读取工具定义 ==========
                tc_raw = obj.get("tool_calls")
                if tc_raw:
                    if isinstance(tc_raw, str):
                        try:
                            tc = json.loads(tc_raw)
                        except:
                            tc = {}
                    elif isinstance(tc_raw, dict):
                        tc = tc_raw
                    else:
                        tc = {}
                    
                    tools_def = tc.get("tools") or []
                    for t in tools_def:
                        if not isinstance(t, dict):
                            continue
                        func = t.get("function") or {}
                        name = func.get("name")
                        if not name:
                            continue
                        
                        if name not in tool_base_info:
                            tool_base_info[name] = {
                                "type": t.get("type", "function"),
                                "description": func.get("description", ""),
                                "parameters_schema": func.get("parameters", {}),
                            }
                
                # ========== 读取实际调用的参数 ==========
                messages = obj.get("messages")
                if isinstance(messages, str):
                    try:
                        messages = json.loads(messages)
                    except:
                        messages = []
                
                if not isinstance(messages, list):
                    continue
                
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("role") != "assistant":
                        continue
                    
                    tool_calls = msg.get("tool_calls")
                    if not tool_calls:
                        continue
                    
                    if isinstance(tool_calls, str):
                        try:
                            tool_calls = json.loads(tool_calls)
                        except:
                            continue
                    
                    if isinstance(tool_calls, dict):
                        tool_calls = [tool_calls]
                    
                    if not isinstance(tool_calls, list):
                        continue
                    
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        
                        func = tc.get("function") or {}
                        name = func.get("name")
                        if not name:
                            continue
                        
                        args_raw = func.get("arguments")
                        pattern, keys = normalize_key_pattern(args_raw)
                        
                        call_count += 1
                        
                        if pattern not in tool_key_patterns[name]:
                            tool_key_patterns[name][pattern] = {
                                "keys": keys,
                                "count": 0,
                                "example_args": args_raw,
                            }
                        tool_key_patterns[name][pattern]["count"] += 1
    
    print(f"\n扫描完成: {file_count} 文件, {call_count} 调用")
    print(f"工具定义数: {len(tool_base_info)}")
    print(f"有实际调用的工具数: {len(tool_key_patterns)}")
    
    return tool_base_info, dict(tool_key_patterns)


def load_tool_sequences(sequences_path: str) -> Dict[str, Dict[str, int]]:
    """
    从 tool_sequences_cleaned.json 加载工具转移统计
    
    返回: {tool_name: {next_tool_name: count}}
    """
    with open(sequences_path, 'r', encoding='utf-8') as f:
        sequences = json.load(f)
    
    # 统计转移
    transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    for seq_record in sequences:
        tool_sequence = seq_record.get("tool_sequence", [])
        if not tool_sequence or len(tool_sequence) < 2:
            continue
        
        for i in range(len(tool_sequence) - 1):
            curr_tool = tool_sequence[i]
            next_tool = tool_sequence[i + 1]
            transitions[curr_tool][next_tool] += 1
    
    # 转换为普通 dict
    result = {}
    for tool, nexts in transitions.items():
        result[tool] = dict(nexts)
    
    print(f"加载工具序列: {len(sequences)} 条")
    print(f"有转移记录的工具数: {len(result)}")
    
    return result


def build_tools_list(
    tool_base_info: Dict,
    tool_key_patterns: Dict,
) -> List[Dict]:
    """
    构建工具列表（含参数变体）
    """
    tools_list = []
    
    single_pattern_count = 0
    multi_pattern_count = 0
    
    for tool_name in sorted(tool_key_patterns.keys()):
        patterns = tool_key_patterns[tool_name]
        base_info = tool_base_info.get(tool_name, {
            "type": "function",
            "description": "",
            "parameters_schema": {},
        })
        
        # 按调用次数排序
        sorted_patterns = sorted(
            patterns.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        
        if len(sorted_patterns) == 1:
            single_pattern_count += 1
            pattern, info = sorted_patterns[0]
            
            tools_list.append({
                "name": tool_name,
                "original_name": tool_name,
                "type": base_info["type"],
                "description": base_info["description"],
                "parameters_schema": base_info["parameters_schema"],
                "actual_keys": sorted(info["keys"]),
                "key_pattern": pattern,
                "key_hash": hash_key_set(info["keys"]),
                "call_count": info["count"],
            })
        else:
            multi_pattern_count += 1
            
            for idx, (pattern, info) in enumerate(sorted_patterns):
                if idx == 0:
                    unique_name = tool_name
                else:
                    unique_name = f"{tool_name}_v{idx}"
                
                tools_list.append({
                    "name": unique_name,
                    "original_name": tool_name,
                    "type": base_info["type"],
                    "description": base_info["description"],
                    "parameters_schema": base_info["parameters_schema"],
                    "actual_keys": sorted(info["keys"]),
                    "key_pattern": pattern,
                    "key_hash": hash_key_set(info["keys"]),
                    "call_count": info["count"],
                    "variant_index": idx,
                })
    
    # 按名称排序并添加 ID
    tools_list.sort(key=lambda x: x["name"])
    for idx, tool in enumerate(tools_list):
        tool["id"] = idx
    
    print(f"\n只有一种参数模式的工具: {single_pattern_count}")
    print(f"有多种参数模式的工具: {multi_pattern_count}")
    print(f"总工具数（含变体）: {len(tools_list)}")
    
    return tools_list


def generate_mcp_graph(
    tools_list: List[Dict],
    transitions: Dict[str, Dict[str, int]],
    output_path: str,
):
    """
    生成 MCP 图（含 next_tools）
    
    注意：transitions 是基于 original_name 的，需要映射到变体
    """
    # 构建 original_name -> [变体名列表] 的映射
    original_to_variants: Dict[str, List[str]] = defaultdict(list)
    for tool in tools_list:
        original_to_variants[tool["original_name"]].append(tool["name"])
    
    # 构建 name -> id 映射
    name_to_id = {tool["name"]: tool["id"] for tool in tools_list}
    
    # 构建 MCP 工具列表
    mcp_tools = []
    
    for tool in tools_list:
        original_name = tool["original_name"]
        
        # 从 transitions 获取 next_tools
        # transitions 是基于 original_name 的
        raw_next = transitions.get(original_name, {})
        
        # 将 next_tools 映射到实际的工具名（取第一个变体，即最常用的）
        next_tools = {}
        for next_original, count in raw_next.items():
            # 找到 next_original 对应的变体
            variants = original_to_variants.get(next_original, [])
            if variants:
                # 取第一个（最常用的）
                next_tools[variants[0]] = count
        
        # 构建 params 字段（从 actual_keys 构建）
        params = {}
        for key in tool.get("actual_keys", []):
            params[key] = {
                "type": "string",  # 默认类型
                "required": True,
                "description": "",
            }
        
        # 尝试从 parameters_schema 获取更详细的信息
        schema = tool.get("parameters_schema", {})
        properties = schema.get("properties", {})
        required_list = schema.get("required", [])
        
        for key in params:
            if key in properties:
                prop = properties[key]
                params[key]["type"] = prop.get("type", "string")
                params[key]["description"] = prop.get("description", "")
            params[key]["required"] = key in required_list
        
        mcp_tools.append({
            "id": tool["id"],
            "name": tool["name"],
            "original_name": tool["original_name"],
            "description": tool.get("description", ""),
            "params": params,
            "actual_keys": tool.get("actual_keys", []),
            "key_hash": tool.get("key_hash", ""),
            "call_count": tool.get("call_count", 0),
            "next_tools": next_tools,
        })
    
    # 统计
    tools_with_next = sum(1 for t in mcp_tools if t["next_tools"])
    total_edges = sum(len(t["next_tools"]) for t in mcp_tools)
    
    output = {
        "tools": mcp_tools,
        "meta": {
            "total_tools": len(mcp_tools),
            "tools_with_transitions": tools_with_next,
            "total_edges": total_edges,
            "source": "extract_tools_v2.py",
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nMCP 图已生成: {output_path}")
    print(f"工具数: {len(mcp_tools)}")
    print(f"有转移记录的工具: {tools_with_next}")
    print(f"总边数: {total_edges}")


def main():
    PROJECT_ROOT = "/seu_share2/home/fenglei/230250004/Agent_Tool/tool-use/tool-use"
    
    print("=" * 70)
    print("提取工具定义 V2")
    print("=" * 70)
    
    # Step 1: 从轨迹提取工具和参数模式
    print("\n[Step 1] 从轨迹文件提取工具...")
    tool_base_info, tool_key_patterns = extract_tools_from_trajectories(
        f"{PROJECT_ROOT}/Toolathlon-Trajectories-merge"
    )
    
    # Step 2: 加载工具转移序列
    print("\n[Step 2] 加载工具转移序列...")
    transitions = load_tool_sequences(
        f"{PROJECT_ROOT}/json_file/tool_sequences_cleaned.json"
    )
    
    # Step 3: 构建工具列表
    print("\n[Step 3] 构建工具列表...")
    tools_list = build_tools_list(tool_base_info, tool_key_patterns)
    
    # Step 4: 保存 all_tools_v2.json
    all_tools_path = f"{PROJECT_ROOT}/json_file/all_tools_v2.json"
    with open(all_tools_path, 'w', encoding='utf-8') as f:
        json.dump(tools_list, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {all_tools_path}")
    
    # Step 5: 生成 MCP 图
    print("\n[Step 5] 生成 MCP 图...")
    generate_mcp_graph(
        tools_list,
        transitions,
        f"{PROJECT_ROOT}/json_file/mcp_rl_graph_v2.json",
    )
    
    # 打印有多种参数模式的工具示例
    print(f"\n{'='*60}")
    print("有多种参数模式的工具（前 10 个）")
    print(f"{'='*60}")
    
    multi_pattern_tools = [
        (name, patterns)
        for name, patterns in tool_key_patterns.items()
        if len(patterns) > 1
    ]
    multi_pattern_tools.sort(key=lambda x: len(x[1]), reverse=True)
    
    for name, patterns in multi_pattern_tools[:10]:
        print(f"\n{name}: {len(patterns)} 种参数模式")
        for pattern, info in sorted(patterns.items(), key=lambda x: x[1]["count"], reverse=True)[:3]:
            print(f"  - {pattern}: {info['count']} 次")
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()