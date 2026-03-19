# generate_tool_simulator_data.py
"""
工具模拟输出数据生成器

从历史轨迹中提取工具调用，生成 JSON 格式的工具模拟数据库。

输入：
- all_tools.json: 工具定义
- mcp_rl_graph.json: 工具ID映射
- Toolathlon-Trajectories-merge/*.jsonl 或 all_messages.json: 历史轨迹

输出：
- tool_simulator_database.json: 完整的工具模拟数据库
  格式: {
    "meta": {...},
    "tools": {
      tool_name: {
        "tool_id": int,
        "schemas": {
          schema_hash: {
            "keys": ["param1", "param2"],
            "count": int,
            "pheromone": float,
            "calls": [
              {
                "args": {...},
                "output": "...",
                "value_text": "...",   # 用于 embedding 匹配
                "embedding": [...]      # 可选
              }
            ]
          }
        }
      }
    }
  }

使用:
    python generate_tool_simulator_data.py --source trajectories
    python generate_tool_simulator_data.py --source all_messages
    python generate_tool_simulator_data.py --source both
"""

import json
import hashlib
import math
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any, Optional, Tuple
import numpy as np


# ============================================================================
# Part 1: 数据解析
# ============================================================================

def parse_arguments(args: Any) -> Dict:
    """解析工具参数，统一返回 dict"""
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        args = args.strip()
        if not args:
            return {}
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def extract_param_keys(args: Any) -> Set[str]:
    """提取参数的 key 集合"""
    return set(parse_arguments(args).keys())


def hash_param_keys(keys: Set[str]) -> str:
    """对参数 key 集合进行哈希"""
    sorted_keys = sorted(keys)
    return hashlib.md5(str(sorted_keys).encode()).hexdigest()[:12]


def extract_value_text(args: Any) -> str:
    """
    提取参数值的文本表示，用于计算 embedding 和相似度匹配
    """
    parsed = parse_arguments(args)
    if not parsed:
        return ""
    
    parts = []
    for key in sorted(parsed.keys()):
        value = parsed[key]
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.extend(str(v) for v in value if v)
        elif value is not None:
            parts.append(str(value))
    
    return " ".join(parts)


def extract_tool_output(content: Any) -> str:
    """从 tool message 的 content 中提取输出文本"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def extract_tool_calls_from_messages(messages: List[Dict]) -> List[Dict]:
    """从 messages 列表中提取所有工具调用"""
    if not messages or not isinstance(messages, list):
        return []
    
    # 构建 tool_call_id -> output 映射
    id_to_output = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        tc_id = msg.get("tool_call_id")
        if tc_id:
            id_to_output[tc_id] = extract_tool_output(msg.get("content"))
    
    # 提取工具调用
    calls = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            continue
        
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except json.JSONDecodeError:
                continue
        
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]
        
        if not isinstance(tool_calls, list):
            continue
        
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            
            func = tc.get("function") or {}
            tool_name = func.get("name")
            tc_id = tc.get("id")
            arguments = func.get("arguments", {})
            
            if not tool_name:
                continue
            
            output = id_to_output.get(tc_id, "") if tc_id else ""
            if output and len(output.strip()) > 1:
                calls.append({
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "output": output,
                })
    
    return calls


def parse_trajectory_record(record: Dict) -> List[Dict]:
    """解析一条轨迹记录"""
    messages = record.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return []
    
    if not isinstance(messages, list):
        return []
    
    return extract_tool_calls_from_messages(messages)


# ============================================================================
# Part 2: 简单文本 Embedding（TF-IDF）
# ============================================================================

class SimpleTextEmbedder:
    """简单的 TF-IDF 文本 embedding"""
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.word_to_idx: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.fitted = False
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b|[\u4e00-\u9fff]+', text)
        return tokens
    
    def fit(self, texts: List[str]):
        """拟合词表和 IDF"""
        from collections import Counter
        
        word_counts = Counter()
        doc_counts = Counter()
        
        for text in texts:
            tokens = self._tokenize(text)
            word_counts.update(tokens)
            doc_counts.update(set(tokens))
        
        most_common = word_counts.most_common(self.vocab_size)
        self.word_to_idx = {word: idx for idx, (word, _) in enumerate(most_common)}
        
        n_docs = len(texts) + 1
        for word, count in doc_counts.items():
            if word in self.word_to_idx:
                self.idf[word] = math.log(n_docs / (count + 1)) + 1
        
        self.fitted = True
    
    def embed(self, text: str) -> List[float]:
        """计算文本 embedding，返回 list"""
        if not self.fitted:
            return []
        
        tokens = self._tokenize(text)
        vector = [0.0] * len(self.word_to_idx)
        
        token_counts = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1
        
        for token, count in token_counts.items():
            if token in self.word_to_idx:
                idx = self.word_to_idx[token]
                tf = count / max(len(tokens), 1)
                idf = self.idf.get(token, 1.0)
                vector[idx] = tf * idf
        
        # L2 归一化
        norm = math.sqrt(sum(x*x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        
        return vector
    
    def save_vocab(self, path: str):
        """保存词表"""
        data = {
            "word_to_idx": self.word_to_idx,
            "idf": self.idf,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    
    def load_vocab(self, path: str):
        """加载词表"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.word_to_idx = data.get("word_to_idx", {})
        self.idf = data.get("idf", {})
        self.fitted = bool(self.word_to_idx)


# ============================================================================
# Part 3: 工具模拟数据库构建器
# ============================================================================

class ToolSimulatorDatabaseBuilder:
    """
    工具模拟数据库构建器
    """
    
    def __init__(
        self,
        max_calls_per_schema: int = 30,
        max_output_length: int = 1500,
        compute_embedding: bool = True,
        embedding_dim: int = 500,  # 降低维度以减小文件大小
    ):
        self.max_calls_per_schema = max_calls_per_schema
        self.max_output_length = max_output_length
        self.compute_embedding = compute_embedding
        self.embedding_dim = embedding_dim
        
        # tool_name -> schema_hash -> data
        self.data: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(dict))
        
        # 工具ID映射
        self.name_to_id: Dict[str, int] = {}
        
        # Embedder
        self.embedder = SimpleTextEmbedder(vocab_size=embedding_dim) if compute_embedding else None
        self._value_texts: List[str] = []  # 用于拟合 embedder
        
        # 统计
        self.total_calls = 0
    
    def set_tool_mapping(self, name_to_id: Dict[str, int]):
        """设置工具名到ID的映射"""
        self.name_to_id = name_to_id
    
    def add_call(self, tool_name: str, arguments: Any, output: str):
        """添加一条工具调用记录"""
        if not output or len(output.strip()) < 2:
            return
        
        # 解析参数
        args_dict = parse_arguments(arguments)
        keys = extract_param_keys(arguments)
        schema_hash = hash_param_keys(keys)
        value_text = extract_value_text(arguments)
        
        # 截断输出
        output = output[:self.max_output_length]
        
        # 初始化 schema
        if schema_hash not in self.data[tool_name]:
            self.data[tool_name][schema_hash] = {
                "keys": sorted(keys),
                "count": 0,
                "calls": [],
            }
        
        schema_data = self.data[tool_name][schema_hash]
        schema_data["count"] += 1
        self.total_calls += 1
        
        # 添加调用记录（限制数量）
        if len(schema_data["calls"]) < self.max_calls_per_schema:
            call_record = {
                "args": args_dict,
                "output": output,
                "value_text": value_text,
            }
            schema_data["calls"].append(call_record)
            
            # 收集 value_text 用于拟合 embedder
            if value_text and self.compute_embedding:
                self._value_texts.append(value_text)
    
    def compute_pheromones(self):
        """计算参数信息素"""
        for tool_name, schemas in self.data.items():
            total_count = sum(s["count"] for s in schemas.values())
            
            for schema_hash, schema_data in schemas.items():
                count = schema_data["count"]
                freq = count / total_count if total_count > 0 else 0
                # τ = 1 + log(1 + count) * freq * 2
                tau = 1.0 + math.log(1 + count) * freq * 2
                schema_data["pheromone"] = round(tau, 4)
    
    def compute_embeddings(self):
        """计算所有 value_text 的 embedding"""
        if not self.compute_embedding or not self._value_texts:
            return
        
        print(f"[Embedding] Fitting on {len(self._value_texts)} texts...")
        self.embedder.fit(self._value_texts)
        
        # 计算每个 call 的 embedding
        for tool_name, schemas in self.data.items():
            for schema_hash, schema_data in schemas.items():
                for call in schema_data["calls"]:
                    value_text = call.get("value_text", "")
                    if value_text:
                        emb = self.embedder.embed(value_text)
                        # 只保留非零元素以节省空间
                        sparse_emb = {i: round(v, 4) for i, v in enumerate(emb) if abs(v) > 0.001}
                        call["embedding"] = sparse_emb
        
        print(f"[Embedding] Computed embeddings for all calls")
    
    def build(self) -> Dict:
        """构建最终的数据库"""
        self.compute_pheromones()
        if self.compute_embedding:
            self.compute_embeddings()
        
        # 构建输出
        tools_data = {}
        for tool_name, schemas in self.data.items():
            tool_id = self.name_to_id.get(tool_name, -1)
            
            schemas_data = {}
            for schema_hash, schema_data in schemas.items():
                schemas_data[schema_hash] = {
                    "keys": schema_data["keys"],
                    "count": schema_data["count"],
                    "pheromone": schema_data.get("pheromone", 1.0),
                    "calls": schema_data["calls"],
                }
            
            tools_data[tool_name] = {
                "tool_id": tool_id,
                "schemas": schemas_data,
            }
        
        return {
            "meta": {
                "version": "2.0",
                "total_tools": len(self.data),
                "total_calls": self.total_calls,
                "total_schemas": sum(len(s) for s in self.data.values()),
                "has_embeddings": self.compute_embedding,
                "embedding_dim": self.embedding_dim if self.compute_embedding else 0,
            },
            "tools": tools_data,
        }
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        schema_counts = [len(schemas) for schemas in self.data.values()]
        return {
            "num_tools": len(self.data),
            "total_calls": self.total_calls,
            "total_schemas": sum(schema_counts),
            "avg_schemas_per_tool": sum(schema_counts) / len(schema_counts) if schema_counts else 0,
        }


# ============================================================================
# Part 4: 数据加载
# ============================================================================

def load_tool_names(all_tools_path: str) -> Set[str]:
    """加载工具名集合"""
    with open(all_tools_path, 'r', encoding='utf-8') as f:
        tools = json.load(f)
    return {t["name"] for t in tools}


def load_tool_name_to_id(mcp_graph_path: str) -> Dict[str, int]:
    """
    加载工具名到ID的映射
    
    适配 mcp_rl_graph_v2.json 格式:
    - 使用 tool["id"] 而不是枚举索引
    - 同时建立 original_name -> id 的映射（用于匹配轨迹中的工具名）
    """
    with open(mcp_graph_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tools = data.get("tools", [])
    
    name_to_id = {}
    for t in tools:
        tool_id = t.get("id", len(name_to_id))
        name = t.get("name")
        original_name = t.get("original_name", name)
        
        # 主名称映射
        if name:
            name_to_id[name] = tool_id
        
        # 如果有 original_name 且不同于 name，也建立映射
        # 但只映射到第一个变体（避免覆盖）
        if original_name and original_name != name and original_name not in name_to_id:
            name_to_id[original_name] = tool_id
    
    return name_to_id


def build_from_trajectory_dir(
    builder: ToolSimulatorDatabaseBuilder,
    traj_dir: str,
    tool_names: Optional[Set[str]] = None,
):
    """从轨迹目录添加数据"""
    traj_dir = Path(traj_dir)
    
    file_count = 0
    for path in sorted(traj_dir.glob("*.jsonl")):
        file_count += 1
        print(f"  Processing {path.name}...")
        
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                for call in parse_trajectory_record(record):
                    tool_name = call["tool_name"]
                    if tool_names and tool_name not in tool_names:
                        continue
                    builder.add_call(tool_name, call["arguments"], call["output"])
    
    print(f"  Processed {file_count} files")


def build_from_json_file(
    builder: ToolSimulatorDatabaseBuilder,
    json_path: str,
    tool_names: Optional[Set[str]] = None,
):
    """从 JSON 文件添加数据"""
    print(f"  Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get("records") or data.get("episodes") or [data]
    else:
        records = []
    
    print(f"  Found {len(records)} records")
    
    for record in records:
        if not isinstance(record, dict):
            continue
        
        for call in parse_trajectory_record(record):
            tool_name = call["tool_name"]
            if tool_names and tool_name not in tool_names:
                continue
            builder.add_call(tool_name, call["arguments"], call["output"])


# ============================================================================
# Part 5: 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate tool simulator database")
    parser.add_argument(
        "--project-root",
        default="/seu_share2/home/fenglei/230250004/Agent_Tool/tool-use/tool-use",
        help="Project root directory",
    )
    parser.add_argument(
        "--source",
        choices=["trajectories", "all_messages", "both"],
        default="trajectories",
        help="Data source",
    )
    parser.add_argument(
        "--max-calls-per-schema",
        type=int,
        default=30,
        help="Maximum calls per schema",
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=1500,
        help="Maximum output length",
    )
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Disable embedding computation",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=500,
        help="Embedding dimension (vocab size)",
    )
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root)
    
    # 路径
    all_tools_path = project_root / "json_file" / "all_tools.json"
    mcp_graph_path = project_root / "json_file" / "mcp_rl_graph.json"
    traj_dir = project_root / "Toolathlon-Trajectories-merge"
    all_messages_path = project_root / "json_file" / "all_messages.json"
    
    output_dir = project_root / "GRPO-ACO" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Generating Tool Simulator Database")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Data source: {args.source}")
    print(f"Compute embedding: {not args.no_embedding}")
    print("=" * 70)
    
    # 加载工具信息
    tool_names = None
    name_to_id = {}
    
    if all_tools_path.exists():
        tool_names = load_tool_names(str(all_tools_path))
        print(f"\n[Load] {len(tool_names)} tools from all_tools.json")
    
    if mcp_graph_path.exists():
        name_to_id = load_tool_name_to_id(str(mcp_graph_path))
        print(f"[Load] {len(name_to_id)} tool ID mappings from mcp_rl_graph.json")
    
    # 创建构建器
    builder = ToolSimulatorDatabaseBuilder(
        max_calls_per_schema=args.max_calls_per_schema,
        max_output_length=args.max_output_length,
        compute_embedding=not args.no_embedding,
        embedding_dim=args.embedding_dim,
    )
    builder.set_tool_mapping(name_to_id)
    
    # 构建数据
    if args.source in ["trajectories", "both"] and traj_dir.exists():
        print(f"\n[Source] Trajectory files: {traj_dir}")
        build_from_trajectory_dir(builder, str(traj_dir), tool_names)
    
    if args.source in ["all_messages", "both"] and all_messages_path.exists():
        print(f"\n[Source] all_messages.json: {all_messages_path}")
        build_from_json_file(builder, str(all_messages_path), tool_names)
    
    # 构建数据库
    print("\n[Build] Building database...")
    database = builder.build()
    
    # 保存
    output_path = output_dir / "tool_simulator_database.json"
    print(f"\n[Save] Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(database, f, ensure_ascii=False, indent=2)
    
    # 如果有 embedding，保存词表
    if not args.no_embedding and builder.embedder and builder.embedder.fitted:
        vocab_path = output_dir / "embedding_vocab.json"
        builder.embedder.save_vocab(str(vocab_path))
        print(f"[Save] Embedding vocab: {vocab_path}")
    
    # 打印统计
    stats = builder.get_statistics()
    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)
    print(f"  Tools: {stats['num_tools']}")
    print(f"  Total calls: {stats['total_calls']}")
    print(f"  Total schemas: {stats['total_schemas']}")
    print(f"  Avg schemas/tool: {stats['avg_schemas_per_tool']:.2f}")
    
    # 示例
    print("\n" + "=" * 70)
    print("Sample Tools")
    print("=" * 70)
    
    for tool_name in list(builder.data.keys())[:5]:
        schemas = builder.data[tool_name]
        print(f"\n  {tool_name}: {len(schemas)} schemas")
        for schema_hash, schema_data in list(schemas.items())[:2]:
            keys = schema_data["keys"]
            count = schema_data["count"]
            tau = schema_data.get("pheromone", 1.0)
            print(f"    - {keys}: count={count}, τ={tau:.3f}")
    
    print("\n" + "=" * 70)
    print(f"Output: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()