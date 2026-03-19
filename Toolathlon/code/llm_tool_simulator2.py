#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import random
import hashlib
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict


from openai import OpenAI
OPENAI_AVAILABLE = True

# ============================================================================
# LLM模拟器配置
# ============================================================================

@dataclass
class LLMSimulatorConfig:
    """
    LLM模拟器配置
    
    可以直接实例化，也可以从字典创建
    """
    # API配置
    api_key: str = "sk-PovCrGTefqSW0POpIed5jFWF3HN6Cc95PXoWa70Zx1MKLNg4"
    base_url: str = "http://172.22.2.242:3010/v1"
    model: str = "deepseek-v3.1"
    
    # 请求配置
    max_retries: int = 3
    timeout: float = 30.0
    max_tokens: int = 300
    temperature: float = 0.7
    
    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 5000
    
    # 模式配置
    use_llm: bool = True
    use_hybrid: bool = True
    fallback_to_pool: bool = True
    
    # 上下文配置
    max_history_length: int = 5
    max_prompt_length: int = 2000
    
    # 任务完成检测配置
    enable_completion_check: bool = True
    completion_check_temperature: float = 0.3
    completion_cache_size: int = 1000
    
    # 调试
    verbose: bool = False
    log_api_calls: bool = False
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'LLMSimulatorConfig':
        """从字典创建配置"""
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})
    
    @classmethod
    def from_training_config(cls, cfg) -> 'LLMSimulatorConfig':
        """从训练脚本的Config对象创建配置"""
        llm_config = getattr(cfg, 'llm_simulator_config', {})
        use_llm = getattr(cfg, 'use_llm_simulator', False)
        use_hybrid = getattr(cfg, 'use_hybrid_simulator', True)
        enable_completion = getattr(cfg, 'enable_completion_check', True)
        
        return cls.from_dict({
            **llm_config,
            "use_llm": use_llm,
            "use_hybrid": use_hybrid,
            "enable_completion_check": enable_completion,
        })
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "api_key": self.api_key[:20] + "..." if len(self.api_key) > 20 else self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "enable_cache": self.enable_cache,
            "cache_size": self.cache_size,
            "use_llm": self.use_llm,
            "use_hybrid": self.use_hybrid,
            "enable_completion_check": self.enable_completion_check,
            "verbose": self.verbose,
        }


# ============================================================================
# 任务完成检测结果
# ============================================================================

@dataclass
class TaskCompletionResult:
    """任务完成检测结果"""
    is_complete: bool           # 任务是否完成
    confidence: float           # 置信度 (0-1)
    reason: str                 # 完成/未完成的原因
    quality_score: float        # 完成质量评分 (0-1)
    missing_steps: List[str]    # 可能缺失的步骤
    
    def to_dict(self) -> Dict:
        return {
            "is_complete": self.is_complete,
            "confidence": self.confidence,
            "reason": self.reason,
            "quality_score": self.quality_score,
            "missing_steps": self.missing_steps,
        }


# ============================================================================
# 任务完成检测器
# ============================================================================

class TaskCompletionChecker:
    """
    任务完成检测器
    """
    
    SYSTEM_PROMPT = """你是一个任务完成度评估专家。你需要判断给定的工具调用序列是否已经完成了用户的任务。

## 评估标准：
1. **核心需求满足**：用户请求的主要目标是否达成
2. **逻辑完整性**：工具调用序列是否形成完整的工作流
3. **输出质量**：最终输出是否能满足用户需求

## 输出格式（严格JSON，不要添加任何其他内容）：
{
    "is_complete": true或false,
    "confidence": 0.0到1.0之间的数字,
    "quality_score": 0.0到1.0之间的数字,
    "reason": "简短解释为什么完成或未完成",
    "missing_steps": ["如果未完成，列出可能需要的步骤，完成则为空数组"]
}

## 判断指南：
- 如果任务的核心目标已达成，即使还可以优化，也应该判定为完成
- 如果明显还需要关键步骤才能完成任务，判定为未完成
- confidence表示你对判断的确信程度
- quality_score表示完成的质量（即使is_complete为false，也可以有一定分数）"""

    def __init__(self, config: LLMSimulatorConfig = None):
        """初始化检测器"""
        self.config = config or LLMSimulatorConfig()
        
        if OPENAI_AVAILABLE and self.config.use_llm:
            try:
                self.client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
            except Exception as e:
                print(f"[TaskCompletionChecker] Failed to init client: {e}")
                self.client = None
        else:
            self.client = None
        
        # 缓存
        self._cache: Dict[str, TaskCompletionResult] = {}
        self._cache_order: List[str] = []
        self._cache_lock = threading.Lock()
        self._cache_size = self.config.completion_cache_size
        
        # 统计
        self._stats = {
            "total_checks": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "llm_errors": 0,
            "complete_count": 0,
            "incomplete_count": 0,
            "heuristic_fallbacks": 0,
        }
        self._stats_lock = threading.Lock()
    
    def _compute_cache_key(
        self,
        task_name: str,
        user_prompt: str,
        history: List[Dict]
    ) -> str:
        """计算缓存键"""
        history_str = json.dumps(
            [{"name": h.get("name", ""), "output": str(h.get("output", ""))[:80]} 
             for h in history[-5:]],
            sort_keys=True,
            ensure_ascii=False
        )
        content = f"{task_name[:80]}|{user_prompt[:150]}|{history_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _build_prompt(
        self,
        task_name: str,
        user_prompt: str,
        history: List[Dict]
    ) -> str:
        """构建检测提示词"""
        lines = [
            f"【任务名称】{task_name[:200]}",
            f"【用户请求】{user_prompt[:500]}",
            "",
            "【已执行的工具调用序列】"
        ]
        
        for i, h in enumerate(history, 1):
            name = h.get('name', 'unknown')[:40]
            output = str(h.get('output', ''))[:200]
            lines.append(f"{i}. 工具: {name}")
            lines.append(f"   输出: {output}")
            lines.append("")
        
        lines.append("请评估上述工具调用序列是否已完成用户任务，直接输出JSON：")
        
        return "\n".join(lines)
    
    def _parse_response(self, response: str) -> TaskCompletionResult:
        """解析LLM响应"""
        try:
            # 清理响应
            response = response.strip()
            
            # 移除markdown代码块
            if response.startswith("```"):
                lines = response.split("\n")
                # 找到JSON开始和结束
                start_idx = 0
                end_idx = len(lines)
                for i, line in enumerate(lines):
                    if line.strip().startswith("{"):
                        start_idx = i
                        break
                    elif "json" in line.lower() or line.strip() == "```":
                        start_idx = i + 1
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip() == "```":
                        end_idx = i
                        break
                    elif lines[i].strip().endswith("}"):
                        end_idx = i + 1
                        break
                response = "\n".join(lines[start_idx:end_idx])
            
            # 尝试找到JSON部分
            if "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                response = response[start:end]
            
            data = json.loads(response)
            
            return TaskCompletionResult(
                is_complete=bool(data.get("is_complete", False)),
                confidence=min(1.0, max(0.0, float(data.get("confidence", 0.5)))),
                quality_score=min(1.0, max(0.0, float(data.get("quality_score", 0.5)))),
                reason=str(data.get("reason", ""))[:200],
                missing_steps=list(data.get("missing_steps", []))[:5]
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            # 解析失败，返回保守结果
            return TaskCompletionResult(
                is_complete=False,
                confidence=0.3,
                quality_score=0.0,
                reason=f"Parse error: {str(e)[:50]}",
                missing_steps=[]
            )
    
    def _heuristic_check(
        self,
        task_name: str,
        user_prompt: str,
        history: List[Dict]
    ) -> TaskCompletionResult:
        """
        启发式任务完成检测
        
        判断逻辑：
        1. 检查最后一个工具的输出是否包含完成/错误信号
        2. 检查最后一个工具是否是"最终类型"工具
        3. 检查历史长度和任务复杂度
        """
        if not history:
            return TaskCompletionResult(
                is_complete=False,
                confidence=1.0,
                quality_score=0.0,
                reason="No tools executed yet",
                missing_steps=["Execute first tool"]
            )
        
        last_output = str(history[-1].get("output", "")).lower()
        last_tool = str(history[-1].get("name", "")).lower()
        num_steps = len(history)
        
        # 完成信号关键词（强信号）
        strong_completion_signals = [
            "successfully", "completed", "done", "finished", "saved successfully",
            "created successfully", "exported", "sent successfully", "已完成", "成功保存"
        ]
        
        # 完成信号关键词（弱信号）
        weak_completion_signals = [
            "success", "saved", "created", "generated", "result:", "summary:", 
            "answer:", "output:", "complete", "成功", "结果"
        ]
        
        # 未完成信号关键词
        incomplete_signals = [
            "error", "failed", "not found", "need more", "missing",
            "continue", "next step", "then", "incomplete", "partial",
            "unable to", "cannot", "couldn't", "失败", "错误", "未找到"
        ]
        
        # 最终步骤类工具（通常是任务完成标志）
        final_tools = [
            "save", "export", "send", "submit", "generate", "create",
            "summarize", "finish", "complete", "output", "write", "post",
            "upload", "publish", "deliver", "store"
        ]
        
        # 中间步骤类工具（通常不是最终步骤）
        intermediate_tools = [
            "search", "find", "get", "fetch", "retrieve", "read", "load",
            "parse", "extract", "filter", "process", "analyze", "query"
        ]
        
        # 检查信号
        has_strong_completion = any(s in last_output for s in strong_completion_signals)
        has_weak_completion = any(s in last_output for s in weak_completion_signals)
        has_incomplete_signal = any(s in last_output for s in incomplete_signals)
        is_final_tool = any(t in last_tool for t in final_tools)
        is_intermediate_tool = any(t in last_tool for t in intermediate_tools)
        
        # ========== 判断逻辑 ==========
        
        # 规则1：有明确的未完成信号 → 未完成（高置信度）
        if has_incomplete_signal and not has_strong_completion:
            return TaskCompletionResult(
                is_complete=False,
                confidence=0.85,
                quality_score=0.2,
                reason=f"Output contains incomplete signal",
                missing_steps=["Fix error or continue execution"]
            )
        
        # 规则2：强完成信号 + 没有未完成信号 → 完成（高置信度）
        if has_strong_completion and not has_incomplete_signal:
            return TaskCompletionResult(
                is_complete=True,
                confidence=0.85,
                quality_score=0.8,
                reason="Output indicates successful completion",
                missing_steps=[]
            )
        
        # 规则3：最终类型工具 + 弱完成信号 + 没有未完成信号 → 完成（中等置信度）
        if is_final_tool and has_weak_completion and not has_incomplete_signal:
            return TaskCompletionResult(
                is_complete=True,
                confidence=0.7,
                quality_score=0.7,
                reason=f"Final-type tool ({last_tool}) executed with success signal",
                missing_steps=[]
            )
        
        # 规则4：最终类型工具 + 没有未完成信号 → 可能完成（需要LLM确认）
        if is_final_tool and not has_incomplete_signal:
            return TaskCompletionResult(
                is_complete=True,
                confidence=0.5,  # 置信度不够，会触发LLM确认
                quality_score=0.6,
                reason=f"Final-type tool ({last_tool}) executed",
                missing_steps=[]
            )
        
        # 规则5：只有1步且是中间工具 → 未完成（高置信度）
        if num_steps == 1 and is_intermediate_tool:
            return TaskCompletionResult(
                is_complete=False,
                confidence=0.8,
                quality_score=0.2,
                reason="Only one intermediate step executed",
                missing_steps=["Continue with more steps"]
            )
        
        # 规则6：多步执行但最后是中间工具 → 可能未完成
        if num_steps >= 2 and is_intermediate_tool and not has_weak_completion:
            return TaskCompletionResult(
                is_complete=False,
                confidence=0.6,
                quality_score=0.4,
                reason="Last tool is intermediate type",
                missing_steps=["May need final step"]
            )
        
        # 规则7：默认情况 → 不确定（低置信度，会触发LLM）
        return TaskCompletionResult(
            is_complete=False,
            confidence=0.4,  # 低置信度，会触发LLM判断
            quality_score=0.3,
            reason="Status unclear (heuristic)",
            missing_steps=[]
        )
    
    def check_completion(
        self,
        task_name: str,
        user_prompt: str,
        history: List[Dict],
        use_cache: bool = True,
        smart_mode: bool = True  # 先启发式，再LLM
    ) -> TaskCompletionResult:
        """
        检查任务是否完成
        """
        with self._stats_lock:
            self._stats["total_checks"] += 1
        
        # 空历史直接返回
        if not history:
            return TaskCompletionResult(
                is_complete=False,
                confidence=1.0,
                quality_score=0.0,
                reason="No tools executed",
                missing_steps=["Start execution"]
            )
        
        # 检查缓存
        cache_key = None
        if use_cache:
            cache_key = self._compute_cache_key(task_name, user_prompt, history)
            with self._cache_lock:
                if cache_key in self._cache:
                    with self._stats_lock:
                        self._stats["cache_hits"] += 1
                    return self._cache[cache_key]
        
        # 先用启发式判断
        if smart_mode:
            heuristic_result = self._heuristic_check(task_name, user_prompt, history)
            
            # 情况1：启发式高置信度判断为"未完成" → 直接返回，不调用LLM
            if not heuristic_result.is_complete and heuristic_result.confidence >= 0.7:
                with self._stats_lock:
                    self._stats["heuristic_fallbacks"] += 1
                    self._stats["incomplete_count"] += 1
                
                # 存入缓存
                if use_cache and cache_key:
                    with self._cache_lock:
                        self._cache[cache_key] = heuristic_result
                        self._cache_order.append(cache_key)
                        while len(self._cache_order) > self._cache_size:
                            old_key = self._cache_order.pop(0)
                            self._cache.pop(old_key, None)
                
                return heuristic_result
            
            # 情况2：启发式判断为"完成"但置信度不够高 → 需要LLM确认
            # 情况3：启发式不确定 → 需要LLM判断
            # 继续执行下面的LLM调用
        
        # LLM不可用时使用启发式
        if self.client is None:
            with self._stats_lock:
                self._stats["heuristic_fallbacks"] += 1
            result = self._heuristic_check(task_name, user_prompt, history)
        else:
            # 调用LLM
            try:
                with self._stats_lock:
                    self._stats["llm_calls"] += 1
                
                prompt = self._build_prompt(task_name, user_prompt, history)
                
                completion = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=self.config.completion_check_temperature,
                )
                
                response = completion.choices[0].message.content
                result = self._parse_response(response)
                
            except Exception as e:
                with self._stats_lock:
                    self._stats["llm_errors"] += 1
                
                if self.config.verbose:
                    print(f"[TaskCompletionChecker] LLM error: {e}")
                
                result = self._heuristic_check(task_name, user_prompt, history)
        
        # 更新统计
        with self._stats_lock:
            if result.is_complete:
                self._stats["complete_count"] += 1
            else:
                self._stats["incomplete_count"] += 1
        
        # 存入缓存
        if use_cache and cache_key:
            with self._cache_lock:
                self._cache[cache_key] = result
                self._cache_order.append(cache_key)
                
                # 缓存淘汰
                while len(self._cache_order) > self._cache_size:
                    old_key = self._cache_order.pop(0)
                    self._cache.pop(old_key, None)
        
        return result
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        with self._stats_lock:
            stats = self._stats.copy()
        
        stats["cache_size"] = len(self._cache)
        total = stats["complete_count"] + stats["incomplete_count"]
        stats["completion_rate"] = stats["complete_count"] / max(1, total)
        stats["cache_hit_rate"] = stats["cache_hits"] / max(1, stats["total_checks"])
        
        return stats


# ============================================================================
# LLM工具模拟器
# ============================================================================

class LLMToolSimulator:
    """
    基于LLM的工具执行模拟器（带任务完成检测）
    
    核心特性：
        - 上下文感知的工具输出生成
        - 任务完成检测（支持动态停止）
        - 多级缓存机制
        - 自动重试与指数退避
        - 线程安全
    """
    
    SYSTEM_PROMPT = """你是一个专业的工具执行模拟器。你的任务是根据给定的工具调用信息，生成真实、合理的模拟输出。

## 输出规则：
1. 直接输出工具的返回结果，不要添加任何解释、前缀或后缀
2. 输出应该看起来像真实的API/工具返回值
3. 根据工具类型、参数和上下文生成合理的模拟数据
4. 输出长度控制在50-200字符之间
5. 保持输出的多样性和真实性

## 不同工具类型的输出指南：
- 搜索类工具：返回搜索结果摘要或列表
- 数据获取类工具：返回JSON格式或结构化数据
- 文件操作类工具：返回操作状态和结果
- API调用类工具：返回API响应格式

## 注意：
- 参考历史调用的输出风格保持一致性
- 根据参数值生成相关的输出内容"""

    ERROR_PATTERNS = [
        "error", "failed", "exception", "invalid",
        "not found", "permission denied", "timeout",
        "connection refused", "unauthorized", "forbidden",
        "bad request", "rate limit", "quota exceeded"
    ]
    
    PARAM_ALIASES = {
        "query": ["q", "search", "keyword", "text", "input", "question"],
        "path": ["file", "filepath", "filename", "dir", "directory"],
        "url": ["link", "uri", "address", "endpoint"],
        "content": ["text", "body", "data", "message", "payload"],
        "name": ["title", "label", "id", "identifier"],
        "limit": ["max", "count", "num", "size", "top", "n"],
    }
    
    def __init__(
        self,
        tool_system,
        config: LLMSimulatorConfig = None,
        **kwargs
    ):
        """初始化LLM工具模拟器"""
        self.tool_system = tool_system
        
        # 处理配置
        if config is not None:
            self.config = config
        else:
            self.config = LLMSimulatorConfig()
        
        # 应用kwargs覆盖
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # 初始化OpenAI客户端
        self.client = None
        if OPENAI_AVAILABLE and self.config.use_llm:
            try:
                self.client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
            except Exception as e:
                print(f"[LLMToolSimulator] Failed to init client: {e}")
        
        # 任务完成检测器
        self.completion_checker: Optional[TaskCompletionChecker] = None
        if self.config.enable_completion_check and self.config.use_llm:
            self.completion_checker = TaskCompletionChecker(self.config)
        
        # 输出缓存
        self._cache: Dict[str, str] = {}
        self._cache_order: List[str] = []
        self._cache_lock = threading.Lock()
        
        # 统计信息
        self._stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "api_successes": 0,
            "api_errors": 0,
            "total_tokens": 0,
        }
        self._stats_lock = threading.Lock()
        
        # 参数别名映射
        self._alias_to_canonical = {}
        for canonical, aliases in self.PARAM_ALIASES.items():
            self._alias_to_canonical[canonical] = canonical
            for alias in aliases:
                self._alias_to_canonical[alias.lower()] = canonical
        
        if self.config.verbose:
            print(f"[LLMToolSimulator] Initialized:")
            print(f"    Model: {self.config.model}")
            print(f"    Completion check: {self.completion_checker is not None}")
    
    def _compute_cache_key(
        self,
        variant_name: str,
        task_name: str,
        user_prompt: str,
        history: List[Dict],
        params: Dict
    ) -> str:
        """计算缓存键"""
        recent_history = history[-self.config.max_history_length:] if history else []
        history_str = json.dumps(
            [{"name": h.get("name", ""), "output": str(h.get("output", ""))[:80]} 
             for h in recent_history],
            sort_keys=True, ensure_ascii=False
        )
        params_str = json.dumps(
            {k: str(v)[:40] for k, v in (params or {}).items()},
            sort_keys=True, ensure_ascii=False
        )
        content = f"{variant_name}|{task_name[:80]}|{user_prompt[:150]}|{history_str}|{params_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_tool_info(self, variant_name: str) -> Tuple[str, str, Dict]:
        """获取工具信息"""
        original_name = self.tool_system.original_name_from_extended(variant_name)
        tool_info = self.tool_system.original_tools.get(original_name, {})
        description = tool_info.get("description", "Execute tool operation")
        parameters = tool_info.get("parameters", {})
        return original_name, description, parameters
    
    def _build_prompt(
        self,
        variant_name: str,
        params: Dict,
        task_name: str,
        user_prompt: str,
        history: List[Dict]
    ) -> str:
        """构建用户提示词"""
        original_name, tool_desc, _ = self._get_tool_info(variant_name)
        variant_info = self.tool_system.extended_tools.get(variant_name, {})
        actual_keys = variant_info.get("actual_keys", [])
        
        lines = [
            f"【任务名称】{task_name[:150]}",
            f"【用户请求】{user_prompt[:400]}",
        ]
        
        if history:
            lines.append("\n【历史工具调用】")
            for h in history[-self.config.max_history_length:]:
                name = h.get('name', 'unknown')[:40]
                output = str(h.get('output', ''))[:120]
                lines.append(f"  [{name}] → {output}")
        
        lines.extend([
            f"\n【当前调用工具】{variant_name}",
            f"【工具描述】{tool_desc[:250]}",
        ])
        
        if params:
            lines.append("【调用参数】")
            for key, value in list(params.items())[:10]:
                lines.append(f"  • {key}: {str(value)[:100]}")
        elif actual_keys:
            lines.append(f"【期望参数】{', '.join(actual_keys[:8])}")
        
        lines.append("\n请模拟该工具的执行输出（直接输出结果）：")
        
        prompt = "\n".join(lines)
        if len(prompt) > self.config.max_prompt_length:
            prompt = prompt[:self.config.max_prompt_length] + "\n..."
        
        return prompt
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """调用LLM API"""
        if self.client is None:
            return None
        
        for attempt in range(self.config.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                
                response = completion.choices[0].message.content.strip()
                
                with self._stats_lock:
                    self._stats["api_successes"] += 1
                    if hasattr(completion, 'usage') and completion.usage:
                        self._stats["total_tokens"] += completion.usage.total_tokens
                
                return response
                
            except Exception as e:
                with self._stats_lock:
                    self._stats["api_errors"] += 1
                
                if self.config.verbose:
                    print(f"[LLMToolSimulator] API error (attempt {attempt+1}): {e}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(min(30, 2 ** attempt + random.random()))
        
        return None
    
    def _post_process_response(self, response: str, variant_name: str) -> str:
        """后处理LLM响应"""
        prefixes = ["工具输出：", "输出：", "结果：", "Output:", "Result:"]
        for prefix in prefixes:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
        
        if response.startswith("```"):
            lines = response.split("\n")
            if len(lines) > 2:
                response = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        
        if len(response) > 500:
            response = response[:500] + "..."
        
        if not response.strip():
            response = f"[{variant_name}] executed successfully"
        
        return response.strip()
    
    def _generate_default_output(self, variant_name: str, params: Dict) -> str:
        """生成默认输出"""
        original_name = self.tool_system.original_name_from_extended(variant_name)
        name_lower = original_name.lower()
        
        if any(kw in name_lower for kw in ["search", "find", "query"]):
            return f"Found 3 relevant results for the query."
        elif any(kw in name_lower for kw in ["get", "fetch", "retrieve", "read"]):
            return f"Retrieved data: {{'status': 'ok', 'count': {random.randint(1, 10)}}}"
        elif any(kw in name_lower for kw in ["create", "add", "insert", "write"]):
            return f"Created successfully. ID: {random.randint(1000, 9999)}"
        elif any(kw in name_lower for kw in ["update", "modify", "edit"]):
            return f"Updated successfully."
        elif any(kw in name_lower for kw in ["delete", "remove"]):
            return f"Deleted successfully."
        elif any(kw in name_lower for kw in ["send", "post", "submit"]):
            return f"Sent successfully. Status: 200"
        else:
            param_str = ", ".join(list((params or {}).keys())[:3]) or "default"
            return f"[{variant_name}] executed with params: {param_str}"
    
    def get_output(
        self,
        variant_name: str,
        params: Dict = None,
        task_name: str = "",
        user_prompt: str = "",
        history: List[Dict] = None
    ) -> str:
        """
        获取工具模拟输出

        """
        with self._stats_lock:
            self._stats["total_calls"] += 1
        
        params = params or {}
        history = history or []
        
        # 检查缓存
        if self.config.enable_cache:
            cache_key = self._compute_cache_key(
                variant_name, task_name, user_prompt, history, params
            )
            with self._cache_lock:
                if cache_key in self._cache:
                    with self._stats_lock:
                        self._stats["cache_hits"] += 1
                    return self._cache[cache_key]
        
        # 调用LLM
        with self._stats_lock:
            self._stats["api_calls"] += 1
        
        prompt = self._build_prompt(variant_name, params, task_name, user_prompt, history)
        response = self._call_llm(prompt)
        
        if response:
            response = self._post_process_response(response, variant_name)
            
            # 存入缓存
            if self.config.enable_cache:
                with self._cache_lock:
                    self._cache[cache_key] = response
                    self._cache_order.append(cache_key)
                    while len(self._cache_order) > self.config.cache_size:
                        old_key = self._cache_order.pop(0)
                        self._cache.pop(old_key, None)
            
            return response
        
        return self._generate_default_output(variant_name, params)
    
    def get_output_simple(self, variant_name: str, params: Dict = None) -> str:
        """简化版输出获取（向后兼容）"""
        return self.get_output(variant_name, params, "", "", [])
    
    def check_task_completion(
        self,
        task_name: str,
        user_prompt: str,
        history: List[Dict],
        smart_mode: bool = True
    ) -> TaskCompletionResult:
        """
        检查任务是否完成
        
        Args:
            task_name: 任务名称
            user_prompt: 用户请求
            history: 工具调用历史
            
        Returns:
            TaskCompletionResult对象
        """
        if self.completion_checker is None:
            return TaskCompletionResult(
                is_complete=False,
                confidence=0.0,
                quality_score=0.0,
                reason="Completion checker not available",
                missing_steps=[]
            )
        
        return self.completion_checker.check_completion(
            task_name, user_prompt, history,
            smart_mode=smart_mode  # 【传递参数】
        )
    
    def is_execution_error(self, output: str) -> bool:
        """检测执行错误"""
        output_lower = output.lower()
        return any(p in output_lower for p in self.ERROR_PATTERNS)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        with self._stats_lock:
            stats = self._stats.copy()
        
        stats["cache_size"] = len(self._cache)
        stats["cache_hit_rate"] = stats["cache_hits"] / max(1, stats["total_calls"])
        stats["api_success_rate"] = stats["api_successes"] / max(1, stats["api_calls"])
        
        if self.completion_checker:
            stats["completion_checker"] = self.completion_checker.get_statistics()
        
        return stats
    
    def clear_cache(self):
        """清空缓存"""
        with self._cache_lock:
            self._cache.clear()
            self._cache_order.clear()


# ============================================================================
# 混合模拟器
# ============================================================================

class HybridToolSimulator:
    """
    混合工具模拟器（LLM + 预定义池）
    """
    
    def __init__(
        self,
        tool_system,
        database_path: str = None,
        config: LLMSimulatorConfig = None,
        **kwargs
    ):
        """初始化混合模拟器"""
        self.tool_system = tool_system
        
        if config is not None:
            self.config = config
        else:
            self.config = LLMSimulatorConfig.from_dict(kwargs)
        
        # LLM模拟器
        self.llm_simulator: Optional[LLMToolSimulator] = None
        if self.config.use_llm and OPENAI_AVAILABLE:
            self.llm_simulator = LLMToolSimulator(tool_system, config=self.config)
        
        # 预定义输出池
        self.pool_outputs: Dict[str, List[str]] = {}
        if database_path:
            self._load_pool(database_path)
        
        # 统计
        self._stats = {
            "total_calls": 0,
            "llm_successes": 0,
            "pool_fallbacks": 0,
            "default_outputs": 0,
        }
        
        # 错误检测
        self.error_patterns = LLMToolSimulator.ERROR_PATTERNS
        
        print(f"[HybridToolSimulator] Initialized:")
        print(f"    LLM enabled: {self.llm_simulator is not None}")
        print(f"    Pool tools: {len(self.pool_outputs)}")
    
    def _load_pool(self, database_path: str):
        """加载预定义输出池"""
        if not Path(database_path).exists():
            return
        
        try:
            with open(database_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for tool_name, tool_data in data.items():
                if isinstance(tool_data, dict):
                    outputs = tool_data.get("outputs", [])
                    if outputs:
                        self.pool_outputs[tool_name] = outputs if isinstance(outputs, list) else [outputs]
                elif isinstance(tool_data, list):
                    self.pool_outputs[tool_name] = tool_data
        except Exception as e:
            print(f"[HybridToolSimulator] Failed to load pool: {e}")
    
    def _get_pool_output(self, variant_name: str) -> Optional[str]:
        """从池获取输出"""
        if variant_name in self.pool_outputs:
            return random.choice(self.pool_outputs[variant_name])
        
        original_name = self.tool_system.original_name_from_extended(variant_name)
        if original_name in self.pool_outputs:
            return random.choice(self.pool_outputs[original_name])
        
        return None
    
    def get_output(
        self,
        variant_name: str,
        params: Dict = None,
        task_name: str = "",
        user_prompt: str = "",
        history: List[Dict] = None
    ) -> str:
        """获取工具模拟输出"""
        self._stats["total_calls"] += 1
        
        # 尝试LLM
        if self.llm_simulator is not None:
            output = self.llm_simulator.get_output(
                variant_name, params, task_name, user_prompt, history
            )
            
            # 检查是否有效输出
            if output and not output.startswith(f"[{variant_name}] executed"):
                self._stats["llm_successes"] += 1
                return output
        
        # 回退到池
        if self.config.fallback_to_pool:
            pool_output = self._get_pool_output(variant_name)
            if pool_output:
                self._stats["pool_fallbacks"] += 1
                return pool_output
        
        self._stats["default_outputs"] += 1
        return f"[{variant_name}] executed successfully"
    
    def check_task_completion(
        self,
        task_name: str,
        user_prompt: str,
        history: List[Dict],
        smart_mode: bool = True
    ) -> TaskCompletionResult:
        """检查任务是否完成"""
        if self.llm_simulator is not None:
            return self.llm_simulator.check_task_completion(
                task_name, user_prompt, history,
                smart_mode=smart_mode  # 【传递参数】
            )
        
        return TaskCompletionResult(
            is_complete=False,
            confidence=0.0,
            quality_score=0.0,
            reason="LLM simulator not available",
            missing_steps=[]
        )
    
    def is_execution_error(self, output: str) -> bool:
        """检测执行错误"""
        return any(p in output.lower() for p in self.error_patterns)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = self._stats.copy()
        stats["llm_success_rate"] = stats["llm_successes"] / max(1, stats["total_calls"])
        
        if self.llm_simulator:
            stats["llm_stats"] = self.llm_simulator.get_statistics()
        
        return stats


# ============================================================================
# 池模拟器
# ============================================================================

class PoolToolSimulator:
    """基于预定义池的工具模拟器（原始ToolSimulator）"""
    
    def __init__(self, database_path: str, tool_system):
        self.tool_system = tool_system
        self.outputs: Dict[str, List[str]] = {}
        self.error_patterns = ["error", "failed", "exception", "invalid", "not found"]
        
        if Path(database_path).exists():
            try:
                with open(database_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for tool_name, tool_data in data.items():
                    if isinstance(tool_data, dict):
                        outputs = tool_data.get("outputs", [])
                        if outputs:
                            self.outputs[tool_name] = outputs if isinstance(outputs, list) else [outputs]
                    elif isinstance(tool_data, list):
                        self.outputs[tool_name] = tool_data
            except Exception as e:
                print(f"[PoolToolSimulator] Error loading: {e}")
        
        print(f"[PoolToolSimulator] Loaded {len(self.outputs)} tools")
    
    def get_output(self, variant_name: str, params: Dict = None, **kwargs) -> str:
        """获取模拟输出"""
        if variant_name in self.outputs:
            return random.choice(self.outputs[variant_name])
        
        original_name = self.tool_system.original_name_from_extended(variant_name)
        if original_name in self.outputs:
            return random.choice(self.outputs[original_name])
        
        return f"[{variant_name}] executed successfully"
    
    def is_execution_error(self, output: str) -> bool:
        return any(p in output.lower() for p in self.error_patterns)
    
    def get_statistics(self) -> Dict:
        return {"type": "pool", "num_tools": len(self.outputs)}



def create_simulator(
    tool_system,
    config: LLMSimulatorConfig = None,
    database_path: str = None,
    use_llm: bool = True,
    use_hybrid: bool = True,
    **kwargs
) -> Any:
    """创建工具模拟器"""
    if config is None:
        config = LLMSimulatorConfig.from_dict({
            "use_llm": use_llm,
            "use_hybrid": use_hybrid,
            **kwargs
        })
    
    if config.use_llm and not OPENAI_AVAILABLE:
        print("[create_simulator] OpenAI not available, using pool simulator")
        config.use_llm = False
    
    if not config.use_llm:
        if database_path and Path(database_path).exists():
            return PoolToolSimulator(database_path, tool_system)
        return LLMToolSimulator(tool_system, config=config)
    
    if config.use_hybrid and database_path:
        return HybridToolSimulator(tool_system, database_path=database_path, config=config)
    
    return LLMToolSimulator(tool_system, config=config)


def create_simulator_from_config(tool_system, cfg) -> Any:
    """从训练Config创建模拟器"""
    use_llm = getattr(cfg, 'use_llm_simulator', False)
    use_hybrid = getattr(cfg, 'use_hybrid_simulator', True)
    llm_config_dict = getattr(cfg, 'llm_simulator_config', {})
    database_path = getattr(cfg, 'simulator_database_path', None)
    enable_completion = getattr(cfg, 'enable_completion_check', True)
    
    if database_path:
        database_path = str(database_path)
    
    if use_llm:
        config = LLMSimulatorConfig.from_dict({
            **llm_config_dict,
            "use_llm": True,
            "use_hybrid": use_hybrid,
            "fallback_to_pool": use_hybrid,
            "enable_completion_check": enable_completion,
        })
        return create_simulator(tool_system, config=config, database_path=database_path)
    else:
        if database_path and Path(database_path).exists():
            return PoolToolSimulator(database_path, tool_system)
        return PoolToolSimulator(database_path or "", tool_system)


ToolSimulator = PoolToolSimulator


# ============================================================================
# 测试函数
# ============================================================================

def test_llm_connection(config: LLMSimulatorConfig = None) -> bool:
    """测试LLM API连接"""
    print("\n" + "=" * 50)
    print("Testing LLM API Connection")
    print("=" * 50)
    
    if not OPENAI_AVAILABLE:
        print("✗ OpenAI package not installed")
        return False
    
    config = config or LLMSimulatorConfig()
    print(f"  API Base: {config.base_url}")
    print(f"  Model: {config.model}")
    
    try:
        client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=15.0
        )
        
        completion = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": "Say 'OK'"}],
            max_tokens=10
        )
        
        response = completion.choices[0].message.content
        print(f"\n✓ Connection successful! Response: {response}")
        return True
        
    except Exception as e:
        print(f"\n✗ Connection failed: {e}")
        return False


if __name__ == "__main__":
    test_llm_connection()