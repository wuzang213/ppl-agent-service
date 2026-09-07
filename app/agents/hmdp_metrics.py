"""Agent 请求指标采集与三次重试。"""

import asyncio
import json
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

import aiosqlite
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

_metrics: ContextVar[dict[str, Any]] = ContextVar("hmdp_metrics", default=None)


def begin_metrics(thread_id: str, user_id: str, mode: str):
    data = {
        "thread_id": thread_id,
        "user_id": user_id,
        "mode": mode,
        "success": True,
        "model_call_count": 0,
        "model_failure_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "tools_called": [],
        "error": None,
        "started_at": time.time(),
    }
    return _metrics.set(data)


def finish_metrics(token, success: bool, error: str | None = None) -> dict[str, Any]:
    data = _metrics.get()
    if data is None:
        data = {}
    data["success"] = success
    data["error"] = error
    data["duration_ms"] = int((time.time() - data.get("started_at", time.time())) * 1000)
    data.pop("started_at", None)
    _metrics.reset(token)
    return data


def _ensure_metrics() -> dict[str, Any]:
    data = _metrics.get()
    if data is None:
        raise RuntimeError("metrics context not initialized")
    return data


def record_tool_call(tool_name: str) -> None:
    data = _metrics.get()
    if data is not None and tool_name and tool_name not in data["tools_called"]:
        data["tools_called"].append(tool_name)


def record_node(node_name: str) -> None:
    """记录本轮经过的节点 / worker，复用 tools_called 字段（节点编排后不再有工具循环）。"""
    record_tool_call(node_name)


def record_model_call(node_name: str = "") -> None:
    """记录一次模型调用。节点可能运行在子任务里，取不到上下文时静默跳过。"""
    data = _metrics.get()
    if data is None:
        return
    data["model_call_count"] += 1
    if node_name and node_name not in data["tools_called"]:
        data["tools_called"].append(node_name)


def record_usage(prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
    data = _metrics.get()
    if data is None:
        return
    data["prompt_tokens"] += int(prompt_tokens or 0)
    data["completion_tokens"] += int(completion_tokens or 0)
    data["total_tokens"] += int(prompt_tokens or 0) + int(completion_tokens or 0)


def record_usage_from_message(message: Any) -> None:
    """从模型返回的消息里抽取 token 用量，抽不到就忽略。"""
    usage = getattr(message, "usage_metadata", None) or {}
    if not usage and hasattr(message, "response_metadata"):
        meta = getattr(message, "response_metadata", {}) or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
    if not usage:
        return
    record_usage(
        usage.get("input_tokens", usage.get("prompt_tokens", 0)),
        usage.get("output_tokens", usage.get("completion_tokens", 0)),
    )


def _collect_usage(response: ModelResponse) -> None:
    data = _ensure_metrics()
    for message in response.result or []:
        if not hasattr(message, "usage_metadata") and not hasattr(message, "response_metadata"):
            continue
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            meta = getattr(message, "response_metadata", {}) or {}
            usage = meta.get("token_usage") or meta.get("usage")
        if not usage:
            continue
        data["prompt_tokens"] += usage.get("input_tokens", usage.get("prompt_tokens", 0))
        data["completion_tokens"] += usage.get("output_tokens", usage.get("completion_tokens", 0))
        data["total_tokens"] += usage.get("total_tokens", 0)


class ModelRetryMetricsMiddleware(AgentMiddleware):
    """最多尝试 3 次模型调用，记录调用次数与 token，3 次失败抛异常。"""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        data = _ensure_metrics()
        for attempt in range(1, 4):
            data["model_call_count"] += 1
            try:
                response = handler(request)
                _collect_usage(response)
                return response
            except Exception as exc:
                data["model_failure_count"] += 1
                data["error"] = str(exc)
                if attempt == 3:
                    raise
                time.sleep(min(1.5 ** attempt, 5))
        return None

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> ModelResponse:
        data = _ensure_metrics()
        for attempt in range(1, 4):
            data["model_call_count"] += 1
            try:
                response = await handler(request)
                _collect_usage(response)
                return response
            except Exception as exc:
                data["model_failure_count"] += 1
                data["error"] = str(exc)
                if attempt == 3:
                    raise
                await asyncio.sleep(min(1.5 ** attempt, 5))
        return None


async def save_metrics(data: dict[str, Any]) -> None:
    db_path = Path("db") / "hmdp_metrics.db"
    db_path.parent.mkdir(exist_ok=True)
    conn = await aiosqlite.connect(str(db_path))
    try:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                success INTEGER NOT NULL,
                model_call_count INTEGER NOT NULL DEFAULT 0,
                model_failure_count INTEGER NOT NULL DEFAULT 0,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                tools_called TEXT,
                error TEXT,
                duration_ms INTEGER,
                created_at TEXT NOT NULL
            )
            """
        )
        await conn.execute(
            """
            INSERT INTO agent_metrics (
                thread_id, user_id, mode, success, model_call_count,
                model_failure_count, prompt_tokens, completion_tokens,
                total_tokens, tools_called, error, duration_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("thread_id", ""),
                data.get("user_id", ""),
                data.get("mode", ""),
                1 if data.get("success") else 0,
                data.get("model_call_count", 0),
                data.get("model_failure_count", 0),
                data.get("prompt_tokens", 0),
                data.get("completion_tokens", 0),
                data.get("total_tokens", 0),
                json.dumps(data.get("tools_called", []), ensure_ascii=False),
                data.get("error"),
                data.get("duration_ms"),
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        await conn.commit()
    finally:
        await conn.close()