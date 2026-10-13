"""链路追踪 ID：X-Trace-Id 一次请求一个，贯穿日志与账本，有请求头则用之否则生成；用纯 ASGI 中间件（BaseHTTPMiddleware 会缓冲 SSE）；取值调用时读 env。"""

from __future__ import annotations

import os
import uuid
from contextvars import ContextVar

from app.common.logger import logger

#: 请求头默认名（可用 `AGENT_TRACE_HEADER` 覆盖，与 Gateway 侧约定保持一致）。
DEFAULT_TRACE_HEADER = "X-Trace-Id"

_trace_id: ContextVar[str | None] = ContextVar("agent_trace_id", default=None)


def trace_header_name() -> str:
    return (os.getenv("AGENT_TRACE_HEADER") or "").strip() or DEFAULT_TRACE_HEADER


def trace_enabled() -> bool:
    """总开关 `AGENT_TRACE_ENABLED`（默认开）。关掉后所有取值返回空串 —— 日志里不再带 trace id。"""
    raw = os.getenv("AGENT_TRACE_ENABLED")
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() in ("1", "true", "yes", "on")


def get_trace_id() -> str:
    """当前上下文的 trace id（未启用 / 未设置 ⇒ 空串，**不**自动生成）。"""
    if not trace_enabled():
        return ""
    return _trace_id.get() or ""


def ensure_trace_id() -> str:
    """取当前 trace id；拿不到就现生成一个并写进上下文（与 get_trace_id 区别：SSE 生成器体可能在中间件返回后才执行，此时 get_trace_id 为空，此处补生成以保证本轮一定有 ID）。"""
    if not trace_enabled():
        return ""
    current = _trace_id.get()
    if current:
        return current
    _trace_id.set(uuid.uuid4().hex)
    return _trace_id.get() or ""


def set_trace_id(value: str):
    """写入上下文，返回 reset token（调用方在 finally 里 `reset_trace_id(token)`）。"""
    return _trace_id.set(value)


def reset_trace_id(token) -> None:
    try:
        _trace_id.reset(token)
    except Exception:  # noqa: BLE001 - token 用错不该把请求搞挂
        pass


def new_trace_id() -> str:
    return uuid.uuid4().hex


class TraceMiddleware:
    """纯 ASGI 中间件：读请求头 → 写 ContextVar → 回写响应头 → 请求结束清理；刻意不用 BaseHTTPMiddleware（会缓冲 SSE）。"""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        if not trace_enabled():
            await self.app(scope, receive, send)
            return

        incoming = ""
        for key, value in scope.get("headers") or []:
            if key.decode("latin-1").lower() == trace_header_name().lower():
                incoming = value.decode("latin-1").strip()
                break
        trace_id = incoming or new_trace_id()
        token = set_trace_id(trace_id)

        async def send_with_trace(message):
            # 响应头回写：前端/网关拿到它就能对上服务端的日志
            if message.get("type") == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append(
                    (trace_header_name().lower().encode("latin-1"),
                     trace_id.encode("latin-1"))
                )
            await send(message)

        try:
            await self.app(scope, receive, send_with_trace)
        except Exception as exc:
            logger.error("trace_id=%s 请求处理异常: %s", trace_id, exc)
            raise
        finally:
            reset_trace_id(token)


__all__ = [
    "TraceMiddleware", "DEFAULT_TRACE_HEADER", "trace_header_name", "trace_enabled",
    "get_trace_id", "ensure_trace_id", "set_trace_id", "reset_trace_id", "new_trace_id",
]
