"""把 LangChain / LangGraph 的回调接成**账本事件**（`specs/0013` T3）。用 `BaseCallbackHandler` 一处接线覆盖全部调用点（逐个埋会漏且不报错）。结束事件无 metadata，只能靠 `run_id` 与开始事件配对；纪律：绝不抛异常、拿不到 `run_id` 不写、只认 LangGraph 节点链。usage 锚点改由调用方自查 `response`（异步链路上回调写 `ContextVar` 是断的），本模块唯一职责是写账本。"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from app.agents import ledger
from app.common.logger import logger

#: 参与 fingerprint 的最前面的消息条数（见 `_fingerprint`）。
_FP_HEAD = 3


def _fingerprint(messages: list[Any]) -> str:
    """上下文"头部形状"的指纹：取前 `_FP_HEAD` 条**非系统**消息的文本做 sha1。`trim_history` 下轮要能独立复算同一值（它只看 State 的 messages，节点自拼的 system 不在其中）；压缩删的是最老几条，头部一变即锚点失效；取尾部每轮必变，锚点永远用不上。"""
    texts: list[str] = []
    for msg in messages or []:
        if getattr(msg, "type", "") == "system":
            continue
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            content = "".join(
                str(i.get("text", "")) for i in content if isinstance(i, dict)
            )
        texts.append(str(content or ""))
        if len(texts) >= _FP_HEAD:
            break
    return hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:16]


def _node_of(metadata: Any) -> str:
    md = metadata or {}
    if isinstance(md, dict):
        return str(md.get("langgraph_node") or "")
    return ""


def _model_of(serialized: Any) -> str:
    """从 `serialized` 里掏模型标识符（不同 provider 放的键不一样，逐个试）。"""
    if not isinstance(serialized, dict):
        return ""
    kwargs = serialized.get("kwargs") or {}
    for key in ("model", "model_name", "model_id", "deployment_name"):
        value = kwargs.get(key)
        if value:
            return str(value)
    ident = serialized.get("id")
    if isinstance(ident, list) and ident:
        return str(ident[-1])
    return ""


def _int_or_zero(value: Any) -> int:
    """把 provider 塞进来的 token 数转成 int（可能是 `None` / 字符串 / 浮点）。"""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _from_usage_dict(usage: Any) -> tuple[int, int]:
    """从 provider 的原始 usage 字典里取 `(prompt, completion)`（键名各家不同，逐个试）。"""
    if not isinstance(usage, dict) or not usage:
        return 0, 0
    return (
        _int_or_zero(usage.get("prompt_tokens") or usage.get("input_tokens")),
        _int_or_zero(usage.get("completion_tokens") or usage.get("output_tokens")),
    )


def usage_of(response: Any) -> tuple[int, int]:
    """从**任意**模型返回体里掏真实 usage ⇒ `(prompt_tokens, completion_tokens)`；掏不到 `(0, 0)`。兼容四种形态（AIMessage / AIMessageChunk 聚合体 / LLMResult / generations[][].message），按 usage_metadata → response_metadata → llm_output → generations 顺序取。流式必须传**聚合体**（最后一个 chunk 的 usage_metadata 为 None，传它会静默拿到 0）；`prompt_tokens` 已含 cached，不另加（避免 double count）。"""
    own = getattr(response, "usage_metadata", None)
    if own:
        prompt, completion = _from_usage_dict(own)
        if prompt or completion:
            return prompt, completion

    meta = getattr(response, "response_metadata", None)
    if isinstance(meta, dict):
        prompt, completion = _from_usage_dict(
            meta.get("token_usage") or meta.get("usage")
        )
        if prompt or completion:
            return prompt, completion

    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        prompt, completion = _from_usage_dict(
            llm_output.get("token_usage") or llm_output.get("usage")
        )
        if prompt or completion:
            return prompt, completion

    for gens in getattr(response, "generations", None) or []:
        for gen in gens:
            prompt, completion = _from_usage_dict(
                getattr(getattr(gen, "message", None), "usage_metadata", None)
            )
            if prompt or completion:
                return prompt, completion
    return 0, 0


class LedgerCallbackHandler(BaseCallbackHandler):
    """把节点边界 / 模型调用 / 工具调用写成账本事件。一把实例对应**一次请求**（`run_id` 从 ContextVar 取，无需构造参数），在 `hmdp_agent.generate_sse` 里以 `config={"callbacks": [LedgerCallbackHandler()]}` 挂上。"""

    def __init__(self) -> None:
        # run_id → (开始时刻(monotonic), 节点名)：结束事件没有 metadata，只能这样配对
        self._chain_starts: dict[UUID, tuple[float, str]] = {}
        self._llm_starts: dict[UUID, tuple[float, str, str, int]] = {}
        self._tool_starts: dict[UUID, tuple[float, str, str]] = {}
        self._run_id = ledger.get_run_id()

    # ---------------- 节点边界（LangGraph 把每个节点包成一条 chain）----------------
    def on_chain_start(self, serialized: Any, inputs: Any, **kwargs: Any) -> None:
        try:
            node = _node_of(kwargs.get("metadata"))
            if not node:
                return  # 顶层 LangGraph 链 / 内部 RunnableSequence：不算"一个节点"
            run_id = kwargs.get("run_id")
            if run_id is not None:
                self._chain_starts[run_id] = (time.monotonic(), node)
        except Exception as exc:  # noqa: BLE001
            logger.debug("账本回调 on_chain_start 失败（忽略）: %s", exc)

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        try:
            run_id = kwargs.get("run_id")
            started = self._chain_starts.pop(run_id, None)
            if not started:
                return
            t0, node = started
            self._emit_node(node, t0, "success", None)
        except Exception as exc:  # noqa: BLE001
            logger.debug("账本回调 on_chain_end 失败（忽略）: %s", exc)

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        try:
            run_id = kwargs.get("run_id")
            started = self._chain_starts.pop(run_id, None)
            if not started:
                return
            t0, node = started
            self._emit_node(node, t0, "failed", str(error))
        except Exception as exc:  # noqa: BLE001
            logger.debug("账本回调 on_chain_error 失败（忽略）: %s", exc)

    def _emit_node(self, node: str, t0: float, status: str, error: str | None) -> None:
        started_at = datetime.now()
        ledger.record_tool_call(
            run_id=self._run_id or "",
            tool_name=node,
            node=node,
            started_at=started_at,
            finished_at=datetime.now(),
            duration_ms=int((time.monotonic() - t0) * 1000),
            status=status,
            error=error,
        )

    # ---------------- 模型调用 ----------------
    def on_llm_start(
        self, serialized: Any, messages: list[Any], **kwargs: Any
    ) -> None:
        try:
            run_id = kwargs.get("run_id")
            if run_id is None:
                return
            node = _node_of(kwargs.get("metadata"))
            # `messages` 参数在这里**用不到**了（锚点改由调用方自查，见模块 docstring），
            #    但签名必须保留：LangChain 按位置传，少一个参数会 TypeError。
            self._llm_starts[run_id] = (time.monotonic(), node, _model_of(serialized))
        except Exception as exc:  # noqa: BLE001
            logger.debug("账本回调 on_llm_start 失败（忽略）: %s", exc)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            run_id = kwargs.get("run_id")
            started = self._llm_starts.pop(run_id, None)
            if not started:
                return
            t0, node, model = started
            prompt_tokens, completion_tokens = usage_of(response)
            ledger.record_llm_call(
                run_id=self._run_id or "",
                node=node,
                model=model,
                started_at=datetime.now(),
                duration_ms=int((time.monotonic() - t0) * 1000),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                status="success",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("账本回调 on_llm_end 失败（忽略）: %s", exc)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        try:
            run_id = kwargs.get("run_id")
            started = self._llm_starts.pop(run_id, None)
            if not started:
                return
            t0, node, model = started
            ledger.record_llm_call(
                run_id=self._run_id or "",
                node=node,
                model=model,
                started_at=datetime.now(),
                duration_ms=int((time.monotonic() - t0) * 1000),
                status="failed",
                error=str(error),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("账本回调 on_llm_error 失败（忽略）: %s", exc)

    # ---------------- 工具调用（`@tool` / `bind_tools` 的真实工具）----------------
    def on_tool_start(self, serialized: Any, input_str: str, **kwargs: Any) -> None:
        try:
            run_id = kwargs.get("run_id")
            if run_id is None:
                return
            name = ""
            if isinstance(serialized, dict):
                name = str(serialized.get("name") or "")
            self._tool_starts[run_id] = (
                time.monotonic(), name, _node_of(kwargs.get("metadata")),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("账本回调 on_tool_start 失败（忽略）: %s", exc)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        try:
            run_id = kwargs.get("run_id")
            started = self._tool_starts.pop(run_id, None)
            if not started:
                return
            t0, name, node = started
            ledger.record_tool_call(
                run_id=self._run_id or "",
                tool_name=name,
                node=node,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                duration_ms=int((time.monotonic() - t0) * 1000),
                status="success",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("账本回调 on_tool_end 失败（忽略）: %s", exc)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        try:
            run_id = kwargs.get("run_id")
            started = self._tool_starts.pop(run_id, None)
            if not started:
                return
            t0, name, node = started
            ledger.record_tool_call(
                run_id=self._run_id or "",
                tool_name=name,
                node=node,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                duration_ms=int((time.monotonic() - t0) * 1000),
                status="failed",
                error=str(error),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("账本回调 on_tool_error 失败（忽略）: %s", exc)


#: 对外名字（`nodes.trim_history` 要独立复算同一个指纹，故不能用下划线私有名）。
fingerprint_of = _fingerprint


__all__ = ["LedgerCallbackHandler", "fingerprint_of", "usage_of"]
