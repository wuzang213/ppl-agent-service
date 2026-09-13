"""评评哩本地生活 Agent —— 基于 LangGraph 节点编排。

对外接口（generate_sse / get_messages / clear_messages）与改造前完全一致，
"""

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.checkpoint import create_checkpointer
from app.agents.graph import build_graph
from app.agents.graph.llm import get_light_model, get_main_model
from app.agents.graph.nodes import as_text
from app.agents.hmdp_metrics import (
    begin_metrics,
    finish_metrics,
    save_metrics,
)
from app.agents.hmdp_tools import reset_user_context, set_user_context
from app.common.logger import logger

load_dotenv()

# 只有这两个节点做 token 级流式输出，中间 worker 阶段不产出给用户看的文本
STREAM_NODES = {"daily_chat", "summarize"}

# 是否把增量 token 也推给前端（默认关闭，保持"只发一条完整 message"的旧契约）
SSE_DELTA = os.getenv("AGENT_SSE_DELTA", "false").lower() == "true"

# 是否在 SSE 里推送节点进度
SSE_STATUS = os.getenv("AGENT_SSE_STATUS", "true").lower() == "true"

# 节点 -> 给用户的进度文案
_NODE_STATUS = {
    "entry_router": "正在理解你的问题",
    "grounding": "正在定位店铺",
    "nearby_worker": "正在查询附近店铺",
    "shop_worker": "正在查询店铺信息",
    "summarize": "正在整理回答",
    "daily_chat": "正在回复",
    "clarify": "需要补充一点信息",
}


class HmdpAgent:

    def __init__(self) -> None:
        self.checkpointer: Any | None = None
        self._checkpointer_close: Any | None = None
        self.graph = None

    async def init(self) -> None:
        # 提前把两个模型建好，配置有问题时启动即失败
        get_main_model()
        get_light_model()
        await self._init_checkpointer()
        self.graph = build_graph(checkpointer=self.checkpointer)
        logger.info("hmdp agent 初始化完成（LangGraph 节点编排）")

    async def _init_checkpointer(self) -> None:
        """按 AGENT_CHECKPOINT_BACKEND 构造 checkpoint 后端（默认 redis）。

        原来是本地 sqlite（db/hmdp_agent.db）—— 那是进程内的本地文件，
        多实例部署时同一个 thread_id 打到不同实例会读不到历史，容器重建即丢。
        """
        self.checkpointer, self._checkpointer_close = await create_checkpointer()

    async def close(self) -> None:
        if self._checkpointer_close is not None:
            await self._checkpointer_close()
            self._checkpointer_close = None
        logger.info("hmdp agent 连接已关闭")

    # ------------------------------------------------------------------
    # SSE
    # ------------------------------------------------------------------
    def _sse(self, event: str, payload: dict[str, Any]) -> dict[str, str]:
        return {
            "event": event,
            "data": json.dumps(payload, ensure_ascii=False, default=str),
        }

    async def generate_sse(
        self,
        thread_id: str,
        message: str,
        user_id: str = "",
        x: float | None = None,
        y: float | None = None,
        mode: str = "recommend",
        interrupt_decision: dict[str, Any] | None = None,
    ):
        tokens = set_user_context(user_id, x, y)
        metrics_token = begin_metrics(thread_id, user_id, mode)
        config = {"configurable": {"thread_id": thread_id}}

        if interrupt_decision:
            logger.warning("节点编排版已移除 interrupt 流程，interrupt_decision 参数被忽略")

        _input = {
            "messages": [HumanMessage(content=message)],
            "query": message,
            "mode_hint": mode,
            "user_id": user_id or "",
            "x": x,
            "y": y,
            # 不在这里写 history_summary，否则每轮都会覆盖掉
            # 上轮压缩生成的滚动摘要，导致多轮对话里历史压缩失效。
            # 首轮该字段缺省为 ""，之后由 checkpoint 持久化。
        }

        success = True
        error = None
        full_parts: list[str] = []
        final_answer = ""
        final_sources: list[str] = []

        try:
            async for chunk in self.graph.astream(
                _input,
                config=config,
                stream_mode=["messages", "updates"],
                version="v2",
            ):
                event_type = chunk.get("type")
                data = chunk.get("data")

                if event_type == "messages":
                    token, meta = data
                    node = (meta or {}).get("langgraph_node")
                    if node not in STREAM_NODES:
                        continue
                    content = as_text(getattr(token, "content", ""))
                    if not content:
                        continue
                    full_parts.append(content)
                    if SSE_DELTA:
                        yield self._sse("delta", {"type": "delta", "content": content})

                elif event_type == "updates":
                    for node_name, update in (data or {}).items():
                        if not isinstance(update, dict):
                            continue
                        if SSE_STATUS and node_name in _NODE_STATUS:
                            yield self._sse(
                                "status",
                                {"type": "status", "node": node_name, "content": _NODE_STATUS[node_name]},
                            )
                        if "answer" in update:
                            final_answer = str(update.get("answer") or "")
                            final_sources = list(update.get("sources") or [])

            answer = final_answer or "".join(full_parts).strip()
            sources = final_sources
            if not sources:
                sources = []

            if answer:
                yield self._sse(
                    "message",
                    {"type": "message", "content": answer, "source": sources},
                )

            yield self._sse("done", {"type": "done", "content": "处理完成"})
        except Exception as exc:
            success = False
            error = str(exc)
            logger.error(f"hmdp SSE 流中断: {exc}", exc_info=True)
            yield self._sse("error", {"type": "error", "error": str(exc)})
        finally:
            reset_user_context(tokens)
            metrics = finish_metrics(metrics_token, success=success, error=error)
            await save_metrics(metrics)

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------
    async def get_messages(self, thread_id: str) -> dict[str, Any]:
        state = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        if state is None or not state.values:
            return {"messages": []}

        result = []
        for msg in state.values.get("messages", []):
            content = getattr(msg, "content", None)
            if not content:
                continue
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": as_text(content)})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": as_text(content)})
        return {"messages": result}

    async def clear_messages(self, thread_id: str) -> None:
        await self.checkpointer.adelete_thread(thread_id)


hmdp_agent = HmdpAgent()

__all__ = ["hmdp_agent"]
