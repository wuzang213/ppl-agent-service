"""评评哩本地生活 Agent —— 基于 LangGraph 节点编排。

对外接口（generate_sse / get_messages / clear_messages）。

"""

import json
import os
from typing import Any, Optional

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
from app.models.message import (
    ROLE_ASSISTANT,
    ROLE_USER,
    append_message,
    count_messages,
    delete_messages,
    list_messages,
)

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

    async def _safe_append(self, thread_id: str, role: str, content: str, user_id: str = "") -> None:
        """往 append-only 消息表追加一条，失败只记日志。

        刻意不让它抛异常：这是"给用户看的历史"的旁路记录，写库失败不应该
        把整轮对话搞挂（对话本身已经通过 SSE 返回给用户了）。
        """
        try:
            await append_message(thread_id, role, content, user_id)
        except Exception as exc:
            logger.error("写入会话消息失败 thread_id=%s role=%s: %s", thread_id, role, exc)

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

        # 用户消息先落 append-only 表：即使本轮模型失败，用户问过什么也应留在历史里
        await self._safe_append(thread_id, ROLE_USER, message, user_id)

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
            # 记录助手回复：含异常/客户端断开时已生成的部分，保证"用户看到的"与"存下来的"一致
            recorded = final_answer or "".join(full_parts).strip()
            if recorded:
                await self._safe_append(thread_id, ROLE_ASSISTANT, recorded, user_id)
            metrics = finish_metrics(metrics_token, success=success, error=error)
            await save_metrics(metrics)

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------
    async def get_messages(
        self, thread_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> dict[str, Any]:
        """返回会话的完整聊天记录（展示用）。

        数据源是 append-only 的 `agent_message` 表，**不是** checkpoint 里那份会被压缩的
        `messages` —— 这样才能做到"历史压缩只影响喂给模型的上下文"，用户刷新后记录依然完整。

        老会话（迁表之前创建、表里还没有记录）回退到读 checkpoint，保持向后兼容。

        分页：`limit` 省略即返回全部；`offset` 是"从最新往老"跳过的条数。
        """
        total = 0
        try:
            total = await count_messages(thread_id)
        except Exception as exc:
            logger.error("查询会话消息数失败，回退 checkpoint thread_id=%s: %s", thread_id, exc)

        if total:
            try:
                messages = await list_messages(thread_id, limit=limit, offset=offset)
                payload: dict[str, Any] = {"messages": messages, "total": total}
                if limit is not None:
                    payload["has_more"] = (max(0, offset) + len(messages)) < total
                return payload
            except Exception as exc:
                logger.error("读取会话消息失败，回退 checkpoint thread_id=%s: %s", thread_id, exc)

        # 回退：老会话只能读 checkpoint（注意这份是**被压缩过**的窗口）
        return await self._messages_from_checkpoint(thread_id)

    async def _messages_from_checkpoint(self, thread_id: str) -> dict[str, Any]:
        """从 checkpoint 的 State 里取消息（仅用于老会话的向后兼容）。"""
        state = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        if state is None or not state.values:
            return {"messages": []}

        result = []
        for msg in state.values.get("messages", []):
            content = getattr(msg, "content", None)
            if not content:
                continue
            if isinstance(msg, HumanMessage):
                result.append({"role": ROLE_USER, "content": as_text(content)})
            elif isinstance(msg, AIMessage):
                result.append({"role": ROLE_ASSISTANT, "content": as_text(content)})
        return {"messages": result}

    async def clear_messages(self, thread_id: str) -> None:
        """清空会话：checkpoint 与展示用的消息表一起删。"""
        await self.checkpointer.adelete_thread(thread_id)
        try:
            removed = await delete_messages(thread_id)
            if removed:
                logger.info("已删除会话消息 %s 条 thread_id=%s", removed, thread_id)
        except Exception as exc:
            logger.error("删除会话消息失败 thread_id=%s: %s", thread_id, exc)


hmdp_agent = HmdpAgent()

__all__ = ["hmdp_agent"]
