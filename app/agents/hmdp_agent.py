"""评评哩本地生活 Agent —— 基于 LangGraph 节点编排；对外接口（generate_sse / get_messages / clear_messages）不变。
历史有两个来源，不要混：checkpoint（Redis）里的 `messages` 是**喂给模型的上下文**，会被 `trim_history` 压缩、
早期消息只留摘要，不能拿来展示；`agent_message` 表（MySQL，append-only）才是展示用的完整记录。"""

import json
import os
import time
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.types import Command

from app.agents import ledger
from app.agents.checkpoint import create_checkpointer
from app.agents.graph import build_graph
from app.agents.graph.ledger_callback import LedgerCallbackHandler
from app.agents.graph.llm import get_light_model, get_main_model
from app.agents.graph.nodes import SUMMARY_NODE, as_text
from app.agents.hmdp_metrics import (
    begin_metrics,
    finish_metrics,
    save_metrics,
)
from app.agents.hmdp_tools import reset_user_context, set_user_context
from app.common.logger import logger
from app.common.trace import ensure_trace_id
from app.models.message import (
    ROLE_ASSISTANT,
    ROLE_USER,
    append_message,
    count_messages,
    delete_messages,
    list_messages,
)

load_dotenv()

# 仅 `STREAM_NODES` 内的节点做 token 级流式输出；其余节点（entry_router / grounding / worker / summarize）的 token 不转发。
# `suggest` 自 specs/0019 起走模型流式，必须在此集合内，否则 token 被 continue 静默丢弃、计划轨无打字机效果。
STREAM_NODES = {SUMMARY_NODE, "daily_chat", "suggest"}

# 是否把增量 token 也推给前端（默认关闭，保持"只发一条完整 message"的旧契约）
SSE_DELTA = os.getenv("AGENT_SSE_DELTA", "false").lower() == "true"

# 是否在 SSE 里推送节点进度
SSE_STATUS = os.getenv("AGENT_SSE_STATUS", "true").lower() == "true"

# 节点 -> 给用户的进度文案
# 必须覆盖 `STREAM_NODES` 里的每个节点，否则流式期间前端没有进度可显示。
_NODE_STATUS = {
    "entry_router": "正在理解你的问题",
    "grounding": "正在定位店铺",
    "nearby_worker": "正在查询附近店铺",
    "shop_worker": "正在查询店铺信息",
    SUMMARY_NODE: "正在整理回答",
    "summarize": "正在整理回答",
    "daily_chat": "正在回复",
    "capability_chat": "正在介绍我的能力",
    "clarify": "需要补充一点信息",
    "plan_slots": "正在了解你的安排",
    "clarify_slots": "需要补充一点信息",
    "plan_worker": "正在查找合适的门店",
    "itinerary": "正在编排行程",
    "replace_stop": "正在替换这一站",
    "suggest": "正在整理回答",
    "booking_confirm": "正在确认预约信息",
}


def _iter_order_cards(update: dict[str, Any]):
    """遍历**任何** node update 里产出的下单卡片（`worker_outputs[*].items[*].order_cards`）。
    按数据而非按节点名发射（`plans/0012` U3）：旧写法 `if node_name == "shop_worker"` 是"枚举生产者"，
    漏掉了 `react_loop` 取回的券卡（算出来了却不发）；"update 里有没有 `order_cards`"才是契约。"""
    for wo in update.get("worker_outputs") or []:
        if not isinstance(wo, dict):
            continue
        for item in wo.get("items") or []:
            if not isinstance(item, dict):
                continue
            for card in item.get("order_cards") or []:
                if isinstance(card, dict):
                    yield card


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
        本地 sqlite（db/hmdp_agent.db）是进程内文件：多实例时同一 thread_id 打到不同实例会读不到历史，容器重建即丢。"""
        self.checkpointer, self._checkpointer_close = await create_checkpointer()

    async def close(self) -> None:
        if self._checkpointer_close is not None:
            await self._checkpointer_close()
            self._checkpointer_close = None
        # 关停前把账本队列里剩下的事件写掉（best-effort：写失败只记日志）
        await ledger.stop()
        logger.info("hmdp agent 连接已关闭")

    # SSE
    def _sse(self, event: str, payload: dict[str, Any]) -> dict[str, str]:
        return {
            "event": event,
            "data": json.dumps(payload, ensure_ascii=False, default=str),
        }

    async def _safe_append(
        self,
        thread_id: str,
        role: str,
        content: str,
        user_id: str = "",
        payload: str | None = None,
    ) -> None:
        """往 append-only 消息表追加一条，失败只记日志；`payload` 为结构化 JSON（行程/澄清/建议，specs/0012），正文只放摘要。"""
        try:
            await append_message(thread_id, role, content, user_id, payload)
        except Exception as exc:
            logger.error("写入会话消息失败 thread_id=%s role=%s: %s", thread_id, role, exc)

    @staticmethod
    def _interrupt_payload(update: Any) -> dict[str, Any] | None:
        """从 `__interrupt__` 取出 `interrupt(payload)` 的载荷。值形如 tuple `(Interrupt(value=…),)`，
        原 `isinstance(update, dict)` 分支会整条吞掉，故对 tuple / list / 单对象都兼容（specs/0012 §12.3.5）。"""
        items = update if isinstance(update, (tuple, list)) else [update]
        for item in items:
            value = getattr(item, "value", None)
            if isinstance(value, dict):
                return value
            if isinstance(item, dict) and isinstance(item.get("value"), dict):
                return item["value"]
        return None

    @staticmethod
    def _has_pending_interrupt(st: Any) -> bool:
        """判断是否真有挂起的 interrupt（澄清/预约确认卡）。
        不能用 `st.next` 非空判据：`aupdate_state` 后 next 也会非空，会把"灌状态后的新提问"误判成续跑，
        致使其既不进 state.messages 也不写 state.query；真实判据是任务上确有 interrupt。"""
        for task in getattr(st, "tasks", ()) or ():
            if getattr(task, "interrupts", None):
                return True
        return False

    async def generate_sse(
        self,
        thread_id: str,
        message: str,
        user_id: str = "",
        x: float | None = None,
        y: float | None = None,
        #: 入口模式提示（`mode_hint`）。已**不再由客户端下发**（`specs/0021`）：路由层去掉了请求字段，
        #:    统一走服务端内部默认 `"recommend"`；仅保留该参数供离线脚本（回放/评测）显式指定。
        #:    scene（daily/recommend/plan）由入口模型判定，本值只作提示 + 降级兜底参考。
        mode: str = "recommend",
        interrupt_decision: dict[str, Any] | None = None,
        action: dict[str, Any] | None = None,
        #: 结构化澄清载荷（`specs/0025`）：澄清卡把各槽位按标准结构回传，续跑时直接映射槽位、跳过抽槽模型。
        clarify_slots: dict[str, Any] | None = None,
    ):
        tokens = set_user_context(user_id, x, y)
        metrics_token = begin_metrics(thread_id, user_id, mode)
        # ---- 运行账本（`specs/0013`）：一次请求 = 一个 run_id，三张表靠它关联 ----
        # trace id 在这里**显式兜底生成**：SSE 的生成器体可能跑在中间件已返回的上下文里，
        #    那时 ContextVar 是空的 ⇒ 至少保证本轮日志与账本有 ID 可串。
        run_id = ledger.new_run_id()
        _run_token = ledger.set_run_id(run_id)
        trace_id = ensure_trace_id()
        _run_started = time.monotonic()
        ledger.start_run(run_id, thread_id=thread_id, user_id=user_id, mode=mode,
                         trace_id=trace_id, query_text=message)
        # 回调**一处接线**覆盖全部模型/工具/节点事件（逐个调用点埋必漏，且漏了不报错）
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [LedgerCallbackHandler()],
        }

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
            # `action` 走 State 声明字段（`plans/0019`）：写操作的唯一入口必须是
            #    **结构化数据**，不能塞进自然语言让模型去理解。
            "action": action,
            # 不在这里写 history_summary，否则每轮都会覆盖掉
            # 上轮压缩生成的滚动摘要，导致多轮对话里历史压缩失效。
            # 首轮该字段缺省为 ""，之后由 checkpoint 持久化。
        }

        # 续跑判定（specs/0012 §10.5）：有 pending interrupt 则用 Command(resume=…) 续跑（之前节点不重跑），否则走新回合 _input。
        # 判据必须用 `_has_pending_interrupt`（任务上真有 interrupt），不能用 `st.next` 非空（aupdate_state 后 next 也会非空）。
        payload: Any = _input
        try:
            st = await self.graph.aget_state(config)
            if self._has_pending_interrupt(st):
                # 挂起态续跑（plans/0019 / specs/0025）：结构化载荷优先，不经模型。
                # 优先级：clarify_slots（澄清卡）→ action（预约确认）→ 用户原话（文本澄清）。三者不写 state.messages，由对应节点回填。
                if clarify_slots:
                    resume_value: Any = {"type": "slot_fill", "slots": clarify_slots}
                    resume_desc = "结构化澄清"
                elif action:
                    resume_value = action
                    resume_desc = "结构化 action"
                else:
                    resume_value = message
                    resume_desc = "用户输入"
                payload = Command(resume=resume_value)
                logger.info(
                    "检测到挂起中断（next=%s），以 %s 续跑",
                    st.next, resume_desc,
                )
        except Exception as exc:
            logger.warning("读取挂起状态失败，按新回合处理: %s", exc)

        success = True
        error = None
        full_parts: list[str] = []
        final_answer = ""
        final_sources: list[str] = []
        #: 本轮若以"澄清挂起"结束，这里存 `clarification` 载荷（用于落 `agent_message.payload`）。
        clarification_payload: dict[str, Any] | None = None
        #: 计划轨产物（`specs/0012` §12.1/§12.2）：同 JSON 落 `payload` 列，刷新后可回看行程。
        itinerary_payload: dict[str, Any] | None = None
        suggestions_payload: list[dict] | None = None
        #: recommend 轨卡片（#117）：店铺卡片 + LLM 推荐理由 + 地图坐标。
        recommend_shops_payload: list[dict] | None = None
        #: recommend 轨**多方案**（`specs/0027` / `plans/0044`）：按偏好维度各一套；与 `shops` 并存。
        schemes_payload: list[dict] | None = None
        #: 预约回执（`booking_result`）—— 供末尾兜底与测试断言
        booking_result_payload: dict | None = None

        # 已**逐 token 流出**过内容的节点集合（用于丢掉随后写回 state 的整段 `AIMessage` 副本）
        streamed_nodes: set[str] = set()

        # 已发过的下单卡片（去重键 `(shop_id, voucher_id)`）。
        # 必须**每次流新建**：提到模块级就会跨请求吞卡（同名券在下一轮再也弹不出来）。
        emitted_cards: set[tuple[Any, Any]] = set()

        try:
            async for chunk in self.graph.astream(
                payload,
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
                    # 去重：`summarize_answer` 既逐 token 流出 `AIMessageChunk`，又把一条完整 `AIMessage` 写回
                    # state（多轮上下文必需），后者会让前端"逐字打完再整段重来"。判据必须写
                    # `isinstance(token, AIMessage) and not isinstance(token, AIMessageChunk)`：Chunk 是子类，只判前者会静默截断。
                    if node in streamed_nodes and isinstance(token, AIMessage) and not isinstance(token, AIMessageChunk):
                        continue
                    if isinstance(token, AIMessageChunk):
                        streamed_nodes.add(node)
                    full_parts.append(content)
                    if SSE_DELTA:
                        yield self._sse("delta", {"type": "delta", "content": content})

                elif event_type == "updates":
                    for node_name, update in (data or {}).items():
                        # `__interrupt__` 的值不是 dict，必须先于 `isinstance(update, dict)` 处理（否则静默吞掉）。
                        # 事件只在此处发（节点内手发会在续跑重放时双发）；两挂起点靠载荷 `type` 区分，不靠节点名。
                        if node_name == "__interrupt__":
                            hit = self._interrupt_payload(update)
                            if hit:
                                final_answer = str(hit.get("text") or "")
                                if hit.get("type") == "booking_confirm":
                                    clarification_payload = hit
                                    yield self._sse("booking_confirm", hit)
                                else:
                                    clarification_payload = hit
                                    yield self._sse("clarification", hit)
                            continue
                        if not isinstance(update, dict):
                            continue
                        if SSE_STATUS and node_name in _NODE_STATUS:
                            yield self._sse(
                                "status",
                                {"type": "status", "node": node_name, "content": _NODE_STATUS[node_name]},
                            )
                        # plans/0012（U3）：只要产出里有 `order_cards` 就发，不枚举"谁产出的"（旧写法只认
                        # `shop_worker` ⇒ `react_loop` 的券卡被静默吞掉）。只给参数（voucher_id / shop_id /
                        # action），不给完整 URL。去重键 `(shop_id, voucher_id)`；`voucher_id` 为 `None` 时不去重。
                        for card in _iter_order_cards(update):
                            key = (card.get("shop_id"), card.get("voucher_id"))
                            if card.get("voucher_id") is not None:
                                if key in emitted_cards:
                                    continue
                                emitted_cards.add(key)
                            yield self._sse(
                                "card",
                                {"type": "order_pending", "card": card},
                            )
                        # 计划轨产物（`specs/0012` §12.4）：**只增不改** —— 既有 delta / message /
                        # status / done / error / card 的语义与相对顺序不受影响。
                        if update.get("itinerary"):
                            itinerary_payload = update["itinerary"]
                            yield self._sse(
                                "itinerary",
                                {"type": "itinerary", "itinerary": itinerary_payload},
                            )
                        if "suggestions" in update:
                            suggestions_payload = list(update.get("suggestions") or [])
                            yield self._sse(
                                "suggestions",
                                {"type": "suggestions", "suggestions": suggestions_payload},
                            )
                        # recommend 轨卡片（#117）：店铺卡片 + LLM 推荐理由 + 地图坐标。
                        # 与 itinerary / suggestions 同级：既有的 delta / message / status /
                        # done / error / card / booking_confirm / clarification 语义不受影响。
                        if "recommend_shops" in update:
                            recommend_shops_payload = list(update.get("recommend_shops") or [])
                            yield self._sse(
                                "shops",
                                {"type": "shops", "shops": recommend_shops_payload},
                            )
                        # recommend 轨**多方案**（`specs/0027` / `plans/0044`）：独立事件、与 `shops` 并存。
                        # 只在**非空**时发（单方案 / 关闭时 `schemes` 为空列表 ⇒ 不发多余事件）。
                        if "schemes" in update:
                            schemes_payload = list(update.get("schemes") or [])
                            if schemes_payload:
                                yield self._sse(
                                    "schemes",
                                    {"type": "schemes", "schemes": schemes_payload},
                                )
                        # 预约**结果**（`plans/0027`）：确认/取消的真实回执。
                        # 以前**从不发射** ⇒ 前端不知道单子已经下了，卡片上那两颗按钮还在
                        #    ⇒ 用户可以**反复点「确认预约」**（重复下单；Java 侧幂等只兜"同店同时间"）。
                        if "booking_result" in update:
                            br = dict(update.get("booking_result") or {})
                            if br:
                                booking_result_payload = br
                                yield self._sse(
                                    "booking_result",
                                    {"type": "booking_result", **br},
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
            if not recorded and clarification_payload:
                # 兜底：载荷万一没有 text 字段，也要留下一句可读的话
                recorded = str(clarification_payload.get("text") or "")
            # 结构化载荷选一个落库：澄清轮存澄清载荷；行程轮存 {itinerary, suggestions}。
            structured: dict[str, Any] | None = None
            if clarification_payload:
                structured = clarification_payload
            elif itinerary_payload is not None:
                structured = {
                    "itinerary": itinerary_payload,
                    "suggestions": suggestions_payload or [],
                }
            # recommend 卡片与预约回执以前**不落 payload** ⇒ 刷新后"推荐卡片/已预约状态"直接消失
            #    （前端 `restorePayload` 早就有 `payload.shops` 分支，只是后端一直没写这个键）。
            if recommend_shops_payload:
                structured = structured or {}
                structured["shops"] = recommend_shops_payload
            # 多方案（`specs/0027`）：与 `shops` 并列落 payload ⇒ 刷新后可回看多套方案。
            if schemes_payload:
                structured = structured or {}
                structured["schemes"] = schemes_payload
            if booking_result_payload:
                structured = structured or {}
                structured["booking_result"] = booking_result_payload
            if recorded:
                await self._safe_append(
                    thread_id,
                    ROLE_ASSISTANT,
                    recorded,
                    user_id,
                    # 结构化载荷单独进 `payload` 列（`plans/0014` 建的列）：正文只放摘要，刷新后可回看
                    json.dumps(structured, ensure_ascii=False) if structured else None,
                )
            metrics = finish_metrics(metrics_token, success=success, error=error)
            await save_metrics(metrics)
            # 账本收尾：真实 token 汇总直接复用 metrics（采集侧早就有了，只是以前没人用）
            ledger.finish_run(
                run_id,
                status="success" if success else "failed",
                duration_ms=int((time.monotonic() - _run_started) * 1000),
                model_call_count=int(metrics.get("model_call_count") or 0),
                prompt_tokens=int(metrics.get("prompt_tokens") or 0),
                completion_tokens=int(metrics.get("completion_tokens") or 0),
                error=error,
            )
            # 立即刷一次：让"刚跑完这一轮"就能查到账本（否则要等后台任务的下一个周期）
            await ledger.flush_pending()
            ledger.reset_run_id(_run_token)

    # 会话
    async def get_messages(
        self, thread_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> dict[str, Any]:
        """返回会话的完整聊天记录（展示用）。
        数据源是 append-only 的 `agent_message` 表，**不是** checkpoint 里那份会被压缩的 `messages`，
        故刷新后记录依然完整；老会话（表里无记录）回退读 checkpoint；`offset` 是"从最新往老"跳过的条数。"""
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


def build_agent(checkpointer: Any | None = None):
    """LangGraph CLI / 平台加载入口 —— `langgraph.json` 的 `ppl_agent` 指向本函数。
    必须是工厂（不能是模块级图对象或别名）：`hmdp_agent` 实例的 `graph` 要等 `await init()` 后才存在，CLI 取不到。
    与 `HmdpAgent.init()` 不同：这里**不建 checkpointer**（否则本地调试会去连生产 Redis）、**不预建模型**。"""
    return build_graph(checkpointer=checkpointer)


hmdp_agent = HmdpAgent()

__all__ = ["hmdp_agent", "build_agent"]
