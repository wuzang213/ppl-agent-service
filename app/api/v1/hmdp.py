"""Agent 对话路由。"""

import asyncio
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, field_validator
from sse_starlette import EventSourceResponse

from app.agents.hmdp_agent import hmdp_agent
from app.models.session import get_session
from app.rag.hmdp_rag import get_index_stats, rebuild_blog_index

router = APIRouter()


class HmdpChatRequest(BaseModel):
    #: 必须容忍非字符串：前端澄清卡选项 value 可能为数字，直接回传会触发 Pydantic 422 string_type。
    #: 校验前归一化为字符串，兼容前端发 value / label / 字符串数字三种写法。
    #: 只放宽类型不改语义：空串仍由下游 entry_router 按"空问题"处理。
    message: str = ""

    @field_validator("message", mode="before")
    @classmethod
    def _coerce_message(cls, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, float) and value.is_integer():
            # JSON 里 `2` 与 `2.0` 都能到这儿；浮点整数值要还原成 `"2"`，
            # 否则槽位抽出会看到 `"2.0"`（多一位小数，虽不致命但没必要）。
            return str(int(value))
        return str(value)

    thread_id: str = ""
    x: Optional[float] = None
    y: Optional[float] = None
    user_id: str = ""
    interrupt_decision: Optional[dict] = None
    #: 结构化动作（plans/0019）：`{type, args}`，预约写路径唯一入口（booking 建单 / confirm 确认）。
    #: 不复用 interrupt_decision：旧中断流程已移除，该参数被忽略，复用会让确认动作静默失效。
    action: Optional[dict] = None
    #: 结构化澄清载荷（`specs/0025`）：前端澄清卡把每个槽位按**标准化结构**回传
    #: （时间 `{type,start,end}` / 人数·预算数值 / 区域·场景枚举或文本），后端**直接映射槽位、跳过文本抽槽链路**。
    #: 与原 `message` 文本通道**并存、向下兼容**（字段可选；不传时走原文本抽槽 + 规则校验）。不影响预约 `action`。
    clarify_slots: Optional[dict] = None


def _resolve_thread_id(thread_id: str, user_id: str) -> str:
    """前端未持久化 thread_id 时，用用户维度生成默认会话。"""
    return thread_id or f"user:{user_id or 'anonymous'}"

async def _ensure_thread_owned(thread_id: str, user_id: str) -> str:
    if not thread_id:
        return _resolve_thread_id(thread_id, user_id)
    resolved = _resolve_thread_id(thread_id, user_id)
    if user_id and thread_id.startswith(f"user:{user_id}"):
        return resolved
    if user_id:
        # 会话元数据已搬到 MySQL，查询是异步的（原 SQLite 是同步直连）；
        # 用主键查单条，避免"列出该用户全部会话来比对"
        session = await get_session(thread_id)
        if session is None or session.user_id != user_id:
            raise HTTPException(status_code=403, detail="无权访问该会话")
    return resolved


@router.post("/chat/stream")
async def chat_stream(
    request: HmdpChatRequest,
    user_info: Optional[str] = Header(default=None, alias="user-info"),
):
    user_id = user_info or request.user_id
    thread_id = _resolve_thread_id(request.thread_id, user_id)
    return EventSourceResponse(
        hmdp_agent.generate_sse(
            thread_id=thread_id,
            message=request.message,
            user_id=user_id,
            x=request.x,
            y=request.y,
            interrupt_decision=request.interrupt_decision,
            action=request.action,
            clarify_slots=request.clarify_slots,
        )
    )


@router.post("/chat/send")
async def chat_send(
    request: HmdpChatRequest,
    user_info: Optional[str] = Header(default=None, alias="user-info"),
):
    user_id = user_info or request.user_id
    thread_id = _resolve_thread_id(request.thread_id, user_id)
    return EventSourceResponse(
        hmdp_agent.generate_sse(
            thread_id=thread_id,
            message=request.message,
            user_id=user_id,
            x=request.x,
            y=request.y,
            interrupt_decision=request.interrupt_decision,
            action=request.action,
            clarify_slots=request.clarify_slots,
        )
    )


@router.get("/chat/messages")
async def chat_messages(
    thread_id: str = "",
    user_id: str = "",
    limit: Optional[int] = Query(
        None, ge=1, le=500,
        description="最多返回多少条（从最新往前取）；省略则返回全部",
    ),
    offset: int = Query(0, ge=0, description="从最新往老跳过的条数，用于翻页"),
    user_info: Optional[str] = Header(default=None, alias="user-info"),
):
    """会话历史。返回 append-only 消息表里的完整记录，不受历史压缩影响。"""
    uid = user_info or user_id
    resolved = await _ensure_thread_owned(thread_id, uid)
    return await hmdp_agent.get_messages(resolved, limit=limit, offset=offset)


@router.delete("/chat/messages")
async def chat_clear_messages(
    thread_id: str = "",
    user_id: str = "",
    user_info: Optional[str] = Header(default=None, alias="user-info"),
):
    uid = user_info or user_id
    resolved = await _ensure_thread_owned(thread_id, uid)
    await hmdp_agent.clear_messages(resolved)
    return {"success": True}


@router.post("/agent/rag/rebuild")
async def rag_rebuild():
    try:
        stats = await asyncio.to_thread(rebuild_blog_index)
        return {"success": True, **stats}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/agent/rag/status")
async def rag_status():
    return await asyncio.to_thread(get_index_stats)