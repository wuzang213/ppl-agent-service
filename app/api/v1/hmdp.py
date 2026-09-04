"""Agent 对话路由。"""

import asyncio
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from app.agents.hmdp_agent import hmdp_agent
from app.models.session import get_sessions
from app.rag.hmdp_rag import get_index_stats, rebuild_blog_index

router = APIRouter()


class HmdpChatRequest(BaseModel):
    message: str
    thread_id: str = ""
    mode: Literal["daily", "recommend"] = "recommend"
    x: Optional[float] = None
    y: Optional[float] = None
    user_id: str = ""
    interrupt_decision: Optional[dict] = None


def _resolve_thread_id(thread_id: str, user_id: str) -> str:
    """前端未持久化 thread_id 时，用用户维度生成默认会话。"""
    return thread_id or f"user:{user_id or 'anonymous'}"

def _ensure_thread_owned(thread_id: str, user_id: str) -> str:
    if not thread_id:
        return _resolve_thread_id(thread_id, user_id)
    resolved = _resolve_thread_id(thread_id, user_id)
    if user_id and thread_id.startswith(f"user:{user_id}"):
        return resolved
    if user_id and not any(s.thread_id == thread_id for s in get_sessions(user_id=user_id)):
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
            mode=request.mode,
            interrupt_decision=request.interrupt_decision,
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
            mode=request.mode,
            interrupt_decision=request.interrupt_decision,
        )
    )


@router.get("/chat/messages")
async def chat_messages(
    thread_id: str = "",
    user_id: str = "",
    user_info: Optional[str] = Header(default=None, alias="user-info"),
):
    uid = user_info or user_id
    resolved = _ensure_thread_owned(thread_id, uid)
    return await hmdp_agent.get_messages(resolved)


@router.delete("/chat/messages")
async def chat_clear_messages(
    thread_id: str = "",
    user_id: str = "",
    user_info: Optional[str] = Header(default=None, alias="user-info"),
):
    uid = user_info or user_id
    resolved = _ensure_thread_owned(thread_id, uid)
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