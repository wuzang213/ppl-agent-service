from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException, Query

from app.agents.hmdp_agent import hmdp_agent
from app.models.session import (
    SessionCreate,
    SessionResponse,
    get_sessions,
    create_session,
    delete_session
)


router = APIRouter()


def _require_user(user_info: Optional[str], user_id: Optional[str] = None) -> str:
    uid = user_info or user_id or ""
    if not uid:
        raise HTTPException(status_code=403, detail="缺少用户信息")
    return uid


@router.post("/sessions", response_model=SessionResponse, tags=["会话"])
def create_new_session(
    session: SessionCreate,
    user_info: Optional[str] = Header(default=None, alias="user-info"),
):
    """创建新会话"""
    if user_info:
        session.user_id = user_info
    return create_session(session)


@router.get("/sessions", response_model=List[SessionResponse], tags=["会话"])
def list_sessions(
    user_id: Optional[str] = Query(None, description="用户ID"),
    biz_type: Optional[str] = Query(None, description="业务类型"),
    user_info: Optional[str] = Header(default=None, alias="user-info"),
):
    """查询会话列表，只返回当前用户自己的会话"""
    uid = _require_user(user_info, user_id)
    return get_sessions(user_id=uid, biz_type=biz_type)


@router.delete("/sessions/{thread_id}", tags=["会话"])
async def remove_session(
    thread_id: str,
    user_info: Optional[str] = Header(default=None, alias="user-info"),
):
    """删除会话及其checkpoint，先清状态再删 DB 行"""
    uid = _require_user(user_info)
    owned = any(s.thread_id == thread_id for s in get_sessions(user_id=uid))
    if not owned:
        raise HTTPException(status_code=403, detail="无权删除该会话")

    try:
        await hmdp_agent.clear_messages(thread_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"清理会话状态失败: {exc}")

    deleted = delete_session(thread_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted successfully"}


@router.get("/sessions/{thread_id}/messages", tags=["会话"])
async def get_session_messages(
    thread_id: str,
    user_info: Optional[str] = Header(default=None, alias="user-info"),
):
    """获取会话的历史消息"""
    uid = _require_user(user_info)
    owned = any(s.thread_id == thread_id for s in get_sessions(user_id=uid))
    if not owned:
        raise HTTPException(status_code=403, detail="无权查看该会话")
    try:
        return await hmdp_agent.get_messages(thread_id)
    except Exception as e:
        return {"messages": [], "error": str(e)}