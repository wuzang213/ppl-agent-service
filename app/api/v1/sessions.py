from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException, Query

from app.agents.hmdp_agent import hmdp_agent
from app.models.session import (
    USER_ID_MAX,
    SessionCreate,
    SessionResponse,
    get_session,
    get_sessions,
    create_session,
    delete_session
)


router = APIRouter()


def _require_user(user_info: Optional[str], user_id: Optional[str] = None) -> str:
    uid = user_info or user_id or ""
    if not uid:
        raise HTTPException(status_code=403, detail="缺少用户信息")
    # 用户标识存的是 varchar(USER_ID_MAX)，超长会在 INSERT 时抛 DataError（500）。
    # 在这里拦成 400，让错误类型对得上（是入参问题，不是服务端问题）。
    if len(uid) > USER_ID_MAX:
        raise HTTPException(status_code=400, detail=f"用户标识过长（最多 {USER_ID_MAX} 字符）")
    return uid


@router.post("/sessions", response_model=SessionResponse, tags=["会话"])
async def create_new_session(
    session: SessionCreate,
    user_info: Optional[str] = Header(default=None, alias="user-info"),
):
    """创建新会话"""
    if user_info:
        session.user_id = user_info
    return await create_session(session)


@router.get("/sessions", response_model=List[SessionResponse], tags=["会话"])
async def list_sessions(
    user_id: Optional[str] = Query(None, description="用户ID"),
    biz_type: Optional[str] = Query(None, description="业务类型"),
    user_info: Optional[str] = Header(default=None, alias="user-info"),
):
    """查询会话列表，只返回当前用户自己的会话"""
    uid = _require_user(user_info, user_id)
    return await get_sessions(user_id=uid, biz_type=biz_type)


@router.delete("/sessions/{thread_id}", tags=["会话"])
async def remove_session(
    thread_id: str,
    user_info: Optional[str] = Header(default=None, alias="user-info"),
):
    """删除会话及其checkpoint，先清状态再删 DB 行

    区分两种拒绝：会话**不存在** → 404；存在但**不属于当前用户** → 403。
    （原实现先做归属校验，导致"已删除的会话"被判成 403，下面那句 `not deleted -> 404`
    永远不可达；现改为先按主键查存在性。）
    """
    uid = _require_user(user_info)
    session = await get_session(thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != uid:
        raise HTTPException(status_code=403, detail="无权删除该会话")

    try:
        await hmdp_agent.clear_messages(thread_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"清理会话状态失败: {exc}")

    deleted = await delete_session(thread_id)
    if not deleted:
        # 并发下被别的请求抢先删掉
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted successfully"}


@router.get("/sessions/{thread_id}/messages", tags=["会话"])
async def get_session_messages(
    thread_id: str,
    user_info: Optional[str] = Header(default=None, alias="user-info"),
):
    """获取会话的历史消息

    与 DELETE 同样区分：会话不存在 → 404，不属于当前用户 → 403
    （原实现把两种情况都判成 403，这里一并修正）。
    """
    uid = _require_user(user_info)
    session = await get_session(thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.user_id != uid:
        raise HTTPException(status_code=403, detail="无权查看该会话")
    try:
        return await hmdp_agent.get_messages(thread_id)
    except Exception as e:
        return {"messages": [], "error": str(e)}