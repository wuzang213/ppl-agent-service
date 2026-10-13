"""会话元数据的 MySQL 存储（库 `hmdp_agent`，表 `sessions`；连接池见 `app/common/mysql.py`）。原为本地 `sqlite3`
文件、多实例互不可见；对外接口签名与返回结构未变，仅因 aiomysql 改为 async。时间戳由应用写 `datetime.now()`，
DDL 刻意不加 `DEFAULT` / `ON UPDATE CURRENT_TIMESTAMP`（MySQL 时区为 UTC，兜底会混进差 8 小时的值）。"""

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from app.common.mysql import acquire_cursor

# 最长字段长度：user_id / biz_type 是外部（前端 + user-info 头）传入的短标识，
# name 是会话标题。给足余量的同时保留 VARCHAR 以便建索引（SQLite 时代是 TEXT，无长度约束）。
THREAD_ID_MAX = 64
USER_ID_MAX = 64
BIZ_TYPE_MAX = 64
NAME_MAX = 255

# 表名固定为 sessions。与 sql/agent_session.sql 必须保持一致。
TABLE_NAME = "sessions"


# Pydantic 模型
class SessionCreate(BaseModel):
    """创建会话的请求模型"""

    user_id: str = Field(max_length=USER_ID_MAX)
    biz_type: str = Field(max_length=BIZ_TYPE_MAX)
    name: str = Field(max_length=NAME_MAX)


class SessionResponse(BaseModel):
    """会话响应模型"""

    thread_id: str
    user_id: str
    biz_type: str
    name: str
    created_at: str
    updated_at: str


# 表结构不在代码里维护：DDL 唯一权威是 sql/agent_session.sql（部署方执行），本模块只做数据读写；
# 启动时只校验表是否存在（app.common.mysql.verify_tables），缺失即启动失败。
# 要点：`datetime(6)` 保留微秒（isoformat() 与改造前一致）；不写 DEFAULT CURRENT_TIMESTAMP；(user_id, biz_type, updated_at) 联合索引覆盖 GET /sessions。


def _iso(value: object) -> str:
    """把 DATETIME(6) 读回来的 datetime 转成与改造前一致的 ISO 字符串。"""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


async def create_session(session: SessionCreate) -> SessionResponse:
    """创建新会话"""
    thread_id = str(uuid.uuid4())
    now = datetime.now()

    async with acquire_cursor(dict_rows=False) as cur:
        await cur.execute(
            f"""
            INSERT INTO `{TABLE_NAME}`
                (thread_id, user_id, biz_type, name, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (thread_id, session.user_id, session.biz_type, session.name, now, now),
        )

    return SessionResponse(
        thread_id=thread_id,
        user_id=session.user_id,
        biz_type=session.biz_type,
        name=session.name,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )


async def get_sessions(
    user_id: Optional[str] = None, biz_type: Optional[str] = None
) -> List[SessionResponse]:
    """查询会话列表，支持按 user_id 和 biz_type 筛选"""
    sql = (
        f"SELECT thread_id, user_id, biz_type, name, created_at, updated_at "
        f"FROM `{TABLE_NAME}`"
    )
    conditions: List[str] = []
    params: List[str] = []

    if user_id:
        conditions.append("user_id = %s")
        params.append(user_id)
    if biz_type:
        conditions.append("biz_type = %s")
        params.append(biz_type)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    # 时间相同时按 thread_id 兜底排序，保证分页/展示顺序稳定（原来 SQLite 下是未定义的）
    sql += " ORDER BY updated_at DESC, thread_id DESC"

    async with acquire_cursor() as cur:
        await cur.execute(sql, params)
        rows = await cur.fetchall()

    return [
        SessionResponse(
            thread_id=row["thread_id"],
            user_id=row["user_id"],
            biz_type=row["biz_type"],
            name=row["name"],
            created_at=_iso(row["created_at"]),
            updated_at=_iso(row["updated_at"]),
        )
        for row in rows
    ]


async def get_session(thread_id: str) -> Optional[SessionResponse]:
    """按主键取单条会话，不存在返回 None。用于归属/存在性校验：主键查询省一次全表扫描，且能区分「不存在」与「不是你的」。"""
    async with acquire_cursor() as cur:
        await cur.execute(
            f"SELECT thread_id, user_id, biz_type, name, created_at, updated_at "
            f"FROM `{TABLE_NAME}` WHERE thread_id = %s",
            (thread_id,),
        )
        row = await cur.fetchone()

    if row is None:
        return None
    return SessionResponse(
        thread_id=row["thread_id"],
        user_id=row["user_id"],
        biz_type=row["biz_type"],
        name=row["name"],
        created_at=_iso(row["created_at"]),
        updated_at=_iso(row["updated_at"]),
    )


async def delete_session(thread_id: str) -> bool:
    """删除会话。返回是否真的删掉了（调用方据此区分 404）。"""
    async with acquire_cursor(dict_rows=False) as cur:
        await cur.execute(
            f"DELETE FROM `{TABLE_NAME}` WHERE thread_id = %s", (thread_id,)
        )
        return cur.rowcount > 0
