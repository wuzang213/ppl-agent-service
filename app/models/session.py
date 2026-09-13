"""会话元数据的 MySQL 存储。
"""

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


# 建表语句。与 sql/agent_session.sql 一致：
#   - `datetime(6)` 保留微秒，读回来 isoformat() 的字符串与改造前完全一致
#   - 刻意不写 DEFAULT CURRENT_TIMESTAMP，见模块注释的「时间戳口径」
#   - (user_id, biz_type, updated_at) 联合索引覆盖 GET /sessions 的过滤 + 排序
CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
    `thread_id`  varchar({THREAD_ID_MAX})  NOT NULL COMMENT '会话ID（uuid4）',
    `user_id`    varchar({USER_ID_MAX})    NOT NULL COMMENT '归属用户ID，来自 user-info 头',
    `biz_type`   varchar({BIZ_TYPE_MAX})   NOT NULL COMMENT '业务类型，前端透传',
    `name`       varchar({NAME_MAX})       NOT NULL COMMENT '会话名称',
    `created_at` datetime(6)               NOT NULL COMMENT '创建时间（应用侧写入本地墙钟时间，不使用数据库默认值）',
    `updated_at` datetime(6)               NOT NULL COMMENT '更新时间（应用侧写入本地墙钟时间，不使用数据库默认值）',
    PRIMARY KEY (`thread_id`),
    KEY `idx_user_biz_updated` (`user_id`, `biz_type`, `updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
  COMMENT='智能体会话元数据（agent-service）'
"""


async def ensure_session_table() -> None:
    """建表（幂等）。在应用启动阶段、连接池就绪之后调用。"""
    async with acquire_cursor(dict_rows=False) as cur:
        await cur.execute(CREATE_TABLE_SQL)


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
    """按主键取单条会话，不存在返回 None。

    用途：归属/存在性校验。以前是「把该用户的全部会话列出来再逐个比对」，
    现在是主键查询 —— 既省一次全表扫描，也能区分「不存在」与「不是你的」。
    """
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
