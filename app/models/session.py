"""会话元数据的 MySQL 存储。

**原实现**：`sqlite3` 直连本地文件 `db/sessions.db`。多实例部署下实例之间互不可见。
**现实现**：MySQL 独立库 `hmdp_agent`，表 `sessions`，与 Java 侧共用同一个 MySQL 实例
（连接池与选型说明见 `app/common/mysql.py` 的模块注释）。

对外接口（`create_session` / `get_sessions` / `delete_session`）的**签名与返回结构都没变**，
只是从同步改成了 async —— 因为 aiomysql 是协程驱动，阻塞调用会卡住整个事件循环。

## 时间戳口径

`created_at` / `updated_at` 由**应用侧**写入 `datetime.now()`（本地墙钟时间，带微秒），
DDL 里刻意**不加** `DEFAULT CURRENT_TIMESTAMP` / `ON UPDATE CURRENT_TIMESTAMP`。

原因：MySQL 实例的 `time_zone` 是 `SYSTEM` = **UTC**（实测 `NOW()` 比本地时间少 8 小时），
一旦用 `CURRENT_TIMESTAMP` 兜底，同一列里会混进 UTC 值；而接口是把列值 `isoformat()` 后
原样返回给前端的，混了就会让部分会话的显示时间差 8 小时。

保持"应用侧写入本地时间"也和改造前的行为完全一致：原来 `datetime.now().isoformat()` 产出
`2026-09-13T16:40:27.123456`，现在 `DATETIME(6)` 读回来再 `isoformat()`，字符串一模一样。
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


# ⚠️ 表结构不在代码里维护：本模块只做数据读写，不建表。
# DDL 的唯一权威是 sql/agent_session.sql（由部署方执行）；
# 应用启动时只校验表是否存在（app.common.mysql.verify_tables），缺失即启动失败并提示脚本路径。


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
