"""智能体会话消息（append-only，供前端展示）。

## 为什么必须单独一张表

LangGraph checkpoint 里那份 `messages` 是**喂给模型的上下文**，会被 `trim_history`
压缩：超过历史窗口后，早期消息会被 `RemoveMessage` 删掉、只留一段摘要。

而用户**刷新页面必须还能看到完整的聊天记录**。如果前端历史直接读 checkpoint 里那份
`messages`，就会出现"聊到第 9 轮开始，刷新后最早那几轮不见了"——数据其实还在摘要里，
但用户看不到，主观上就是"我的记录丢了"。

所以两者必须**解耦**：
- checkpoint（含压缩摘要）→ 只服务于模型推理，省 token；
- 本表 → 只服务于展示，**只追加、永不删除**（除非显式删会话）。

## 设计要点

- 主键自增 `id` 同时充当**时间序**：排序用 `id` 而不是 `created_at`，
  避免同一秒内多条消息排序不确定（`created_at` 只用于展示）。
- `content` 用 `MEDIUMTEXT`：单条回答可能很长，`VARCHAR` 有长度上限
  （超长会在 INSERT 时抛 `DataError` → 500）。
- 时间戳由应用写 `datetime.now()` + `datetime(6)` 保留微秒，**不使用数据库默认值**：
  MySQL 服务器时区是 UTC，用 `CURRENT_TIMESTAMP` 存下来的值会比前端展示的时间少 8 小时。
- 索引 `(thread_id, id)` 覆盖"按会话取最近 N 条 / 翻页"的查询。

## ⚠️ 表结构不在代码里维护

本模块**只做数据读写，不建表**。表结构（DDL）的唯一权威是 `sql/agent_message.sql`，
由部署方执行；应用启动时只**校验表是否存在**（见 `app.common.mysql.verify_tables`），
缺失则启动即失败并提示该执行哪个脚本。这样避免"代码里一份 DDL + 脚本里一份 DDL"两处漂移。
"""

from datetime import datetime
from typing import List, Optional

from app.common.mysql import acquire_cursor

# 单条消息正文上限（与 sql/agent_message.sql 里的 MEDIUMTEXT 对应）。
# 这里只用于入参校验，目的是把"超长"变成明确的 400，而不是让数据库报错变 500。
CONTENT_MAX = 1_000_000

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

TABLE_NAME = "agent_message"


async def append_message(
    thread_id: str, role: str, content: str, user_id: str = ""
) -> None:
    """追加一条消息。调用方需自行兜底异常（写入失败不应中断对话）。"""
    if not content:
        return
    async with acquire_cursor(dict_rows=False) as cur:
        await cur.execute(
            f"""
            INSERT INTO `{TABLE_NAME}` (thread_id, user_id, role, content, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (thread_id, user_id or "", role, content, datetime.now()),
        )


async def count_messages(thread_id: str) -> int:
    """该会话的消息总数。表里没有记录（老会话）时返回 0，调用方据此回退 checkpoint。"""
    async with acquire_cursor() as cur:
        await cur.execute(
            f"SELECT COUNT(*) AS n FROM `{TABLE_NAME}` WHERE thread_id = %s",
            (thread_id,),
        )
        row = await cur.fetchone()
    return int(row["n"]) if row else 0


async def list_messages(
    thread_id: str, limit: Optional[int] = None, offset: int = 0
) -> List[dict]:
    """按时间正序返回消息，支持"从最新往前翻页"。

    分页语义：`offset` 是**从最新往老**跳过的条数，`limit` 是本次取多少条
    （都省略 => 返回全部）。返回值始终按时间正序，方便前端直接从头追加渲染。

    - `limit=20, offset=0`  → 最近 20 条
    - `limit=20, offset=20` → 再往前的 20 条
    """
    sql = f"SELECT role, content, created_at FROM `{TABLE_NAME}` WHERE thread_id = %s"
    params: List[object] = [thread_id]
    if limit is not None:
        # 先按 id 倒序取"最近的 limit 条"，再在 Python 侧反转成正序
        sql += " ORDER BY id DESC LIMIT %s OFFSET %s"
        params.extend([int(limit), max(0, int(offset))])
    else:
        sql += " ORDER BY id ASC"

    async with acquire_cursor() as cur:
        await cur.execute(sql, params)
        rows = await cur.fetchall()

    result = [
        {
            "role": row["role"],
            "content": row["content"],
            "created_at": _iso(row["created_at"]),
        }
        for row in rows
    ]
    if limit is not None:
        result.reverse()
    return result


async def delete_messages(thread_id: str) -> int:
    """删除该会话的全部消息（删会话时调用）。返回删除条数。"""
    async with acquire_cursor(dict_rows=False) as cur:
        await cur.execute(
            f"DELETE FROM `{TABLE_NAME}` WHERE thread_id = %s", (thread_id,)
        )
        return cur.rowcount


def _iso(value: object) -> str:
    """DATETIME(6) 读回来是 datetime，统一转 ISO 字符串给前端。"""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
