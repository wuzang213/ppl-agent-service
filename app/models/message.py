"""智能体会话消息（append-only，供前端展示），与 checkpoint 解耦：后者被 trim_history 压缩只服务推理，本表只追加不删除以保记录完整。
排序用自增 id，content 用 MEDIUMTEXT，时间戳应用写 datetime.now()+datetime(6)（UTC，不用 DB 默认值）；不建表，DDL 权威 sql/agent_message.sql（payload 列为增量脚本 agent_message_add_payload.sql，只增不改）。"""


import json
from datetime import datetime
from typing import Any, List, Optional

from app.common.logger import logger
from app.common.mysql import acquire_cursor

# 单条消息正文上限（与 sql/agent_message.sql 里的 MEDIUMTEXT 对应）。
# 这里只用于入参校验，目的是把"超长"变成明确的 400，而不是让数据库报错变 500。
CONTENT_MAX = 1_000_000

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

TABLE_NAME = "agent_message"


async def append_message(
    thread_id: str,
    role: str,
    content: str,
    user_id: str = "",
    payload: str | None = None,
) -> None:
    """追加一条消息；调用方需自行兜底异常（写入失败不应中断对话）。
    payload 为结构化载荷的 JSON 字符串（行程/澄清/建议 specs/0012），与 content 分离：正文只放摘要，大对象走该列仅供前端回看；None 表示无结构化载荷（绝大多数消息如此）。"""
    if not content:
        return
    async with acquire_cursor(dict_rows=False) as cur:
        await cur.execute(
            f"""
            INSERT INTO `{TABLE_NAME}` (thread_id, user_id, role, content, payload, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (thread_id, user_id or "", role, content, payload, datetime.now()),
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
    """按时间正序返回消息，支持"从最新往前翻页"：`offset` 是从最新往老跳过的条数，`limit` 是本次取多少条
    （都省略则返回全部）。返回值始终按时间正序；`limit=20, offset=20` 即再往前的 20 条。
    """
    sql = f"SELECT role, content, payload, created_at FROM `{TABLE_NAME}` WHERE thread_id = %s"
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
            # 结构化载荷：解析成对象给前端（坏 JSON 只记 warning、返回 None，不让整条历史读失败）
            "payload": _parse_payload(row["payload"]),
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


def _parse_payload(raw: Any) -> Any:
    """把 `payload` 列（JSON 文本）解析成对象给前端；空值返回 `None`。
    坏 JSON 只记 warning 并返回 None，绝不抛错：单条历史消息载荷损坏不应让整个会话列表（GET /chat/messages）读不出来。"""
    if raw is None or raw == "":
        return None
    if isinstance(raw, (dict, list)):
        return raw          # 驱动在个别配置下会把 JSON 列直接反序列化成对象
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        logger.warning("消息 payload 不是合法 JSON，按空处理: %s", exc)
        return None
