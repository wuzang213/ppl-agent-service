"""checkpoint 后端的选型与构造。

可用环境变量：

==============================  ==========================================================
变量                             说明
==============================  ==========================================================
``AGENT_CHECKPOINT_BACKEND``    ``redis``（默认）| ``sqlite``。出问题时一键回退。
``AGENT_CHECKPOINT_REDIS_URL``  完整连接串，给了就用它（优先级最高）
``REDIS_HOST`` / ``REDIS_PORT``  没给 URL 时按分项拼装
``REDIS_PASSWORD``              留空表示不发 AUTH
``REDIS_DB``                    默认 ``0``
``AGENT_CHECKPOINT_KEY_PREFIX`` Redis key 前缀，默认 ``agent:checkpoint``
``AGENT_CHECKPOINT_SQLITE_PATH`` 回退到 sqlite 时的库文件路径，默认 ``db/hmdp_agent.db``
==============================  ==========================================================

"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Any

from app.common.logger import logger

DEFAULT_BACKEND = "redis"

DEFAULT_KEY_PREFIX = "agent:checkpoint"

DEFAULT_SQLITE_PATH = "db/hmdp_agent.db"

# 连接/读写超时：Redis 地址写错时让启动**快速失败**，
# 而不是长时间卡在 TCP 重连上（默认无超时会让 asetup 的 PING 挂很久）。
CONNECT_TIMEOUT_SECONDS = 5.0

SOCKET_TIMEOUT_SECONDS = 10.0

# (checkpointer, 关闭函数)
CheckpointerHandle = tuple[Any, Callable[[], Awaitable[None]]]


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def build_redis_url() -> str:
    """优先用显式 URL，否则用分项拼装。"""
    explicit = _env("AGENT_CHECKPOINT_REDIS_URL")
    if explicit:
        return explicit
    host = _env("REDIS_HOST", "127.0.0.1")
    port = _env("REDIS_PORT", "6379")
    db = _env("REDIS_DB", "0")
    password = _env("REDIS_PASSWORD")
    auth = f":{password}@" if password else ""
    return f"redis://{auth}{host}:{port}/{db}"


def mask_url(url: str) -> str:
    """日志里不打印密码。"""
    if "@" not in url or "://" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"


async def create_checkpointer() -> CheckpointerHandle:
    """按 ``AGENT_CHECKPOINT_BACKEND`` 构造 checkpointer。

    @return ``(checkpointer, aclose)``，调用方负责在停机时 ``await aclose()``
    """
    backend = _env("AGENT_CHECKPOINT_BACKEND", DEFAULT_BACKEND).lower() or DEFAULT_BACKEND
    if backend == "redis":
        return await _create_redis_checkpointer()
    if backend == "sqlite":
        return await _create_sqlite_checkpointer()
    raise ValueError(
        f"AGENT_CHECKPOINT_BACKEND 只支持 redis / sqlite，当前值：{backend!r}"
    )


async def _create_redis_checkpointer() -> CheckpointerHandle:
    import redis.asyncio as aioredis

    from app.agents.checkpoint.redis_saver import AsyncRedisCheckpointSaver

    url = build_redis_url()
    # decode_responses=False：checkpoint 的序列化结果是二进制，必须保持 bytes
    client = aioredis.from_url(
        url,
        encoding="utf-8",
        decode_responses=False,
        socket_connect_timeout=CONNECT_TIMEOUT_SECONDS,
        socket_timeout=SOCKET_TIMEOUT_SECONDS,
    )
    saver = AsyncRedisCheckpointSaver(client, prefix=_env("AGENT_CHECKPOINT_KEY_PREFIX", DEFAULT_KEY_PREFIX))
    try:
        await saver.asetup()
    except Exception:
        # 连不上就立刻把连接池关掉，别把句柄泄漏出去
        await saver.aclose()
        raise
    logger.info("checkpoint 后端 = redis（url={}，key 前缀={}）", mask_url(url), saver.prefix)
    return saver, saver.aclose


async def _create_sqlite_checkpointer() -> CheckpointerHandle:
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    path = _env("AGENT_CHECKPOINT_SQLITE_PATH", DEFAULT_SQLITE_PATH)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = await aiosqlite.connect(path)
    saver = AsyncSqliteSaver(conn=conn)
    await saver.setup()
    logger.warning(
        "checkpoint 后端 = sqlite（本地文件 {}）：多实例部署时各实例会各存各的会话状态，"
        "仅建议本地调试使用；要切回 Redis 把 AGENT_CHECKPOINT_BACKEND 设为 redis",
        path,
    )
    return saver, conn.close
