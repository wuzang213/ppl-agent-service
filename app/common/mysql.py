"""agent-service 的共享关系型存储（MySQL / aiomysql 连接池）。

"""

from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import aiomysql

from app.common.logger import logger

# 本地不配 .env 时默认连本机 MySQL（root / 无密码，常见本地配置）；
# 部署到其它环境必须显式配置 MYSQL_HOST / MYSQL_PASSWORD，切勿在代码里写死内网地址或账号密码
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 3306
DEFAULT_USER = "root"
DEFAULT_PASSWORD = ""
DEFAULT_DB = "hmdp_agent"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_]+$")

_pool: Optional[aiomysql.Pool] = None


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r 不是合法整数，回退到默认值 %s", name, raw, default)
        return default


def safe_identifier(name: str) -> str:
    """校验库名/表名。

    库名要拼进 DDL（标识符不能用占位符传），所以这里做白名单校验，拒绝任何可能改变
    语句结构的内容。参数值仍然一律走 `%s` 占位符，不受此影响。
    """
    if not _IDENTIFIER_RE.match(name):
        raise RuntimeError(f"非法的 MySQL 标识符：{name!r}（只允许字母、数字、下划线）")
    return name


def build_mysql_config() -> dict[str, Any]:
    """从环境变量拼出 aiomysql 的连接参数（**不含 db**，便于先连上去建库）。"""
    return {
        "host": _env_str("MYSQL_HOST", DEFAULT_HOST),
        "port": _env_int("MYSQL_PORT", DEFAULT_PORT),
        "user": _env_str("MYSQL_USER", DEFAULT_USER),
        "password": _env_str("MYSQL_PASSWORD", DEFAULT_PASSWORD),
        "charset": "utf8mb4",
        # 会话接口都是单条语句，不需要显式事务；开 autocommit 避免连接带着未提交事务回池
        "autocommit": True,
        "connect_timeout": 5,
    }


async def _ensure_database(config: dict[str, Any], database: str) -> None:
    """确保目标库存在。

    建库这一步需要"还没选定库"的连接，所以单独 connect 一次再关掉。做这一步的原因是：
    库不存在时 aiomysql 建池会直接抛 `Unknown database`，报错信息不会告诉你"该先建库"，
    启动期排查成本高；这里顺手建掉，让部署少一个必做步骤。
    """
    conn = await aiomysql.connect(**config)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
            )
    finally:
        conn.close()


async def init_mysql_pool(*, create_database: bool = True) -> aiomysql.Pool:
    """初始化全局连接池（幂等）。在 FastAPI lifespan 启动阶段调用。"""
    global _pool
    if _pool is not None:
        return _pool

    config = build_mysql_config()
    database = safe_identifier(_env_str("MYSQL_DB", DEFAULT_DB))
    minsize = _env_int("MYSQL_POOL_MIN", 1)
    maxsize = max(minsize, _env_int("MYSQL_POOL_MAX", 5))

    try:
        if create_database:
            await _ensure_database(config, database)
        _pool = await aiomysql.create_pool(
            db=database,
            minsize=minsize,
            maxsize=maxsize,
            pool_recycle=_env_int("MYSQL_POOL_RECYCLE", 3600),
            **config,
        )
    except Exception as exc:  # noqa: BLE001 - 统一包装成可读的启动失败信息
        raise RuntimeError(
            f"连接 MySQL 失败（{config['host']}:{config['port']}，db={database}）：{exc}；"
            "请检查 MYSQL_* 环境变量、MySQL 是否已启动、账号是否有建库/建表权限。"
        ) from exc

    logger.info(
        "MySQL 连接池已就绪：%s:%s/%s（minsize=%s, maxsize=%s）",
        config["host"], config["port"], database, minsize, maxsize,
    )
    return _pool


async def close_mysql_pool() -> None:
    """关闭连接池并等待所有连接归还。在 FastAPI lifespan 收尾阶段调用。"""
    global _pool
    if _pool is None:
        return
    _pool.close()
    await _pool.wait_closed()
    _pool = None
    logger.info("MySQL 连接池已关闭")


def get_mysql_pool() -> aiomysql.Pool:
    if _pool is None:
        raise RuntimeError("MySQL 连接池尚未初始化，请确认应用启动时调用了 init_mysql_pool()")
    return _pool


@asynccontextmanager
async def acquire_cursor(*, dict_rows: bool = True) -> AsyncIterator[Any]:
    """借一个 cursor 用，退出时自动归还连接到池。

    `autocommit=True` 已开，所以这里不做 commit / rollback —— 单条语句失败不会留下半截事务。
    需要多语句原子性的场景请自行 `pool.acquire()` 后显式 begin/commit。
    """
    pool = get_mysql_pool()
    async with pool.acquire() as conn:
        cursor_cls = aiomysql.DictCursor if dict_rows else aiomysql.Cursor
        async with conn.cursor(cursor_cls) as cur:
            yield cur
