"""agent-service 的共享关系型存储（MySQL / aiomysql 连接池）。

## 为什么需要它

会话的**元数据**（`sessions` 表：thread_id / user_id / biz_type / name / 时间戳）原本存在本地文件
`db/sessions.db`（`sqlite3` 直连）。单进程下没问题，但多实例部署时：

- 实例 A 创建的会话，在实例 B 的 `GET /sessions` 里**看不到**；
- `DELETE /sessions/{id}` 的归属校验查不到记录，会把合法请求误判成 403。

对话历史（checkpoint）已经搬到 Redis，会话元数据必须同样搬到**进程外共享**存储，多实例才成立。

## 存储选型

Java 侧四个业务服务已经共用一个 MySQL 实例（各自独立 database：`hmdp_blog` / `hmdp_shop` /
`hmdp_user` / `hmdp_voucher`），而 `sessions` 是一张规整的小表，正好适合关系型库，所以按同一
口径接入：**同一个实例、独立 database `hmdp_agent`**（独立库是本项目的既有约定，避免跨服务
建外键/互相读表）。连接参数默认取 Nacos `shared-jdbc.yaml` 的同口径（本机 MySQL，root / 无密码）；
部署到其它环境请通过 `MYSQL_*` 环境变量覆盖，**切勿在代码里写死内网地址或账号密码**。

## 为什么用连接池

aiomysql 是 asyncio 驱动，每次 `connect()` 都是一次真实 TCP + 握手 + 认证往返。会话接口虽然
QPS 不高，但把建连代价摊到每个请求上没有意义，所以这里维护一个全局 `Pool`：启动时
`init_mysql_pool()` 建池并预热，请求期 `acquire_cursor()` 借连接，`close_mysql_pool()` 归还。

`pool_recycle` 默认 1 小时：MySQL 侧 `wait_timeout` 默认 8 小时，空闲超时后服务端会单方面断开，
池里就会留下"看起来还活着"的死连接，第一个用到的请求会莫名报错。

## 建库建表不在这里

本模块**只负责连接**，不建库、不建表。库和表由部署方执行 `sql/` 下的脚本创建
（DDL 的唯一权威，见 `sql/README.md`），避免"代码里一份 DDL + 脚本里一份 DDL"两处漂移。
应用启动时用 `verify_tables()` 校验必需的表是否存在，缺失即 fail fast 并提示该执行哪个脚本。
"""

from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterable, Optional

import aiomysql

from app.common.logger import logger

# 本地不配 .env 时默认连本机 MySQL（root / 无密码，常见本地配置）；
# 部署请通过 MYSQL_* 环境变量覆盖，切勿在代码里写死内网地址或账号密码
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


async def init_mysql_pool() -> aiomysql.Pool:
    """初始化全局连接池（幂等）。在 FastAPI lifespan 启动阶段调用。

    **不建库、不建表** —— 库/表由部署方执行 `sql/` 下的脚本创建。
    库不存在时 aiomysql 会抛 `Unknown database`，下面会把它包装成带建表指引的可读错误。
    """
    global _pool
    if _pool is not None:
        return _pool

    config = build_mysql_config()
    database = safe_identifier(_env_str("MYSQL_DB", DEFAULT_DB))
    minsize = _env_int("MYSQL_POOL_MIN", 1)
    maxsize = max(minsize, _env_int("MYSQL_POOL_MAX", 5))

    try:
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
            "请检查 MYSQL_* 环境变量与 MySQL 是否已启动；"
            "若提示 Unknown database，说明库还没建 —— 请先执行 sql/agent_session.sql（脚本内含建库）。"
        ) from exc

    logger.info(
        "MySQL 连接池已就绪：%s:%s/%s（minsize=%s, maxsize=%s）",
        config["host"], config["port"], database, minsize, maxsize,
    )
    return _pool


# 表名 → 建表脚本，仅用于缺表时给出"该执行哪个文件"的指引
_TABLE_SCRIPTS = {
    "sessions": "sql/agent_session.sql",
    "agent_message": "sql/agent_message.sql",
}


async def verify_tables(tables: Iterable[str]) -> None:
    """启动时校验必需的表是否存在；缺失则抛错并指出对应的建表脚本。

    刻意**不自动建表**：表结构由部署方执行 `sql/` 下的脚本创建，代码里只保留读写逻辑。
    但"不建表"≠"不检查" —— 缺表时在**启动期**就明确失败，好过运行时第一个请求才报错
    （那时错误信息离根因很远，排查成本高）。
    """
    names = list(dict.fromkeys(tables))
    if not names:
        return

    database = safe_identifier(_env_str("MYSQL_DB", DEFAULT_DB))
    placeholders = ", ".join(["%s"] * len(names))
    async with acquire_cursor() as cur:
        await cur.execute(
            "SELECT table_name AS t FROM information_schema.tables "
            f"WHERE table_schema = %s AND table_name IN ({placeholders})",
            [database, *names],
        )
        rows = await cur.fetchall()

    found = {row["t"] for row in rows}
    missing = [n for n in names if n not in found]
    if missing:
        hints = "；".join(f"`{n}` → {_TABLE_SCRIPTS.get(n, 'sql/')}" for n in missing)
        raise RuntimeError(
            f"数据库 {database} 缺少必需的表：{', '.join(missing)}。"
            f"请先执行建表脚本（{hints}）后重启服务。"
        )

    logger.info("数据表校验通过：%s（库 %s）", ", ".join(names), database)


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
