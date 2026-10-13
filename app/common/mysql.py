"""agent-service 的共享关系型存储（MySQL / aiomysql 连接池）：会话元数据 `sessions` 原存本地 `db/sessions.db`（sqlite3），多实例下互相看不见、`DELETE` 归属校验还会误判 403，故改为进程外共享存储。
与 Java 侧共用同一 MySQL 实例、**独立 database `hmdp_agent`**（默认连本机 `127.0.0.1:3306`、无密码，可用 `MYSQL_*` 覆盖）；用全局 `Pool` 避免每请求一次建连，`pool_recycle` 默认 1 小时（MySQL `wait_timeout` 默认 8 小时，否则池里会留下"看起来还活着"的死连接）。
本模块**只负责连接**，不建库、不建表（DDL 唯一权威是 `sql/` 下的脚本，见 `sql/README.md`）；启动时 `verify_tables()` 校验必需表，缺失即 fail fast 并提示该执行哪个脚本。"""

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
    """校验库名/表名：库名要拼进 DDL（标识符不能用占位符传），故做白名单校验，
    拒绝任何可能改变语句结构的内容；参数值仍一律走 `%s` 占位符。"""
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
    **不建库、不建表**（库/表由部署方执行 `sql/` 下的脚本创建）；库不存在时 aiomysql 会抛
    `Unknown database`，下面会把它包装成带建表指引的可读错误。"""
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
    # 运行账本（`specs/0013`）：一次请求 1 行头 + 每次模型/工具调用各 1 行，按 run_id 关联。
    # 三张表一同登记在 `main.REQUIRED_TABLES` ⇒ 缺表即启动失败，并提示执行这一个脚本。
    "agent_runs": "sql/agent_ledger.sql",
    "agent_llm_calls": "sql/agent_ledger.sql",
    "agent_tool_calls": "sql/agent_ledger.sql",
}

# (表, 列) → **增量**脚本：列缺失时给出"该执行哪个文件"的指引。`sql/` 只增不改，故历史脚本不动。
_COLUMN_SCRIPTS = {
    ("agent_message", "payload"): "sql/agent_message_add_payload.sql",
    # 账本「上下文」增量列（`specs/0017` T1）：query / 意图路由 / 卡片摘要，供冻结行为回归集。
    # 六列同出一个脚本（`sql/agent_ledger_add_context.sql`，幂等）。
    ("agent_runs", "query_text"): "sql/agent_ledger_add_context.sql",
    ("agent_runs", "scene"): "sql/agent_ledger_add_context.sql",
    ("agent_runs", "sub_intents"): "sql/agent_ledger_add_context.sql",
    ("agent_runs", "depends_on"): "sql/agent_ledger_add_context.sql",
    ("agent_runs", "card_digest"): "sql/agent_ledger_add_context.sql",
    ("agent_runs", "answer_digest"): "sql/agent_ledger_add_context.sql",
}


async def verify_tables(
    tables: Iterable[str],
    required_columns: "dict[str, Iterable[str]] | None" = None,
) -> None:
    """启动时校验必需的表（及可选列）是否存在；缺失则抛错并指出该执行哪个脚本。
    刻意不自动建表/改表（DDL 权威是 sql/ 脚本），缺失即在启动期 fail-fast，好过首请求才报错。
    列校验面向"只增不改"的增量脚本：代码先读新列而脚本未执行，否则首请求才抛 Unknown column。"""
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

    # ---- 列校验（可选）：只看已存在的表 ----
    for table, cols in (required_columns or {}).items():
        want = list(dict.fromkeys(cols))
        if not want or table not in found:
            continue
        col_placeholders = ", ".join(["%s"] * len(want))
        async with acquire_cursor() as cur:
            await cur.execute(
                "SELECT column_name AS c FROM information_schema.columns "
                f"WHERE table_schema = %s AND table_name = %s "
                f"AND column_name IN ({col_placeholders})",
                [database, table, *want],
            )
            col_rows = await cur.fetchall()
        present = {r["c"] for r in col_rows}
        missing_cols = [c for c in want if c not in present]
        if missing_cols:
            hints = "；".join(
                f"`{table}.{c}` → {_COLUMN_SCRIPTS.get((table, c), 'sql/')}"
                for c in missing_cols
            )
            raise RuntimeError(
                f"数据库 {database} 的表 `{table}` 缺少必需列：{', '.join(missing_cols)}。"
                f"请先执行增量脚本（{hints}）后重启服务。"
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
    `autocommit=True` 已开，故这里不做 commit / rollback（单条语句失败不会留下半截事务）；
    需要多语句原子性的场景请自行 `pool.acquire()` 后显式 begin/commit。"""
    pool = get_mysql_pool()
    async with pool.acquire() as conn:
        cursor_cls = aiomysql.DictCursor if dict_rows else aiomysql.Cursor
        async with conn.cursor(cursor_cls) as cur:
            yield cur
