"""Agent 运行账本（specs/0013）：把一次请求按事件落到 MySQL 三张表（agent_runs / agent_llm_calls / agent_tool_calls；DDL 权威为 sql/agent_ledger.sql）。
旁路纪律：绝不阻塞/上抛（失败只 warning）、无池不写（单测 no-op）、开关一键关停；env 开关（AGENT_LEDGER_*）调用时现读，不做导入期常量。"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from collections import deque
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from app.common import mysql as mysql_mod
from app.common.logger import logger

#: 三张账本表的表名（**唯一真相源**：`main.REQUIRED_TABLES` 与查询接口都引用这里）。
TABLE_RUNS = "agent_runs"
TABLE_LLM_CALLS = "agent_llm_calls"
TABLE_TOOL_CALLS = "agent_tool_calls"
LEDGER_TABLES: tuple[str, ...] = (TABLE_RUNS, TABLE_LLM_CALLS, TABLE_TOOL_CALLS)

#: `error` 列的宽度（见 `sql/agent_ledger.sql`），超长截断而不是让 INSERT 报 DataError。
ERROR_MAX = 500

#: `query_text` 的入库上限（`specs/0017` T1 的列是 TEXT，但没必要存超长 query）。
QUERY_MAX = 2000

_run_id: ContextVar[str | None] = ContextVar("agent_ledger_run_id", default=None)

# 待落库事件：`(表名, 列值字典)`。用 deque + 锁：入队会发生在**同步**回调里
# （LangChain 的 callback 与 `asyncio.to_thread` 的线程），不能假设有事件循环。
_pending: deque[tuple[str, dict[str, Any]]] = deque()
_pending_lock = threading.Lock()
_flush_task: asyncio.Task | None = None


# 配置（调用时读取）
def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning("%s=%r 不是合法整数，回退到默认值 %s", name, raw, default)
        return default
    return value if value > 0 else default


def ledger_enabled() -> bool:
    """账本总开关。**每次调用现读** env，便于运行期关停与单测切换。"""
    return _env_flag("AGENT_LEDGER_ENABLED", True)


def capture_enabled() -> bool:
    """「上下文捕获」开关（specs/0017 §8 Q3）：是否记 query/意图路由/卡片摘要。默认 true（均为对已有行的低频 UPDATE，默认关则二期数据攒不起来）；调用时现读 env。"""
    return _env_flag("AGENT_LEDGER_CAPTURE_CONTEXT", True)


def batch_size() -> int:
    return _env_int("AGENT_LEDGER_BATCH_SIZE", 50)


def flush_interval_ms() -> int:
    return _env_int("AGENT_LEDGER_FLUSH_MS", 500)


# run_id 生命周期
def new_run_id() -> str:
    """生成一次请求的 `run_id`（三表关联键）。"""
    return uuid.uuid4().hex


def set_run_id(run_id: str) -> Any:
    """把 `run_id` 绑到当前上下文，返回 reset token（调用方在 finally 里 `reset_run_id(token)`）；ContextVar 是跨请求透传的唯一可靠方式。"""
    return _run_id.set(run_id)


def reset_run_id(token: Any) -> None:
    try:
        _run_id.reset(token)
    except Exception:  # noqa: BLE001 - token 用错不该让主链路挂掉
        pass


def get_run_id() -> str | None:
    return _run_id.get()


# 入队（内存操作，永不抛）
def _pool_ready() -> bool:
    """连接池是否已建立。**刻意不主动建连**：没池就是"不在应用里跑"（如单测）⇒ 直接 no-op。"""
    try:
        mysql_mod.get_mysql_pool()
        return True
    except Exception:  # noqa: BLE001 - 无池 / 池未初始化都走这条
        return False


def _enqueue(table: str, row: dict[str, Any]) -> None:
    """入队一条事件；未启用 / 无池 ⇒ 静默丢弃（账本是旁路，不该有副作用）。"""
    if not ledger_enabled() or not row.get("run_id"):
        return
    if not _pool_ready():
        return
    with _pending_lock:
        _pending.append((table, row))
    _ensure_flusher()


def _ensure_flusher() -> None:
    """确保后台刷写任务在跑；无运行中的事件循环时（同步上下文）什么也不做，
    等下一次异步调用点（如 `flush_pending`）来接管。"""
    global _flush_task
    if _flush_task is not None and not _flush_task.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _flush_task = loop.create_task(_flush_forever())


def _clip(text: Any) -> str:
    s = "" if text is None else str(text)
    return s[:ERROR_MAX]


def _json(value: Any) -> str | None:
    """把结构化小对象序列化成 JSON 字符串（写进 `sub_intents` / `card_digest`）。
    空值 → None（列可空，别塞 "[]"/"{}" 这种假非空）。"""
    if value in (None, "", [], {}, ()):
        return None
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None


def _now() -> datetime:
    """本地墙钟时间（`sql/README.md` §7.2：这几张表的时间列**由应用写**，不用 DB 默认值）。"""
    return datetime.now()


# 对外埋点
def start_run(
    run_id: str,
    thread_id: str = "",
    user_id: str = "",
    mode: str = "",
    trace_id: str = "",
    query_text: str = "",
) -> None:
    """写 run 头（status=running，起止时间落 started_at）。`query_text` 为本次用户 query 原文（明文，与 agent_message 同口径），受 AGENT_LEDGER_CAPTURE_CONTEXT 控制。"""
    _enqueue(TABLE_RUNS, {
        "row_op": "insert",
        "run_id": run_id,
        "thread_id": str(thread_id or "")[:64],
        "user_id": str(user_id or "")[:64],
        "mode": str(mode or "")[:32],
        "trace_id": str(trace_id or "")[:64],
        "status": "running",
        "started_at": _now(),
        "model_call_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "query_text": (str(query_text or "")[:QUERY_MAX] if capture_enabled() else ""),
    })


def record_route(
    run_id: str,
    scene: str = "",
    sub_intents: Any = None,
    depends_on: str | None = None,
) -> None:
    """写意图路由结果（specs/0017 T2）：scene/sub_intents/depends_on。由 nodes.entry_router 在产出 plan 后调用（UPDATE 已有 run 行），旁路失败只 warning。"""
    if not capture_enabled():
        return
    _enqueue(TABLE_RUNS, {
        "row_op": "route",
        "run_id": run_id,
        "scene": str(scene or "")[:32],
        "sub_intents": _json(list(sub_intents or [])),
        "depends_on": (str(depends_on)[:64] if depends_on else None),
    })


def record_card_digest(
    run_id: str,
    card_digest: Any = None,
    answer_digest: str = "",
) -> None:
    """写最终产物摘要（specs/0017 T2）：卡片必填字段（JSON）+ 答复摘要。由 nodes.summarize_answer 在卡片抽完后调用，旁路失败只 warning。"""
    if not capture_enabled():
        return
    _enqueue(TABLE_RUNS, {
        "row_op": "cards",
        "run_id": run_id,
        "card_digest": _json(card_digest),
        "answer_digest": _clip(answer_digest) if answer_digest else None,
    })


def finish_run(
    run_id: str,
    status: str = "success",
    duration_ms: int | None = None,
    model_call_count: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    error: str | None = None,
) -> None:
    """收尾：把耗时 / 模型调用数 / 真实 token 汇总 **UPDATE** 回 run 头。
    走 UPDATE 而不是再插一行，是为了让"一次请求 = 一行"这个口径成立（查询接口据此取数）。"""
    _enqueue(TABLE_RUNS, {
        "row_op": "finish",
        "run_id": run_id,
        "status": status,
        "finished_at": _now(),
        "duration_ms": duration_ms,
        "model_call_count": model_call_count,
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "error": _clip(error) if error else None,
    })


def record_llm_call(
    run_id: str,
    node: str = "",
    model: str = "",
    started_at: datetime | None = None,
    duration_ms: int | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    status: str = "success",
    error: str | None = None,
) -> None:
    """一次模型调用 1 行。由 LLM 回调（`app/agents/graph/ledger_callback.py`）统一接入。"""
    _enqueue(TABLE_LLM_CALLS, {
        "run_id": run_id,
        "node": str(node or "")[:64],
        "model": str(model or "")[:64],
        "started_at": started_at or _now(),
        "duration_ms": duration_ms,
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "status": status,
        "error": _clip(error) if error else None,
    })


def record_tool_call(
    run_id: str,
    tool_name: str,
    node: str = "",
    target_shop_id: int | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    duration_ms: int | None = None,
    status: str = "success",
    error: str | None = None,
) -> None:
    """一次工具 / worker 调用 1 行。**刻意不去重** —— "同名工具出现几次"正是要观测的维度。"""
    _enqueue(TABLE_TOOL_CALLS, {
        "run_id": run_id,
        "node": str(node or "")[:64],
        "tool_name": str(tool_name or "")[:64],
        "target_shop_id": target_shop_id,
        "started_at": started_at or _now(),
        "finished_at": finished_at or _now(),
        "duration_ms": duration_ms,
        "status": status,
        "error": _clip(error) if error else None,
    })


# 落库（后台批量；失败只降级）
_INSERT_SQL = {
    TABLE_RUNS: (
        "INSERT INTO agent_runs (run_id, thread_id, user_id, mode, trace_id, status, "
        "started_at, model_call_count, prompt_tokens, completion_tokens, query_text) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    ),
    TABLE_LLM_CALLS: (
        "INSERT INTO agent_llm_calls (run_id, node, model, started_at, duration_ms, "
        "prompt_tokens, completion_tokens, status, error) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    ),
    TABLE_TOOL_CALLS: (
        "INSERT INTO agent_tool_calls (run_id, node, tool_name, target_shop_id, "
        "started_at, finished_at, duration_ms, status, error) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    ),
}
_FINISH_RUN_SQL = (
    "UPDATE agent_runs SET status=%s, finished_at=%s, duration_ms=%s, "
    "model_call_count=%s, prompt_tokens=%s, completion_tokens=%s, error=%s "
    "WHERE run_id=%s"
)
#: 意图路由结果（`specs/0017` T2）
_ROUTE_RUN_SQL = (
    "UPDATE agent_runs SET scene=%s, sub_intents=%s, depends_on=%s WHERE run_id=%s"
)
#: 最终产物摘要（`specs/0017` T2）
_CARDS_RUN_SQL = (
    "UPDATE agent_runs SET card_digest=%s, answer_digest=%s WHERE run_id=%s"
)
#: `row_op` → 专用 UPDATE 语句（其余走 INSERT）。**唯一真相源**：`flush_pending` 引用它。
_RUN_ROW_OP_SQL = {"finish": _FINISH_RUN_SQL, "route": _ROUTE_RUN_SQL, "cards": _CARDS_RUN_SQL}


def _drain() -> list[tuple[str, dict[str, Any]]]:
    """取出当前待落库的事件（**保持顺序**：事件顺序本身就是排障信息）。"""
    with _pending_lock:
        if not _pending:
            return []
        # 一次最多取 batch_size 条，避免长尾请求一次性砸出上千行
        items = [_pending.popleft() for _ in range(min(len(_pending), batch_size()))]
    return items


def _row_values(table: str, row: dict[str, Any]) -> tuple | None:
    row_op = row.pop("row_op", "insert")
    if table == TABLE_RUNS and row_op == "finish":
        return (
            row.get("status"), row.get("finished_at"), row.get("duration_ms"),
            row.get("model_call_count"), row.get("prompt_tokens"),
            row.get("completion_tokens"), row.get("error"), row.get("run_id"),
        )
    if table == TABLE_RUNS and row_op == "route":
        return (row.get("scene"), row.get("sub_intents"), row.get("depends_on"), row.get("run_id"))
    if table == TABLE_RUNS and row_op == "cards":
        return (row.get("card_digest"), row.get("answer_digest"), row.get("run_id"))
    cols = {
        TABLE_RUNS: ("run_id", "thread_id", "user_id", "mode", "trace_id", "status",
                     "started_at", "model_call_count", "prompt_tokens", "completion_tokens",
                     "query_text"),
        TABLE_LLM_CALLS: ("run_id", "node", "model", "started_at", "duration_ms",
                          "prompt_tokens", "completion_tokens", "status", "error"),
        TABLE_TOOL_CALLS: ("run_id", "node", "tool_name", "target_shop_id", "started_at",
                           "finished_at", "duration_ms", "status", "error"),
    }[table]
    return tuple(row.get(c) for c in cols)


async def flush_pending() -> int:
    """把队列事件写库一次（失败只记日志），返回写成功条数；generate_sse 收尾调它以便本轮立即可查。"""
    if not ledger_enabled() or not _pool_ready():
        return 0
    items = _drain()
    if not items:
        return 0

    written = 0
    try:
        async with mysql_mod.acquire_cursor() as cur:
            for table, row in items:
                op = row.get("row_op") if table == TABLE_RUNS else None
                sql = _RUN_ROW_OP_SQL.get(op) if op else None
                if sql is not None:
                    await cur.execute(sql, _row_values(table, row))
                else:
                    await cur.execute(_INSERT_SQL[table], _row_values(table, row))
                written += 1
    except Exception as exc:  # noqa: BLE001 - 账本写失败**绝不上抛**
        # 丢事件要可见：只写 warning，不回滚队列（重试可能再次失败并堆积）。
        logger.warning("账本落库失败（丢弃 %s 条事件，不影响本轮答复）: %s", len(items), exc)
        return 0
    return written


async def _flush_forever() -> None:
    """后台刷写：每 `AGENT_LEDGER_FLUSH_MS` 毫秒把队列排空一次。"""
    try:
        while True:
            await asyncio.sleep(max(0.05, flush_interval_ms() / 1000.0))
            await flush_pending()
    except asyncio.CancelledError:
        # 关停前把剩下的写掉（best-effort）
        try:
            await flush_pending()
        except Exception:  # noqa: BLE001
            pass
        raise


async def stop() -> None:
    """停掉后台任务并做最后一次刷写（lifespan 收尾 / 测试用）。"""
    global _flush_task
    task, _flush_task = _flush_task, None
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    await flush_pending()


# 查询（`GET /api/v1/ledger/runs/{run_id}`）
async def fetch_run(run_id: str) -> dict[str, Any] | None:
    """取一次请求的 run 头 + 全部 llm / tool 事件（**均按时间升序**）。
    找不到返回 `None`（由路由转 404）。查询失败**上抛** —— 这是查询接口，不是旁路埋点。"""
    async with mysql_mod.acquire_cursor() as cur:
        await cur.execute(
            "SELECT run_id, thread_id, user_id, mode, trace_id, status, started_at, "
            "finished_at, duration_ms, model_call_count, prompt_tokens, "
            "completion_tokens, error FROM agent_runs WHERE run_id = %s",
            [run_id],
        )
        run = await cur.fetchone()
        if not run:
            return None

        await cur.execute(
            "SELECT node, model, started_at, duration_ms, prompt_tokens, "
            "completion_tokens, status, error FROM agent_llm_calls "
            "WHERE run_id = %s ORDER BY started_at ASC, id ASC",
            [run_id],
        )
        llm_calls = await cur.fetchall()

        await cur.execute(
            "SELECT node, tool_name, target_shop_id, started_at, finished_at, "
            "duration_ms, status, error FROM agent_tool_calls "
            "WHERE run_id = %s ORDER BY started_at ASC, id ASC",
            [run_id],
        )
        tool_calls = await cur.fetchall()

    return {
        "run": _jsonable(run),
        "llm_calls": [_jsonable(r) for r in llm_calls],
        "tool_calls": [_jsonable(r) for r in tool_calls],
    }


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    """把 `datetime` 转成 `isoformat()`（前端/JSON 不能直接吃 datetime）。"""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, datetime):
            out[key] = value.isoformat()
        elif isinstance(value, (bytes, bytearray)):
            out[key] = value.decode("utf-8", "replace")
        else:
            out[key] = value
    return out


def pending_count() -> int:
    """队列里待落库的事件数（测试与自检用）。"""
    with _pending_lock:
        return len(_pending)


def reset_for_tests() -> None:
    """清空队列与后台任务（**仅供单测**）：避免用例之间互相污染。"""
    global _flush_task
    with _pending_lock:
        _pending.clear()
    _flush_task = None


__all__ = [
    "LEDGER_TABLES", "TABLE_RUNS", "TABLE_LLM_CALLS", "TABLE_TOOL_CALLS",
    "new_run_id", "set_run_id", "reset_run_id", "get_run_id",
    "start_run", "finish_run", "record_llm_call", "record_tool_call",
    "record_route", "record_card_digest",
    "flush_pending", "stop", "fetch_run", "ledger_enabled", "capture_enabled",
    "pending_count", "reset_for_tests",
]
