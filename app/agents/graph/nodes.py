"""LangGraph 节点实现。

节点按职责分两类：LLM 节点（entry_router、daily_chat、summarize）与 Data 节点（grounding、nearby_worker、shop_worker —— 只取数、不生成自然语言，省 token 且稳定）。"""

import asyncio
import json
import math
import os
import re
import time
from datetime import date, datetime, timedelta
from typing import Any, Literal

import pybreaker
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.graph import END
from langgraph.types import Command, Send, interrupt
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents import ledger
from app.agents.graph import degrade
from app.agents.graph.ledger_callback import fingerprint_of, usage_of
from app.agents.graph.llm import (
    get_light_model,
    get_main_model,
    get_planner_model,
    get_slot_model,
    structured,
)
from app.agents.graph.plan_slots import (
    REQUIRED_SLOTS,
    SCENE_TYPES_KEY,
    aligned_scene_types,
    build_clarification,
    is_valid as slot_is_valid,
    max_clarify_retry,
    missing_slots,
    sanitize_slots,
)
from app.agents.graph.prompts import (
    CAPABILITY_ANSWER,
    CATEGORY_CLASSIFY_SYSTEM,
    ENTRY_SYSTEM,
    HISTORY_INJECTED_SECTIONS,
    HISTORY_SECTIONS,
    HISTORY_SECTION_DEAD_END,
    HISTORY_SECTION_GOAL,
    HISTORY_SECTION_LAST,
    HISTORY_SECTION_RECOMMENDED,
    HISTORY_SECTION_TODO,
    ITINERARY_SYSTEM,
    SLOT_SYSTEM,
    STEP_REMIND_FULL,
    STEP_REMIND_SPARSE,
    SUBCATEGORY_FILTER_SYSTEM,
    daily_system,
    daily_time_prefix,
    history_summary_system,
    plan_summary_system,
    summary_system,
)
from app.agents.graph.state import CLEAR, MAX_TARGET_INDICES, HmdpState, RoutePlan, plan_of
from app.agents.graph.token_budget import (
    estimate_messages_tokens,
    estimate_tokens,
    recent_messages,
    should_compress,
    split_index,
    token_budget,
)
from app.agents.hmdp_metrics import (
    record_history_tokens,
    record_model_call,
    record_node,
    record_usage_from_message,
)
from app.agents.hmdp_tools import (
    BOOKING_CANCEL_FAILED,
    BOOKING_FAILED,
    BOOKING_NEEDS_LOGIN,
    REACT_TOOLS,
    SHOP_NAME_NOT_FOUND,
    SHOP_TYPE_NOT_FOUND,
    SHOPS_NEARBY_EMPTY,
    cancel_booking,
    create_booking,
    fetch_rag_chunks,
    fetch_shop_blogs,
    fetch_shop_detail,
    fetch_shop_types,
    fetch_shop_vouchers,
    fetch_shops_by_name,
    fetch_shops_nearby,
    prepare_order_card,
    search_blogs_by_key,
)
from app.common.logger import logger
from app.models.schemas import RagAnswer

load_dotenv()

# ---- 可调参数 -------------------------------------------------------------
MAX_SHOPS = int(os.getenv("AGENT_MAX_SHOPS", "3"))                # 单次最多处理的店铺数
NEARBY_TOP = int(os.getenv("AGENT_NEARBY_TOP", "5"))              # 附近找店返回条数
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
CONTEXT_BLOGS = int(os.getenv("AGENT_CONTEXT_BLOGS", "3"))        # 每家店进上下文的博客数
MAX_SOURCE = int(os.getenv("AGENT_MAX_SOURCE", "3"))
#: 行程摘要进 `messages` 的**硬上限**（字符）——结构化明细走载荷，正文只留摘要（`specs/0012` §11.4）。
PLAN_ANSWER_MAX_CHARS = int(os.getenv("AGENT_PLAN_ANSWER_MAX_CHARS", "400"))
#: 行程最多几站（`specs/0012` R2：行程越长上下文与延迟越贵，必须截断）。
ITINERARY_MAX_STOPS = int(os.getenv("AGENT_ITINERARY_MAX_STOPS", "6"))
#: 每个场景取几家常住候选（够编排用即可，不追求全局最优）。
PLAN_CANDIDATES_PER_SCENE = int(os.getenv("AGENT_PLAN_CANDIDATES_PER_SCENE", "3"))
#: 计划轨最多同时安排几个场景（防止"逛吃看玩买"一次全上）。
PLAN_MAX_SCENES = int(os.getenv("AGENT_PLAN_MAX_SCENES", "3"))
#: 单站默认停留时长（分钟）——`start`/`end` 由它推导，确定性可测。
PLAN_STOP_MINUTES = 90
#: 步行速度（米/分钟），用于把**直线距离**折成 `duration_min`。
PLAN_WALK_M_PER_MIN = 80

# ---- 汇总节点名（唯一被允许逐 token 流式的节点）----
#: 路由函数返回它、`workflow.py` 用它注册节点、`hmdp_agent.STREAM_NODES` 用它做白名单 —— 唯一的真相源就在这一行，三处必须一致，改这里就够了（别在别处硬编码字面量）。
#: 为什么单独拆一个节点：见 `summarize_answer` 的 docstring（structured 输出会屏蔽消息流）。
SUMMARY_NODE = "summarize_answer"

# ---- 扇出资源上限（plans/0005 T4 / specs/0005 §4 验收 2/3/4）----
# 没有上限时，一句"比较这五家的详情、优惠和攻略"会瞬时打出 5×3=15 个 worker 把 DB/ES 打满。
#: 口径 = 店铺数（`plans/0009.2` 把 Send 粒度从"任务"改成"店铺"后，一项 = 一个 Send）；沿用旧口径（同一家店在每个任务里各数一次）上限会被放大成几倍 ⇒ 该截的没截（静默多发）；正常请求是 3 店 ≤ 8，任何口径都不该误伤。
MAX_FANOUT = int(os.getenv("AGENT_MAX_FANOUT", "8"))
#: 单个 worker 的取数超时。语义是**降级**而非失败：超时记 failed，本轮照常 summarize。
WORKER_TIMEOUT = float(os.getenv("AGENT_WORKER_TIMEOUT", "12"))
#: `rag`（细节检索）这一段自己的预算：从 `WORKER_TIMEOUT` 派生（不新增 `.env` 变量）。
#: `rag` 带着可选重排（Cross-Encoder），是唯一会"秒级阻塞"的可选增强；若与 `detail` 共享 `WORKER_TIMEOUT`，rag 一慢就把整包判失败、连 55ms 就到手的 `detail` 一起丢 ⇒ 答复「人均暂未获取」；有自己的预算后，`rag` 超时只写 `chunks_error`，`detail` / `blogs` 照常（见 `bug/BUG-20260928-rag-rerank-network-stall.md` §3 / `specs/0010` §3）。
RAG_TIMEOUT = max(3.0, WORKER_TIMEOUT * 0.6)
#: `RAG_TIMEOUT` 被截断时写进 `chunks_error` 的文案（真机 grep 用，别改字面量）。
#: **定义在 `degrade.py`**（`specs/0015` T3 的降级文案单点真源），这里只做引用 ——
#:    新增降级路径必须去那里取，不许就地写字面量。
RAG_TIMEOUT_MSG = degrade.RAG_TIMEOUT_MSG
#: 同一类 worker 连续失败达此阈值后，后续同类**直接短路**，不再发起实际调用。
FANOUT_FAIL_THRESHOLD = int(os.getenv("AGENT_FANOUT_FAIL_THRESHOLD", "3"))

# ---- ReAct 补跳（plans/0005 T5 / specs/0005 §5.5）----------------------------
#: 补跳的最大步数。到上限**强制收敛**到 `summarize`，绝不无限循环。
REACT_MAX_STEPS = int(os.getenv("AGENT_REACT_MAX_STEPS", "6"))
#: 单步最多并行几个工具（2026-09-28 迁移到 `bind_tools` 后才有"一批多个"这回事）；`REACT_MAX_STEPS` 管轮数、这个管每轮的工具数，两者相乘才是取数调用的上界。
#: 不能放任模型一次点满：`shop_rag` 会带重排（本机 CPU 实测 2.3–6.6s/次），一次点 5 个就把 `WORKER_TIMEOUT` 那点预算打穿（环内没有独立的取数预算，各工具自管）；取 2 既够覆盖"同时补详情 + 补券"，又不至于一次性放空。
REACT_MAX_TOOLS_PER_STEP = int(os.getenv("AGENT_REACT_MAX_TOOLS_PER_STEP", "2"))

#: 每几步注入一次"计划约束"提醒（`specs/0014` T2）。**0 / 留空 = 关闭**（回落现状）。
#: 只在**环内**计数（见 `react_loop` 的局部变量），不进 State —— 跨轮串味会让计数莫名其妙。
def step_remind_every() -> int:
    try:
        return max(0, int(os.getenv("AGENT_REACT_STEP_REMIND_EVERY", "0")))
    except (TypeError, ValueError):
        return 0


def _react_step_reminder(step: int) -> list[Any]:
    """第 step 步（从 1 起）要注入的提醒消息，不需要则空列表。走 user 侧不进 system（specs/0014 T2）：system 前缀随步数变会让 prompt cache 整段失效；首次与每 AGENT_REACT_STEP_REMIND_EVERY 步用 full，其余 sparse，开关 0 则不注入。"""
    every = step_remind_every()
    if every <= 0 or step <= 0:
        return []
    if step == 1 or step % every == 0:
        return [HumanMessage(content=STEP_REMIND_FULL.format(step=step))]
    return [HumanMessage(content=STEP_REMIND_SPARSE)]


#: 扇出 / 环内并行的**并发许可门**（`specs/0014` T5）：与 `MAX_FANOUT` / `MAX_SHOPS` 卡**数量**不同，本门卡**瞬时并发**，突发流量下不打满下游 Java 服务。
#: 默认取 `MAX_FANOUT`：并发上限 = 数量上限 ⇒ **门恒不生效**（安全默认值，行为不变）。
def fanout_concurrency() -> int:
    try:
        return max(1, int(os.getenv("AGENT_FANOUT_CONCURRENCY", str(MAX_FANOUT))))
    except (TypeError, ValueError):
        return MAX_FANOUT


#: 并发许可门。**惰性创建**：`asyncio.Semaphore()` 在**没有运行中的事件循环**时创建会绑定到
#: 错的循环（Python 3.10+ 已不推荐在循环外创建）⇒ 第一次真正用到时才建。
_FANOUT_GATE: asyncio.Semaphore | None = None


def reset_fanout_gate() -> None:
    """清空并发门（**仅供单测**）：换 `AGENT_FANOUT_CONCURRENCY` 后必须重建。"""
    global _FANOUT_GATE
    _FANOUT_GATE = None


def _fanout_gate() -> asyncio.Semaphore:
    global _FANOUT_GATE
    if _FANOUT_GATE is None:
        _FANOUT_GATE = asyncio.Semaphore(fanout_concurrency())
    return _FANOUT_GATE


#: 扇出熔断按 task 独立实例（specs/0022/plans/0038，pybreaker）。旧实现"连续失败计数+布尔短路"达阈值后永久短路、只能重启；pybreaker 带冷却窗口+半开探测（Open 经 reset_timeout 放行一次，成功关、失败重开）。
#: worker 失败是布尔结果（降级放行不抛异常）而 pybreaker 按异常计数，故用哨兵异常 _FanoutWorkerError 把失败翻译成异常，统一经 breaker.call 记账。
_BREAKERS: dict[str, "pybreaker.CircuitBreaker"] = {}
#: task → 最近一次"熔断打开"的时刻（`time.monotonic()`）。pybreaker **读态不会自动转半开**，
#: 故冷却判定在这里做（期满即放行）；真正的状态推进交给 `_record_worker_result` 里的 `breaker.call`。
_BREAKER_OPENED_AT: dict[str, float] = {}


class _FanoutWorkerError(Exception):
    """哨兵异常：把"worker 业务失败（布尔）"翻译成 pybreaker 能计入的**异常**。"""


def breaker_cooldown() -> float:
    """熔断冷却窗口（秒，`AGENT_FANOUT_BREAKER_COOLDOWN`，默认 30）。非法值回落默认，**调用时读**。"""
    try:
        value = float(os.getenv("AGENT_FANOUT_BREAKER_COOLDOWN", "30"))
    except (TypeError, ValueError):
        return 30.0
    return value if value > 0 else 30.0


def _breaker_for(task: str) -> "pybreaker.CircuitBreaker":
    """按 task 惰性创建并缓存独立的熔断器（互不影响）。"""
    breaker = _BREAKERS.get(task)
    if breaker is None:
        breaker = pybreaker.CircuitBreaker(
            fail_max=FANOUT_FAIL_THRESHOLD,
            reset_timeout=breaker_cooldown(),
            name=f"fanout:{task}",
        )
        _BREAKERS[task] = breaker
    return breaker


def reset_fail_streaks() -> None:
    """清空全部熔断器与冷却计时（供测试与每轮请求隔离使用）。"""
    _BREAKERS.clear()
    _BREAKER_OPENED_AT.clear()


def _fail_count_of(task: str) -> int:
    """该 task 当前累计失败数（供测试断言；等价于旧 `_FAIL_STREAK.get(task, 0)`）。"""
    return int(_breaker_for(task).fail_counter)


def _breaker_open(task: str) -> bool:
    """该类 worker 是否应短路（specs/0022）：仅"Open 且仍在冷却期"为真；冷却到期返回 False 放行一次探测，状态推进交给后续 _record_worker_result 的 breaker.call（Open+超时后当半开探测：成功关、失败重开）。"""
    breaker = _breaker_for(task)
    if breaker.current_state != pybreaker.STATE_OPEN:
        return False
    opened_at = _BREAKER_OPENED_AT.get(task)
    if opened_at is None:
        _BREAKER_OPENED_AT[task] = time.monotonic()
        return True
    return (time.monotonic() - opened_at) < breaker_cooldown()


def _record_worker_result(task: str, ok: bool) -> None:
    """记录一次 worker 结果：成功关闭/清零、失败累加、达阈值即开；半开探测成功关、失败重开。pybreaker 把半开探测失败也包成 CircuitBreakerError（与 Open 短路未执行同型），故用 ran 标记区分：只有真正执行过才刷新冷却起点，被短路的不动。"""
    breaker = _breaker_for(task)
    ran = {"v": False}

    def _attempt() -> bool:
        ran["v"] = True
        if not ok:
            raise _FanoutWorkerError()
        return True

    try:
        breaker.call(_attempt)
    except (_FanoutWorkerError, pybreaker.CircuitBreakerError):
        pass
    if not ran["v"]:
        # Open 且被短路（没执行）⇒ **不改冷却起点**，否则冷却会被"每次被短路"反复重置、永不恢复。
        return
    # 真正执行过 ⇒ 冷却起点跟随当前状态：仍 Open（本轮失败触发的）重新计时；否则清掉。
    if breaker.current_state == pybreaker.STATE_OPEN:
        _BREAKER_OPENED_AT[task] = time.monotonic()
    else:
        _BREAKER_OPENED_AT.pop(task, None)


def _deadline_after(seconds: float) -> float:
    """从此刻起 `seconds` 秒后的**绝对截止时刻**（`time.monotonic()` 时钟，不受系统时间调整影响）。"""
    return time.monotonic() + seconds


def _remaining(deadline: float | None, cap: float) -> float:
    """`deadline` 还剩多少秒，且不超过 `cap`；`deadline=None` ⇒ `cap`；已过期 ⇒ `0.0`。
    这是"一个 deadline 贯穿两层"的关键：内层（`rag` 的重排预算）拿到的是外层剩下的时间，而不是重新起算的固定预算 —— 否则外层照样会在 `d + r` 处超时（反例 `specs/0010` §3：`d=8s, r=6s, 上限 12s` ⇒ 内层不截断、外层整包被丢）。"""
    if deadline is None:
        return cap
    return max(0.0, min(cap, deadline - time.monotonic()))


async def _run_with_timeout(
    coro: Any, task: str, deadline: float | None = None
) -> tuple[Any, bool]:
    """跑一个 worker 协程并施加预算（`deadline` 的剩余额度，或默认 `WORKER_TIMEOUT`）。
    返回 `(结果, ok)`：正常完成 → `(result, True)`；无预算/超时/异常 → `(None, False)`，不向调用方抛（一个慢 worker 不能拖垮整轮问答）；超时的语义是"降级"，调用方拿到 `ok=False` 后照常进 `summarize`。
    超时与取数失败分成两支日志（都带预算秒数）—— 只打 `str(asyncio.TimeoutError)` 是空串 ⇒ 无法归因；`except BaseException` 改成 `except Exception` 是因为 `asyncio.CancelledError` 必须上抛，否则客户端断连的取消信号会被吞掉。"""
    budget = _remaining(deadline, WORKER_TIMEOUT)
    if budget <= 0:
        logger.warning(
            "worker(%s) 未获预算（上游已用满 %.1fs），直接降级", task, WORKER_TIMEOUT
        )
        return None, False
    try:
        # 并发许可门（`specs/0014` T5）：MAX_FANOUT 卡的是"一共发几个"，这里卡的是
        #    "同时在飞几个"。放在**发起调用前**，且默认上限 = MAX_FANOUT ⇒ 默认恒不生效。
        async with _fanout_gate():
            return await asyncio.wait_for(coro, timeout=budget), True
    except asyncio.TimeoutError:
        logger.warning(
            "worker(%s) 超时（%.1fs 预算；降级不属失败）: 该块记 *_error，其余块照常",
            task,
            budget,
        )
        return None, False
    except Exception as exc:
        logger.warning("worker(%s) 取数失败（%.1fs 预算）: %s", task, budget, exc)
        return None, False


def _split_list_or_error(value: Any) -> tuple[list[Any], str]:
    """把 `list | str` 的取数结果拆成 `(列表, 错误文本)`。
    `fetch_shop_vouchers` / `fetch_rag_chunks` 的签名是 `-> list[dict] | str`：取数失败或业务上为空（如"这家店没有券"）时返回的是字符串而不是空列表，直接把它塞进 `worker_outputs[].vouchers`，下游 `render_context` 会对元素调 `.get('title')` 而抛 `AttributeError: 'str' object has no attribute 'get'`，整轮问答崩掉。
    与 `_one_shop_bundle` 里 `blogs` / `chunks` 拆成 `*_error` 字段是同一套约定，抽出来避免每条新链路各自漏一次。"""
    if isinstance(value, str):
        return [], value
    if isinstance(value, list):
        # 纵深防御：列表里混进 str 时也过滤掉，不让它流到渲染层
        return [v for v in value if not isinstance(v, str)], ""
    if value is None:
        return [], ""
    return [value], ""


_ORDINAL = {"一": 0, "二": 1, "两": 1, "三": 2, "四": 3, "五": 4, "1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
# 单数/复数指代都算"用了指代"。注意要能匹配「这三家」——原来只写了「这两家」，
# 导致「这三家对比如何」连指代分支都进不去，直接跑到兜底追问。
# 「最后N家 / 倒数第N家」也要算 —— 否则「那最后一家怎么样」连指代分支都进不来
# （`最后` 既不在 `这/那` 前缀里，`一家` 又因 `(?<!第)` 之外的原因不被 `第X家` 覆盖）。
_ANAPHORA = re.compile(
    r"这家|那家|此店|它|它们|这[一二两三四五1-5][家个]|那[一二两三四五1-5][家个]|这几家|"
    r"第[一二两三四五1-5][家个]|倒数第[一二两三四五1-5][家个]|"
    r"最后[这那]?[一二两三四五1-5][家个]|最后[这那]家|"
    r"前[一二两三四五1-5][家个]|"
    r"其中|哪个更好|哪家更"
)
# 复数指代：这类说法指向多家店，不能只取焦点里的第一家。
#
# 🔴 两个"别抢单数"的排除（都踩过）：
# 1. **「第X家」/「倒数第X家」**是单数 ⇒ `(?<!第)` 负向后顾排除（否则「第二家」的「二家」
#    会被当复数，走 `focus[:2]` 把单数问题解析成两家）。
# 2. 🔴 **「最后一家」/「最后那家」是单数**，但 `最` 不是 `第` ⇒ `(?<!第)` 拦不住
#    （旧版这里 `那最后一家` 命中复数分支 → `focus[:1]` → **取到第一家**，
#    而 `_ordinal_index` 明明会返回 `-1`）。⇒ 再补 `(?<!最后)` 负向后顾。
#    ⚠️ 只排「最后**一**家」（`(?<!最后)` 只管前两个字），「最后两家」要留给复数分支。
_PLURAL_ANAPHORA = re.compile(
    r"(?<!第)(?<!最后)[这那]?[一二两三四五1-5][家个]|"
    r"最后[这那]?[二两三四五2-5][家个]|"
    r"这[几些]家|这些|它们|都|分别|各自|对比|比较|哪个更好|哪家更"
)


def _plural_count(query: str) -> int | None:
    """从「这三家 / 那两家 / 两家 / 最后两家」里抽出用户指代的店铺数量（不是索引）：`_ORDINAL` 存的是索引（三→2），所以这里要 +1 才是家数。

    两个排除，与 `_PLURAL_ANAPHORA` 同源：
    - `(?<!第)`：不把「第二家」读成"2 家"；
    - `(?<!最后)`：不把「最后**一**家」读成"1 家"（那是单数指代，家数该由 `_ordinal_index` 处理）。
      ⚠️「最后两家」仍要能读到 2 —— `(?<!最后)` 只看前两字，`两家` 前是「最后」⇒ 也会被排掉，
      所以「最后N家」由下面第二条分支单独兜。
    """
    match = re.search(r"(?<!第)(?<!最后)[这那]?([一二两三四五1-5])[家个]", query)
    if match:
        idx = _ORDINAL.get(match.group(1))
        return None if idx is None else idx + 1
    # 「最后两家 / 最后三家」：`最后` 后的家数（排除「最后一家」，那是单数）
    match = re.search(r"最后[这那]?([二两三四五2-5])[家个]", query)
    if match:
        idx = _ORDINAL.get(match.group(1))
        return None if idx is None else idx + 1
    # 「前两家 / 前三家」：从头部数起（起点固定 0，`_ordinal_index` 对 `前N家` 返回 None）
    match = re.search(r"前([二两三四五2-5])[家个]", query)
    if match:
        idx = _ORDINAL.get(match.group(1))
        return None if idx is None else idx + 1
    return None


# ---- 通用小工具 -----------------------------------------------------------
def as_text(content: Any) -> str:
    """把模型返回内容统一成字符串（新版模型可能返回 list[dict]）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
            if item is not None
        )
    return str(content)


def _yuan(cents: Any) -> str:
    """把分（整数）渲染成元的两位小数字符串，供"给模型看的文本"使用。
    `tb_voucher.pay_value` / `actual_value` 的单位是分（bigint），而渲染层此前直接拼「元」⇒ 把 47.5 元的券说成 4750 元（放大 100 倍）；券金额是下单决策依据，而模型只是照抄这段文本。这是全仓唯一的券金额渲染点。
    只做渲染侧换算（券卡片 `specs/0006` 用同一批字段但保留分不动、由前端自行格式化，改它会破坏已上线契约）；禁浮点（`4750 / 100` 在二进制浮点下有表示误差，金额一律用整数除 + 余数拼字符串）；上游 `voucher/list` 返回字符串（`"4750"`）所以必须能吃 `str`；非数值（`None` / `""` / 非数字）返回 `"?"` 不臆造 0（「0 元」与「没拿到金额」是两件事，混淆会让模型编出"免费"）。"""
    try:
        n = int(str(cents).strip())
    except (TypeError, ValueError):
        return "?"
    sign = "-" if n < 0 else ""
    n = abs(n)
    return f"{sign}{n // 100}.{n % 100:02d}"


def _chat_messages_all(messages: list[Any]) -> list[Any]:
    """取出所有「有内容的人类/AI 消息」，过滤掉系统消息和空消息。"""
    return [
        m
        for m in (messages or [])
        if isinstance(m, (HumanMessage, AIMessage)) and as_text(getattr(m, "content", ""))
    ]


def _history_before_current(messages: list[Any], budget: int) -> list[Any]:
    """当前这条用户消息已经入 State，取它之前按 token 预算划出的历史作为上下文。
    与 `_recent_chat_messages` 的口径一致（都走 `token_budget.recent_messages`）：窗口大小由 token 决定、不再由「固定轮数」决定 —— 原先 `turns=2` / `turns=1` 是按条数截取，而条数 ≠ token（命中 RAG 的一轮就可能带进几千 token），固定条数既不设上限也不设下限。
    做法：先按角色/非空过滤，再丢掉「当前这条用户消息」（它已由调用方单独拼进 prompt），最后交给 `recent_messages` 取累计不超预算的最大连续后缀。"""
    picked = _chat_messages_all(messages)
    if picked and isinstance(picked[-1], HumanMessage):
        picked = picked[:-1]
    return recent_messages(picked, budget)


def _focus_hint(state: HmdpState) -> list[SystemMessage]:
    """把跨轮焦点店铺摘要成一条系统消息，供入口意图识别判定指代与依赖类问题。
    入口也需要焦点：`entry_router` 的 scene 判定要先于 `grounding` 发生，而「这三家怎么样 / 第二家呢」这类纯指代问题不含任何本地生活关键词，若入口看不到焦点就会把它判成 daily 闲聊，`grounding` 的指代消解根本没机会执行；焦点由 `grounding` / `nearby_worker` / `summarize` 累积写入（见 `merge_focus`）。
    只给店名是不够的：入口还负责抽取 `sub_intents` 等路由信息，而「最便宜那家 / 评分最高那家 / 哪家最近」要求模型先看到价格/评分/距离，才能意识到它依赖的是「三家已知事实的比较」，因此按 `render_context` 的口径把已知事实一并列出（缺字段如实写"暂未获取"，不编数字）；字段来源 `_to_focus_item`（`avg_price` / `score` / `distance`）。"""
    focus = [s for s in (state.get("focus_shops") or []) if isinstance(s, dict) and s.get("name")]
    if not focus:
        return []

    lines: list[str] = []
    for i, shop in enumerate(focus, 1):
        parts = [f"{i}. {shop.get('name')}（id={shop.get('id')}）"]
        distance = shop.get("distance")
        parts.append(f"距离{distance}m" if distance is not None else "距离暂未获取")
        score = shop.get("score")
        parts.append(
            f"评分{score / 10:.1f}" if isinstance(score, (int, float)) and score else "评分暂未获取"
        )
        avg = shop.get("avg_price")
        parts.append(f"人均{avg}元" if avg else "人均暂未获取")
        lines.append(" ".join(parts))

    return [
        SystemMessage(
            "本轮对话已涉及以下店铺（跨轮累积，按提及顺序编号；"
            "用户说「这家/第二家/这三家」时指的就是它们）：\n"
            + "\n".join(lines)
            + "\n已知事实（价格/评分/距离）可直接用于判断「最便宜那家」「评分最高那家」这类**比较型**问题；"
            "若用户用指代或就这些店铺继续追问，scene 必须判为 recommend，不要判成 daily。"
        )
    ]


def _recent_chat_messages(messages: list[Any], budget: int) -> list[Any]:
    """按 token 预算取出要注入模型的最近闲聊上下文（口径见 `token_budget.recent_messages`）。
    不再按条数是因为条数 ≠ token（几条带长正文的消息就能顶掉几十条闲聊，固定条数既不设上限、也不设下限）；预算与压缩共用 `AGENT_HISTORY_TOKEN_BUDGET`，压缩划出保留区，这里只注入保留区内的最近消息。"""
    return recent_messages(_chat_messages_all(messages), budget)


def _with_time_context(messages: list[Any]) -> list[Any]:
    """把"当前时间"作为 user 侧前缀注入最后一条用户消息（specs/0013 T6）。不放 system：system 前缀每分钟变会让 prompt cache 整段失效（命中最长公共前缀），时间只影响本轮，放 user 侧足够且 system 逐字节稳定（test_prompt_bytes_stable.py 看住）。
    只改喂给模型那份拷贝、不写回 state.messages（历史存原文，否则每轮叠时间戳、第 N 轮看到 N 个时间）。"""
    if not messages:
        return messages
    last = messages[-1]
    if not isinstance(last, HumanMessage):
        return messages
    out = list(messages)
    # 必须带上原 `id`：既有用例按 id 断言"注入的是哪几条历史"，
    #    丢 id 会让测试红，也会让下游按 id 的去重/删除逻辑对不上。
    out[-1] = HumanMessage(
        content=f"{daily_time_prefix()}\n{as_text(last.content)}",
        id=getattr(last, "id", None),
    )
    return out


# provider usage 锚点（specs/0013 T5）：压缩决策不再只看本地估算。锚点字段空值用于压缩生效后 reset（见 trim_history）。
_ANCHOR_RESET: dict = {
    "last_provider_context_tokens": None,
    "anchor_message_count": None,
    "prompt_fingerprint": "",
}


def anchor_patch(response: Any, messages: list[Any]) -> dict:
    """把本节点刚拿到的真实 usage 转成 State 锚点补丁（无 usage⇒{}）。由调用方自查、不经回调（specs/0013 §11.1）：曾挂 on_llm_end（同步）写 ContextVar，异步链路上同步回调进线程池⇒落在别的上下文、节点读不到，表现为"账本有 token、锚点 None"（见 bug/BUG-20261009）。
    messages 必须传当时 state.messages 而非喂模型那份拷贝：anchor_message_count 供下一轮 trim_history 做 messages[count:]，须同源；喂模型那份挂了临时 system 又被 recent_messages 裁过，拿它计数会静默错位。必须进 State（跨请求存活）；逻辑放 nodes.py 因 token_budget.py 须保持纯标准库。用法：return 里 **anchor_patch(response, state.get("messages") or [])。"""
    if not _env_flag("AGENT_USAGE_ANCHOR_ENABLED", True):
        return {}
    prompt_tokens, _completion = usage_of(response)
    if not prompt_tokens:
        return {}
    history = list(messages or [])
    return {
        "last_provider_context_tokens": int(prompt_tokens),
        "anchor_message_count": len(history),
        "prompt_fingerprint": fingerprint_of(history),
    }


def _anchor_estimate(state: HmdpState, messages: list[Any]) -> int | None:
    """用真实 usage 锚点估当前上下文 token，锚点不可用返回 None。口径=锚点 token + 锚点后新增消息的本地估算（锚点+增量）：锚点取上次模型调用的 prompt_tokens，其后新消息无真实 usage 只能估算后加。两种失效都必须回退纯本地估算（宁用估算不错用）：① anchor_message_count>len(messages)（消息被压缩删过，上下文不完整）；② 指纹不匹配（头部形状变，老消息被动过）。"""
    tokens = state.get("last_provider_context_tokens")
    count = state.get("anchor_message_count")
    if not tokens or not count:
        return None
    if int(count) > len(messages or []):
        return None
    fingerprint = str(state.get("prompt_fingerprint") or "")
    if fingerprint and fingerprint != fingerprint_of(messages):
        return None
    return int(tokens) + estimate_messages_tokens(messages[int(count):])


# ===========================================================================
# 第一层：历史裁剪 + 入口意图识别
# ===========================================================================
async def trim_history(state: HmdpState) -> dict:
    """控制 messages 的 token 占用：按 token 预算从最新往最老切出保留区，切点之前的候选区用便宜模型压成一段 running summary 存入 State，再 RemoveMessage 删掉原文 —— 既避免 token 无限膨胀，又尽量不丢上文语义。
    触发条件只看 token 预算（`AGENT_HISTORY_TOKEN_BUDGET`，默认 6000；见 token_budget.py），不看条数、也不再保留「最近 N 轮永不压缩」的保护区（最近的消息同样参与计算、同样可能被压缩）；切分规则（user 边界对齐、至少保留最后一条）全部由 token_budget 决定。
    删除原文与摘要生成必须是原子决策（只有摘要真正生成出来才删）：否则「摘要失败 + 原文照删」会让这段对话永久丢失（首次压缩时连摘要都没有，非首次时旧摘要也补不回这批新内容，且从摘要字段上看不出来 —— 静默丢）。"""
    record_node("trim_history")
    # 每轮开头清掉上一轮的「预约待落地」标记。`booking_pending` 由 `entry_router` 置位、
    #    同一轮内被 `route_after_grounding` 消费；用户若忽略挂起卡、下一轮问别的，
    #    这个标记会**留在 checkpoint 里**把那一轮也劫持进预约确认 ⇒ 必须每轮清。
    out: dict = {"booking_pending": False}
    messages = state.get("messages") or []
    budget = token_budget()

    # 先看**真实 usage 锚点**（`specs/0013` T5）：拿得到就以它为准绳，拿不到才回退本地估算。
    anchored = _anchor_estimate(state, messages)
    cut_budget = budget
    if anchored is None:
        if not should_compress(messages, budget):
            return out
    elif anchored <= budget:
        return out
    else:
        # 锚点超预算 ⇒ 压缩。但切点仍由 `split_index` 在"估算口径"下划，两个口径必须换算到同一把尺子：k = anchored / 估算总量，把预算换算进估算空间（`budget / k`）—— **不新设阈值**，唯一权威仍是 `AGENT_HISTORY_TOKEN_BUDGET`。
        # 否则会出现"判定要压、却算出 cut=0"的自相矛盾：锚点 100 万 token 说该压，而本地估算只有 40 ⇒ split_index(40) 一条都不切。
        local_total = max(1, estimate_messages_tokens(messages))
        cut_budget = max(1, int(budget * local_total / anchored))

    cut = split_index(messages, cut_budget)
    outdated = [m for m in messages[:cut] if getattr(m, "id", None)]
    if not outdated:
        return out

    before_tokens = estimate_messages_tokens(messages)
    logger.info(
        "历史压缩：拟移除 %s 条早期消息并生成摘要（%s token %s > 预算 %s；切点预算 %s）",
        len(outdated),
        "真实锚点" if anchored is not None else "本地估算",
        anchored if anchored is not None else before_tokens,
        budget, cut_budget,
    )

    prev_summary = (state.get("history_summary") or "").strip()
    new_summary = await _summarize_history(prev_summary, outdated, state)

    if not new_summary:
        # 摘要没生成出来 → 保留原文，本轮放弃压缩（消息暂时超长），下一轮再试。
        # 宁可暂时多占点 token，也不能把已经发生过的对话删掉。
        logger.warning(
            "历史压缩未产出摘要，保留 %s 条原文待下轮重试（本轮不删除）", len(outdated)
        )
        return out

    # 压缩后的估算 = 保留区 + 新摘要：摘要会作为 SystemMessage 注入下游 prompt，同样占上下文。
    after_tokens = estimate_messages_tokens(messages[cut:]) + estimate_tokens(new_summary)
    record_history_tokens(before_tokens, after_tokens)
    logger.info("历史压缩完成：估算 token %s → %s", before_tokens, after_tokens)
    return {
        **out,
        "messages": [RemoveMessage(id=m.id) for m in outdated],
        "history_summary": new_summary,
        # **压缩生效 ⇒ 锚点作废**：锚点对应的是"被删掉之前"的那份上下文，
        #    留着会让下一轮拿一个已经不存在的包去比预算（凭空触发/漏触发压缩）。
        #    下一轮自然走本地估算，直到新的 usage 到达。
        **_ANCHOR_RESET,
    }


def _render_for_summary(messages: list[Any]) -> str:
    """把待压缩的消息渲染成可读文本，供摘要模型消费。"""
    lines: list[str] = []
    for m in messages:
        content = as_text(getattr(m, "content", "")).strip()
        if not content:
            continue
        if isinstance(m, HumanMessage):
            lines.append(f"用户：{content}")
        elif isinstance(m, AIMessage):
            lines.append(f"助手：{content}")
    return "\n".join(lines)


#: 无段标题的旧文本归入此键（不是"段"，是兼容桶）。
HISTORY_LEGACY_KEY = "legacy"


def _match_history_header(line: str, names: tuple[str, ...]) -> tuple[str | None, str]:
    """判一行是不是段标题；是则返回 `(段名, 同行残文)`。宽松匹配：容忍 `#`/`*`/`-` 前缀与
    `:` / 全角 `：` 后缀（模型常加，不该因此丢段）。"""
    s = (line or "").strip().lstrip("#*- \t").strip()
    for name in names:
        if s.startswith(name):
            return name, s[len(name):].lstrip("：: ").strip()
    return None, ""


def parse_history_sections(
    text: str, names: tuple[str, ...] = HISTORY_SECTIONS
) -> dict[str, list[str]]:
    """把摘要文本解析成"段名 → 行列表"（**纯函数**）：命中段标题即切段（同行残文也收进该段）；无段标题的旧文本整体归入 `legacy`，不丢不抛（向后兼容旧 checkpoint，`specs/0016` 验收 7）；段名由 `names` 参数化（`specs/0016` §8 Q7：为 plan 轨段名留口，本期只接 recommend 轨）。
    """
    sections: dict[str, list[str]] = {name: [] for name in names}
    legacy: list[str] = []
    current: list[str] | None = None
    for raw in (text or "").splitlines():
        name, rest = _match_history_header(raw, names)
        if name is not None:
            current = sections[name]
            if rest:
                current.append(rest)
            continue
        line = raw.strip()
        if not line:
            continue
        (current if current is not None else legacy).append(line)
    if legacy:
        sections[HISTORY_LEGACY_KEY] = legacy
    return sections


def render_history_sections(
    sections: dict[str, list[str]], names: tuple[str, ...] = HISTORY_SECTIONS
) -> str:
    """按**固定段序**渲染五段（**纯函数**，同输入同字节；`specs/0016` 验收 12）：空段渲染成 `（无）` ⇒ "五段齐全"恒定成立；`legacy`（旧自由文本）内容折进**第一段**，绝不丢；段名参数化（为 plan 轨留口）。
    """
    legacy = [str(ln) for ln in (sections.get(HISTORY_LEGACY_KEY) or []) if str(ln).strip()]
    blocks: list[str] = []
    for idx, name in enumerate(names):
        lines = [str(ln) for ln in (sections.get(name) or []) if str(ln).strip()]
        if idx == 0 and legacy:
            lines = legacy + lines
        blocks.append(name + "\n" + "\n".join(lines if lines else ["（无）"]))
    return "\n\n".join(blocks)


# ===========================================================================
# 摘要投影：`[已推荐]` / `[死路]` / `[待办]` 的**确定性养料**（`specs/0016` T4 / §10.3）
# ===========================================================================
def digest_max() -> int:
    """摘要投影的跨轮上限（`specs/0016` §10.3：必须有上限，同 `FOCUS_MAX`）。
    调用时读 env（导入期常量陷阱）。"""
    try:
        return max(1, int(os.getenv("AGENT_DIGEST_MAX", "10")))
    except (TypeError, ValueError):
        return 10


def _digest_items(value: Any) -> list[dict]:
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def _digest_score(value: Any) -> float | None:
    """记录里的 `score` 是整数（如 45）⇒ 转成展示值（4.5）；非数值/0 → None。"""
    if isinstance(value, (int, float)) and value:
        return round(value / 10, 1)
    return None


def _first_error(rec: dict) -> str:
    """记录上第一个非空的失败说明（`*_error` 家族）—— §8 Q4 的"真去取过但失败"判据。"""
    for key in ("error", "vouchers_error", "chunks_error", "blogs_error"):
        val = rec.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return ""


def project_recommended_digest(outputs: list[dict], recommended_shops: list[dict] | None = None) -> list[dict]:
    """`[已推荐]` 的确定性投影（`specs/0016` §10.3）：**本轮真正展示**的店。优先用 `recommend_shops`（recommend 轨实际发给前端的卡片，才是"真正展示"）；没有卡片时（plan 轨）退回 `worker_outputs` 里**带能力块**（`tasks` 非空）的记录，纯候选（没真取到内容）**不算**已推荐。纯函数：只读入参，不改。
    """
    src: list[dict] = []
    if recommended_shops:
        for shop in recommended_shops:
            if not isinstance(shop, dict):
                continue
            sid = shop.get("shop_id")
            try:
                sid = int(sid) if sid is not None else None
            except (TypeError, ValueError):
                sid = None
            name = str(shop.get("name") or "").strip()
            if sid is None or not name:
                continue
            score = shop.get("score")
            src.append({
                "id": sid,
                "name": name,
                "score": score if isinstance(score, (int, float)) else None,
                "avgPrice": shop.get("avgPrice"),
                "has_voucher": bool(shop.get("vouchers")),
            })
    else:
        for rec in _react_candidates(outputs):
            if not rec.get("tasks"):
                continue
            sid = _shop_id_of(rec)
            name = str(rec.get("name") or "").strip()
            if sid is None or not name:
                continue
            src.append({
                "id": sid, "name": name,
                "score": _digest_score(rec.get("score")),
                "avgPrice": rec.get("avgPrice"),
                "has_voucher": bool(rec.get("vouchers")),
            })
    return src


def project_excluded_digest(outputs: list[dict]) -> list[dict]:
    """`[死路]` 的确定性投影（`specs/0016` §8 Q4）：**真去取过但失败 / 点名却没找到**的店。收：记录级 `*_error`（详情/券/片段/攻略取数失败）的店、`grounding` **没找到**的点名店；不收：`grounding` **服务故障**（店不一定不存在）、`nearby` 空结果（挪 `[目标与约束]`）、被 `_narrow_by_subcategory` 筛掉的店（是排序压下去、不是死了）。
    """
    out: list[dict] = []
    for item in outputs or []:
        if not isinstance(item, dict):
            continue
        if item.get("worker") == "grounding":
            if item.get("outcome") == _LOOKUP_ERROR:
                continue                      # 服务故障 ≠ 店死了，不收
            for raw_name in item.get("names") or []:
                name = str(raw_name).strip()
                if name:
                    out.append({"name": name, "reason": "没找到这家店"})
            continue
        for rec in item.get("items") or []:
            if not isinstance(rec, dict):
                continue
            sid = _shop_id_of(rec)
            if sid is None:
                continue
            reason = _first_error(rec)
            if reason:
                out.append({
                    "id": sid,
                    "name": str(rec.get("name") or f"店铺{sid}").strip(),
                    "reason": reason,
                })
    return out


def card_digest_of(shops: list[dict]) -> list[dict]:
    """卡片**必填字段**摘要（`specs/0017` T2 / §10.1）：`[{name, score, has_voucher}]`。这三项正是行为回归集断言 3 要检查的维度（店名 / 评分 / 是否带券）——"缺字段即失败"恰好是本项目历史上最常静默出问题的维度。纯函数。
    """
    out: list[dict] = []
    for shop in shops or []:
        if not isinstance(shop, dict):
            continue
        out.append({
            "name": str(shop.get("name") or "").strip(),
            "score": shop.get("score"),
            "has_voucher": bool(shop.get("vouchers")),
        })
    return out


def _todo_lines(state: Any) -> list[str]:
    """`[待办]` 的确定性行（`specs/0016` §8 Q5 **方案 C**）：**只取 `task_draft` 里进行中的行程**。预约"待确认"**不进**（预约轮不走 `summarize`、挂起期一律 `resume` ⇒ 与摘要生成不相交，其载体是 `interrupt()` 卡本身）；**不新造 `status` 字段** —— 只看 `task_draft` 在不在。
    """
    draft = (state or {}).get("task_draft") if isinstance(state, dict) else None
    if not isinstance(draft, dict) or not draft:
        return []
    slots = draft.get("slots") if isinstance(draft.get("slots"), dict) else {}
    bits: list[str] = []
    for key, label in (("time", "时间"), ("area", "区域"), ("people", "人数"),
                       ("budget", "预算"), ("scenes", "活动")):
        val = slots.get(key)
        if val in (None, "", [], ()):
            continue
        bits.append(f"{label}{'/'.join(map(str, val)) if isinstance(val, list) else val}")
    text = "行程进行中" + ("：" + " ".join(bits) if bits else "")
    stops = draft.get("stops") or []
    if stops:
        text += f"（已排 {len(stops)} 站）"
    return [text]


def _recommended_line(item: dict) -> str:
    name = str(item.get("name") or "").strip()
    parts: list[str] = []
    score = item.get("score")
    if isinstance(score, (int, float)) and score:
        parts.append(f"评分{score}")
    avg = item.get("avgPrice")
    if avg:
        parts.append(f"人均{avg}")
    if item.get("has_voucher"):
        parts.append("有券")
    return f"{name}（{' / '.join(parts)}）" if parts else name


def _dead_end_line(item: dict) -> str:
    name = str(item.get("name") or "").strip()
    reason = str(item.get("reason") or "").strip()
    return f"{name}（{reason}）" if reason else name


def _drop_recommended_from_excluded(recommended: list[dict], excluded: list[dict]) -> list[dict]:
    """`[死路]` 清理（`specs/0020` §2 目标 1）：店铺一旦进入 `[已推荐]`，同步移除其 `[死路]` 记录。在**读侧**做：两个 digest 是**跨轮累积**的（`state.merge_digest`，上限 `AGENT_DIGEST_MAX`）⇒ 同一家店可能既在已推荐、又残留旧死路（原本要等 10 条上限挤出才消失），渲染 `[死路]` 段前过滤一次即可根治，且**不动累积语义**。匹配按 `id`（`_shop_id_of`，兼容旧 `shop_id`）**或**去空格的 `name` —— 点名类死路只有 name、没有 id。纯函数，便于单测。
    """
    rec_ids: set[Any] = set()
    rec_names: set[str] = set()
    for item in recommended or []:
        sid = _shop_id_of(item)
        if sid is not None:
            rec_ids.add(sid)
        name = str(item.get("name") or "").strip()
        if name:
            rec_names.add(name)
    out: list[dict] = []
    for item in excluded or []:
        sid = _shop_id_of(item)
        name = str(item.get("name") or "").strip()
        if (sid is not None and sid in rec_ids) or (name and name in rec_names):
            continue
        out.append(item)
    return out


def _inject_deterministic_sections(sections: dict, state: Any) -> None:
    """把 `[已推荐]` / `[死路]` / `[待办]` 三段**确定性行**写进 `sections`（覆盖模型写的）。`specs/0016` §8 Q3：模型只负责 `[目标与约束]` 与 `[最近一句]`；这三段即使模型自由发挥，也在这里被本地**覆盖** —— 从机制上杜绝"把店名录错"。`[死路]` 段额外做**已推荐清理**（`specs/0020`）：见 `_drop_recommended_from_excluded`。
    """
    if not isinstance(sections, dict):
        return
    st = state if isinstance(state, dict) else {}
    recommended = _digest_items(st.get("recommended_digest"))
    sections[HISTORY_SECTION_RECOMMENDED] = [
        _recommended_line(x) for x in recommended if x.get("name")
    ]
    sections[HISTORY_SECTION_DEAD_END] = [
        _dead_end_line(x)
        for x in _drop_recommended_from_excluded(recommended, _digest_items(st.get("excluded_digest")))
        if x.get("name")
    ]
    sections[HISTORY_SECTION_TODO] = _todo_lines(st)


async def _summarize_history(
    prev_summary: str, outdated: list[Any], state: HmdpState | None = None
) -> str | None:
    """用模型把旧消息（连同已有摘要）压缩成新的 running summary（**五段式**，`specs/0016` T3）。
    只有真正生成出摘要才返回非 None，失败或模型返回空一律返回 None，由调用方据此决定"不删原文"；绝不能 fallback 成 `prev_summary`（调用方会把"沿用旧摘要"误判成"压缩成功"进而删掉原文，而这批原文根本没并进摘要 ⇒ 内容静默丢）。
    """
    try:
        record_model_call("history_summary")
        text = _render_for_summary(outdated)
        human = (
            f"已有的对话摘要：\n{prev_summary}\n\n"
            f"需要并入的新一轮对话：\n{text}\n\n"
            f"请输出合并后的最新摘要（五段，`[已推荐]`/`[死路]`/`[待办]` 留空）："
        )
        resp = await get_light_model().ainvoke(
            [SystemMessage(history_summary_system()), HumanMessage(human)]
        )
        raw = as_text(getattr(resp, "content", "")).strip()
        if not raw:
            logger.warning("历史摘要模型返回空内容，本轮不压缩")
            return None
        sections = parse_history_sections(raw)
        # `[已推荐]`/`[死路]`/`[待办]` 由代码**确定性注入**（`specs/0016` §8 Q3）：
        #    模型就算写了这三段，也在这里被本地方**覆盖**，杜绝"店名录错"。
        _inject_deterministic_sections(sections, state)
        return render_history_sections(sections)
    except Exception as exc:
        logger.error("历史摘要生成失败，本轮不压缩（保留原文待下轮重试）: %s", exc)
        return None


def _summary_block(state: HmdpState) -> list[SystemMessage]:
    """把压缩摘要作为一条系统消息注入下游 prompt，弥补被删除的早期消息。
    注入前经 `parse → render` 归一化：旧 checkpoint 的自由文本被折进 `[目标与约束]`，新写入的五段文本原样（幂等）——下游看到的永远是同一套五段结构（`specs/0016` §10.2）。
    """
    s = (state.get("history_summary") or "").strip()
    if not s:
        return []
    return [
        SystemMessage(
            "以下是更早若干轮对话的压缩摘要（原文已移除，关键信息保留在此）：\n"
            + render_history_sections(parse_history_sections(s))
        )
    ]


#: 截断标记：摘要被砍时必须**显式**留下它，禁止静默丢（`specs/0012` §11.4）。
PLAN_DIGEST_OMITTED = "……（更早的内容已省略）"


def _digest_for_memory(text: str, limit: int | None = None) -> str:
    """把长回答压成**进得了对话记忆的摘要**（`specs/0012` §11.4）：大内容（行程 stops/legs、澄清选项）走结构化载荷不进 `messages`，正文只放摘要，否则撑爆 `AGENT_HISTORY_TOKEN_BUDGET`、触发更频繁的历史压缩、压缩把细节压没。
    截断必须显式标注（追加 `PLAN_DIGEST_OMITTED`），静默截断会让"内容少了"无从察觉。
    """
    s = (text or "").strip()
    cap = PLAN_ANSWER_MAX_CHARS if limit is None else int(limit)
    if cap <= 0 or len(s) <= cap:
        return s
    return s[:cap] + PLAN_DIGEST_OMITTED


def recommend_digest_max_chars() -> int:
    """recommend 轨写回 `state.messages` 的正文摘要上限（`specs/0016` T5 / §8 Q6）：默认 **350** 字（整个 `AIMessage` 的 digest，不是每家店）——多轮会话里 `summarize_answer` 原先把完整正文写回 `messages`，message list 里累积长正文，既撑预算又触发频繁压缩（`specs/0016` §1.2）。
    调用时读环境（本项目已知的"导入期常量"陷阱：进程起来后改 env 永不生效且不报错）。
    """
    try:
        return max(0, int(os.getenv("AGENT_RECOMMEND_DIGEST_MAX_CHARS", "350")))
    except (TypeError, ValueError):
        return 350


#: 二次校验数的是"**可验证痕迹**"的类别数：时间序列 / ≥2 类消费场景 / 区域词。
#: 它挡的是"模型把简单问句误判成 plan"，不是"判断这句话像不像行程"——所以刻意做得窄：宁可漏判（放回 recommend 仍有答案，且用户再说一句就能进），也不要误杀真行程。
_PLAN_TIME_SEQ = re.compile(r"先|再|然后|接着|之后|最后")
_PLAN_DAY_PARTS = ("上午", "中午", "下午", "傍晚", "晚上", "夜里", "早上")
_PLAN_AREA = re.compile(r"一带|商圈|附近|周边|街区|片区")
#: 消费场景分组：**必须按"活动类型"分组** —— "下午茶 + 晚餐"同属"吃"，但它们是**两个不同场景**，
#: 这正是 T2 要数的东西；若只用一个"吃"大类，这条最常见的正例就会被自己挡掉。
_PLAN_SCENE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dessert", ("下午茶", "甜品", "蛋糕", "咖啡", "奶茶", "茶饮")),
    ("meal", ("晚餐", "午饭", "午餐", "中餐", "火锅", "烧烤", "日料", "西餐", "小吃", "夜宵", "吃饭")),
    ("show", ("电影", "演出", "展览", "话剧", "脱口秀", "音乐会")),
    ("play", ("ktv", "桌游", "密室", "电玩", "台球", "剧本杀")),
    ("shop", ("逛街", "购物", "商场", "市集")),
    ("leisure", ("散步", "公园", "景点", "打卡", "拍照")),
)


def plan_trace_categories(query: str) -> set[str]:
    """query 里**可验证**的行程痕迹类别，取值域 `{"time","scene","area"}`（纯函数）。
    与 `ENTRY_SYSTEM` 的 T1/T2/T3 一一对应，但这里是代码、可单测：`time`：出现衔接词（先/再/然后…），或同时提到 ≥2 个时段词；`scene`：命中 ≥2 个不同的消费场景组；`area`：出现区域词。
    """
    q = (query or "").strip().lower()
    if not q:
        return set()
    cats: set[str] = set()
    if _PLAN_TIME_SEQ.search(q) or sum(1 for p in _PLAN_DAY_PARTS if p in q) >= 2:
        cats.add("time")
    if sum(1 for _, kws in _PLAN_SCENE_GROUPS if any(k in q for k in kws)) >= 2:
        cats.add("scene")
    if _PLAN_AREA.search(q):
        cats.add("area")
    return cats


def _env_flag(name: str, default: bool) -> bool:
    """布尔开关：未设 / 空串取默认值；接受 `1/true/yes/on`（大小写不敏感）。
    与 `state._task_ttl_turns` 同理，**在调用时读环境**——"导入期常量"在本项目是已知陷阱（进程起来后模块已在 `sys.modules`，再改 env 永不生效且不报错）。
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _demote_to_recommend(plan: RoutePlan) -> RoutePlan:
    """把 plan 回落成 recommend（其余已抽到的字段原样保留；`sub_intents` 为空时补 `nearby`，
    否则下游 `grounding` 会拿到一个"没有任何意图"的推荐请求）。"""
    return plan.model_copy(
        update={"scene": "recommend", "sub_intents": plan.sub_intents or ["nearby"]}
    )


def _apply_plan_guard(plan: RoutePlan, query: str) -> RoutePlan:
    """入口**二次校验**：模型判了 plan，但 query 里的可验证痕迹类别 **< 2** ⇒ 强制回落 recommend；`AGENT_PLAN_GUARD_ENABLED=false` 时整套跳过（退回"完全信任模型"，用于对照实验）。
    warning 里带上 query 摘要是为了能统计误判率——不是为了静默吞掉。
    """
    if plan.scene != "plan" or not _env_flag("AGENT_PLAN_GUARD_ENABLED", True):
        return plan
    cats = plan_trace_categories(query)
    if len(cats) >= 2:
        return plan
    logger.warning(
        "入口二次校验：判为 plan 但可验证痕迹不足（命中 %s），回落 recommend；query=%s",
        sorted(cats), (query or "")[:60],
    )
    return _demote_to_recommend(plan)


#: 用户向的**降级/兜底**文案。住在这里是因为"降级"由入口判定——下游节点（`nearby_worker` / `daily_chat`）按 `state["route_degraded"]` 取用。
#: 两条必须**分开**：用户没说清（正常路由，只是判不出品类）⇒ 请 ta **补充说明**；我们没反应过来（入口模型不可用）⇒ 说**服务忙、稍后再试**。
#:    混在一起就是把**服务故障**说成**用户的表述问题**（2026-10-05 真机：轻量模型额度耗尽时答"没能判断你想找什么类型的店铺"，用户只会反复改措辞）。
NEARBY_TYPE_UNKNOWN_NOTICE = "没能判断你想找什么类型的店铺，能补充一下吗？比如火锅、咖啡、KTV。"

#: 降级文案。不含任何内部实现细节（不提模型/额度/接口），只说"我这边忙" + 可重试。
#: 同一根因（入口降级）**全项目共用这一句** —— 换了节点也不许另起一句。
#: 定义搬到了 `degrade.py`（单点真源），这里只做引用（纯重构，字面量一字未改）。
ROUTE_DEGRADED_NOTICE = degrade.ROUTE_DEGRADED_NOTICE

#: 「追问更多」但**确实已经列尽**时的文案（2026-10-06）。
#: 为什么必须单独一句：用户接着问「还有吗」时，资料里**只有店铺属性、没有"是否已列尽"这条元信息**，汇总模型于是按 `SUMMARY_SYSTEM` 的"资料里没有答案就说暂未查到"规则回答 ⇒ 把"**范围内的店已经全部列出来了**"说成了"**查不到**"（真机：问「有什么好吃的店吗」→ 得到 103茶餐厅，再问「还有吗」→ 答"根据现有博客暂未查到"）。
#:    ⇒ 由 `nearby_worker` 在**去重后无新增**时直接给结论（走 `empty` 通道，语义＝"查过了，确实没有更多"），汇总层照实转达即可，不必自己推断"是不是没有答案"。
NEARBY_NO_MORE = "附近「{type_name}」类的店铺已经全部列出来了，暂时没有其他家。"

#: 细分品类筛选**真的筛空**时的如实文案（`specs/0028`）。
#: 与 `NEARBY_NO_MORE` 的区别：后者是"这个**类型**已经列尽"，本句是"该类型**有店**，但**没有符合用户具体品类**的"。
#:    触发边界**保守**：仅"用户原词存在且比归类词更细（`raw != type_name`）"时筛选才可否决，此时若筛空即**不发不相关卡片**并如实说没有；其它情况保持"只收窄不否决"（`specs/0009` 口径不回归）。
SUBCATEGORY_EMPTY = "附近暂时没有「{term}」相关的店铺。"

#: 「再来一批」类追问的判据（纯规则，不调模型）。命中才启用"**按上文已展示过的店去重**"，
#: 避免用户在同一会话里**认真重问**时被误当成"要新的一批"而什么都查不到。
#: 全部**锚定整句**（`^…$`）——"还有一家在哪儿"这类带实义的问句**不算**追问。
_MORE_REQUEST_PATTERNS = (
    # 还有吗 / 还有别的吗 / 还有其他店吗 / 还有别的店
    r"^还有(吗|么|呢|没有|别的|其他|其它|其它的|别的店|其他店|其他的)"
    r"(的)?(店|几家|一些)?(吗|么|呢)?[?？]?$",
    # 还有什么 / 还有哪些 / 还有几家
    r"^还有(什么|哪些|几家|几个|一些)",
    # 其他的呢 / 别的呢 / 剩下的都列出来 / 其它店
    r"^(其他|其它|别的|剩下的)(的)?(呢|吗|店|都列出来|有哪些|是什么)?[?？]?$",
    r"^(继续|接着)(列|说|推荐|看看|来)?(吧)?[?？]?$",
    r"^换一批$",
    # 就这些吗 / 这些吗
    r"^(就)?这些(吗|么|呢)?[?？]?$",
)


def _is_more_request(query: str) -> bool:
    """用户是不是在要"**更多同类**"（纯函数）。
    保守匹配：宁可漏判（退回"重新查一遍"的老行为）也不误判——误判会让用户认真重问时被去重掉、什么都看不到。
    """
    text = re.sub(r"\s+", "", str(query or ""))
    if not text:
        return False
    return any(re.search(p, text) for p in _MORE_REQUEST_PATTERNS)


async def _llm_plan(state: HmdpState, query: str, mode_hint: str) -> RoutePlan:
    """便宜模型做意图识别 + 实体抽取；**连续失败则抛异常**（由 `entry_router` 决定如何降级）：静默兜底会把**服务故障**伪装成**正常路由结果**，降级是编排决策；规则快通道已删除（`_fast_plan` 会把纯指代问题吞进闲聊）⇒ 统一走一次模型判定并注入 `_focus_hint`，本轮还产出依赖判据（`specs/0005` §5.5.3）。
    :raises Exception: 3 次全部失败时抛出最后一次的异常（保留原始类型/信息，便于区分"额度耗尽"与"网络超时"）。
    """
    router = structured(get_light_model(), RoutePlan)
    messages = [
        SystemMessage(ENTRY_SYSTEM),
        *_summary_block(state),
        *_focus_hint(state),
        *_history_before_current(state.get("messages"), token_budget()),
        HumanMessage(f"入口模式：{mode_hint}\n用户问题：{query}"),
    ]
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            record_model_call("entry_router")
            result = await router.ainvoke(messages)
            plan = result if isinstance(result, RoutePlan) else RoutePlan.model_validate(result)
            return _apply_plan_guard(plan, query)
        except Exception as exc:  # 单次失败仍要重试（限流/抖动多数能自愈）
            last_error = exc
            logger.warning("意图识别失败（第 %s 次）: %s", attempt, exc)
            await asyncio.sleep(min(1.5 ** attempt, 5))

    logger.error("意图识别连续失败，交由入口降级处理: %s", last_error)
    raise last_error if last_error else RuntimeError("意图识别失败")


#: 「能力咨询」的**确定性**识别（不调模型）：问"你能做什么 / 有什么功能 / 你是谁 / 怎么用"。
#: 这类问题不含找店/优惠关键词，会被入口模型判成 daily，而 daily 提示词只教模型"陪聊" ⇒ 在模型判定之前短路，用固定文案如实介绍能力（见 `bug/BUG-20261003-capability-question-misrouted-to-daily.md`）。
#: 只匹配"能力咨询"措辞，不吞找店/优惠问句；动词族一律要求主语"你"（`你能/你会…`），否则"这家店能做什么菜"会被误判；正则漏掉的少数问法落到 daily，由 `DAILY_SYSTEM` 兜底规则接住。
_CAPABILITY_PATTERNS = re.compile(
    "|".join(
        (
            r"有什么(功能|能力|作用|用处)",
            r"你(能|会)(做|帮|干)(我)?(做)?(点)?(什么|啥)",
            r"你(会|能)(做|些)?什么",
            r"你是谁",
            r"你(叫|是)(什么名字|谁)",
            r"介绍(一下)?(你|你自己)",
            r"怎么用",
            r"使用(说明|方法)",
            r"功能(介绍|列表)",
            r"help",
        )
    ),
    re.IGNORECASE,
)


def _is_capability_query(query: str) -> bool:
    """是否为"你能做什么 / 有什么功能"类能力咨询。"""
    q = (query or "").strip()
    return bool(q) and _CAPABILITY_PATTERNS.search(q) is not None


def _capture_route(plan: "RoutePlan") -> None:
    """把**意图路由结果**写进账本（`specs/0017` T2）：埋点复用已有 ContextVar `run_id`，不新增透传机制；受 `AGENT_LEDGER_CAPTURE_CONTEXT`（默认开）与 `AGENT_LEDGER_ENABLED` 双重控制。
    旁路：`record_route` 本身入队即返回（无池/关开关即 no-op），这里再加一层兜底 try。
    """
    try:
        ledger.record_route(ledger.get_run_id(), plan.scene, plan.sub_intents, plan.depends_on)
    except Exception as exc:  # noqa: BLE001 - 埋点绝不打断主链路
        logger.warning("意图路由埋点失败（不影响本轮）: %s", exc)


async def entry_router(
    state: HmdpState,
) -> Command[Literal["daily_chat", "grounding", "capability_chat", "plan_slots", "replace_stop"]]:
    """第一层入口：先做意图判断再转发，不管前端传的是哪种模式。"""
    record_node("entry_router")
    query = (state.get("query") or "").strip()
    mode_hint = state.get("mode_hint") or "recommend"

    # 能力咨询确定性短路：不调模型、直接用固定文案介绍能力（避免被判 daily 后把推荐能力说没）。
    if _is_capability_query(query):
        plan = RoutePlan(scene="capability", rewritten_query=query)
        logger.info("入口路由: scene=capability（能力咨询，确定性短路）")
        _capture_route(plan)
        return Command(
            update={"plan": plan.model_dump(), "route_degraded": False},
            goto="capability_chat",
        )

    # 计划轨上下文里的"第N站换一家"：**确定性**短路。这类问句不含任何找店关键词，交给入口模型
    # 多半判成 recommend ⇒ 把"替换一站"做成了"重新找店"，用户已认可的前后站全被打散（§12.5）。
    if _env_flag("AGENT_PLAN_ENABLED", False) and _replace_stop_seq(
        query, (state.get("task_draft") or {}).get("stops") or []
    ) is not None:
        plan = RoutePlan(scene="plan", rewritten_query=query)
        logger.info("入口路由: scene=plan（单站替换，确定性短路）")
        _capture_route(plan)
        return Command(
            update={"plan": plan.model_dump(), "route_degraded": False},
            goto="replace_stop",
        )

    # 预约：**确定性**短路（写路径的入口判定绝不交给模型，见 `_BOOKING_INTENT` 注释）。
    # 判据是 **`action.type == "booking"`**，不是 query 里的关键词——结构化确认的入口是前端点的按钮，此时 `message` 是空串；早先按 `_BOOKING_INTENT.search(query)` 判 ⇒ 空 message 永不命中 ⇒ 真机点了按钮直接进 daily_chat。
    # 保留关键词分支是为了兼容"用户直接打字说帮我预约"的自然语言入口。
    action = state.get("action") or {}
    action_type = str(action.get("type") or "") if isinstance(action, dict) else ""
    wants_booking = action_type == "booking" or (
        action_type not in ("confirm", "cancel") and bool(_BOOKING_INTENT.search(query))
    )
    if _env_flag("AGENT_PLAN_ENABLED", False) and wants_booking:
        # 未登录**不进**这条分支：`booking_confirm` 缺 user_id 时会 `goto entry_router`，
        #    而入口这边靠关键词仍会命中预约意图 ⇒ 无限往返。未登录保持原行为（交回模型）。
        logged_in = bool(str(state.get("user_id") or "").strip())
        # ① 前端点的结构化建议项：参数最全，**零模型**直接短路。
        draft = _booking_draft_from_action(state)
        if draft and logged_in:
            plan = RoutePlan(scene="plan", rewritten_query=query)
            logger.info(
                "入口路由: scene=plan（预约确认，确定性短路 via=action）shop_id=%s", draft.get("shop_id"),
            )
            _capture_route(plan)
            return Command(
                update={"plan": plan.model_dump(), "booking_draft": draft,
                        # **新预约 ⇒ 计数清零**（`plans/0021` §16.2 的上限是"每张确认卡"的）：
                        #   `booking_draft` 每次点预约按钮都会重建，而计数只增不减 ⇒ 第二次预约会带着
                        #   上一张卡的残值；与"取消时不重置"不矛盾：那里守"同一张卡别被改到没完没了"，这里是"新的一张卡，从零开始"。
                        "booking_modify_count": 0,  # 新预约必须清零，否则上一张卡的残值卡死上限（plans/0021 §16.2）
                                                "booking_modify_error": "",
                        # 同一次清零：补时间的"已试过一轮"标记也属于**这一张卡**的状态
                        "booking_fill_tried": False,
                        "route_degraded": False},
                goto="booking_confirm",
            )

        # ② 自然语言入口（"帮我预约下" / "我要预约星聚会KTV"）。
        #    **本轮点名的店必须先被解析**（真机 2026-10-08 复现）：旧写法直接拿 `last_grounded_shop_id`
        #       建草稿，而它是 `grounding`（本节点下游，本轮还没跑）写的 ⇒ 说的店名只会约成上一轮那家；⇒ 让入口模型抽店名 → `grounding` 解析 → `route_after_grounding` 转 `booking_confirm`；本轮没点名时才用上下文草稿，省掉一次 grounding。
        if logged_in:
            plan = await _llm_plan(state, query, mode_hint)
            named = [n for n in (plan.shop_names or []) if n]
            if named:
                logger.info("入口路由: 预约意图 + 本轮点名 %s ⇒ 交给 grounding 解析", named)
                _capture_route(plan)
                return Command(
                    update={"plan": plan.model_dump(), "booking_pending": True,
                            "route_degraded": False},
                    goto="grounding",
                )
            draft = _booking_draft_from_focus(state)
            if draft:
                plan = RoutePlan(scene="plan", rewritten_query=query)
                logger.info("入口路由: 预约意图（未点名，取上下文最近一家）shop_id=%s", draft.get("shop_id"))
                _capture_route(plan)
                return Command(
                    update={"plan": plan.model_dump(), "booking_draft": draft,
                            "booking_modify_count": 0,
                            "booking_modify_error": "",
                            "booking_fill_tried": False,
                            "route_degraded": False},
                    goto="booking_confirm",
                )
            # 连店铺都定位不到（既没有 action、焦点也空）⇒ **确定性追问**，
            # 不交回模型：模型会把"帮我预约"答成"我暂时还不能帮你预约"，看着像功能缺失。
            plan = RoutePlan(scene="recommend", rewritten_query=query,
                             clarify_question="想预约哪家店呢？告诉我店名就行。")
            logger.info("入口路由: 命中预约意图但定位不到店铺，确定性追问（不交回模型）")
            _capture_route(plan)
            return Command(
                update={"plan": plan.model_dump(), "route_degraded": False}, goto="clarify"
            )
        logger.info("入口路由: 命中预约意图但未登录，交回模型路由")

    # 确认/取消的**兜底**通道：`generate_sse` 正常会用 `Command(resume=action)` 续跑；
    #    万一那时读 checkpoint 失败走了新回合，这里保证 action 不被当成闲聊丢掉。
    if (
        _env_flag("AGENT_PLAN_ENABLED", False)
        and action_type in ("confirm", "cancel")
        # `booking_result` 也要算：确认之后草稿已被清空，取消单号在那里
        #    （只认 `booking_draft` ⇒ 确认后的「取消预约」永远进不了 booking_confirm）。
        and (state.get("booking_draft") or state.get("booking_result"))
    ):
        logger.info("入口路由: 收到确认动作 %s，转 booking_confirm", action_type)
        plan = RoutePlan(scene="plan", rewritten_query=query)
        _capture_route(plan)
        return Command(
            update={"plan": plan.model_dump(), "route_degraded": False},
            goto="booking_confirm",
        )

    # 追问「还有吗」：确定性短路，不调模型（2026-10-07 进程内重放实测）。
    #    不能交给模型：同一句话同一份代码，入口模型的判定会来回翻转 ⇒ "用户能不能看到下一批"成了概率事件。
    #    判据两条都要满足：① `_is_more_request(query)`（纯规则、保守匹配）；② 上文确实有焦点（`focus_shops` 非空）——否则首问就该老老实实追问。放在预约/单站替换之后（那些是更具体的意图）。
    if _is_more_request(query) and (state.get("focus_shops") or []):
        plan = RoutePlan(scene="recommend", sub_intents=["nearby"], rewritten_query=query)
        logger.info("入口路由: scene=recommend（追问「更多」，确定性短路）")
        _capture_route(plan)
        return Command(
            update={"plan": plan.model_dump(), "route_degraded": False},
            goto="grounding",
        )

    # 空 query 不必调模型（原 _fast_plan 的这条兜底保留下来）：直接按闲聊处理。
    degraded = False
    if not query:
        plan = RoutePlan(scene="daily", rewritten_query="")
    else:
        try:
            plan = await _llm_plan(state, query, mode_hint)
        except Exception as exc:
            # 入口模型**不可用**（额度耗尽/网络/持续限流）⇒ 硬判 recommend 兜底，并**显式标记本条为降级**：
            #    降级后 `shop_type` 必为空，下游若照旧答"没能判断你想找什么类型的店铺"，就是把**服务故障**说成**用户没说明白**（2026-10-05 真机：轻量模型额度耗尽复现）；有了标记，下游才能改说"稍后再试"。
            #    兜底**恒判 recommend**（`specs/0021`）：`mode_hint` 已不再由客户端下发（恒为 recommend）⇒ 旧 `"daily" if mode_hint == "daily"` 分支是死代码，已删除；daily / plan 在模型可用时由入口模型正常判出。
            logger.error(
                "入口降级：意图识别不可用，按 recommend 兜底；err=%s: %s",
                type(exc).__name__, exc,
            )
            plan = RoutePlan(
                scene="recommend",
                sub_intents=["nearby"],
                rewritten_query=query,
            )
            degraded = True

    # 多活动问句的**确定性提升**（`specs/0029` / `plans/0047`）：与既有"回落"闸门对称——入口模型
    #    可能把"吃饭 → 唱歌 → 做发型"这类多活动问句误判成 recommend，而 recommend 轨结构性只装得下
    #    一个 `shop_type` ⇒ "唱歌/做发型"被静默丢弃；判据是**纯函数**，**提升必须在回落之前**；入口降级时不提升（计划轨同样依赖模型抽槽），由 T2 的"未覆盖如实说"兜底。
    if not degraded:
        promoted = _should_promote_to_plan(query, plan)
        if promoted:
            logger.info(
                "入口提升：scene=recommend → plan（多活动命中 %s）；query=%s",
                list(promoted), query[:60],
            )
            plan = plan.model_copy(
                update={"scene": "plan", "sub_intents": plan.sub_intents or ["nearby"]}
            )

    # 计划轨开关（默认关）：`plans/0016` 已把两个节点接进图，但**默认不走**——关着时判成 plan 也按 recommend 处理，保证"开关未打开时线上行为不变"。
    # 回落必须发生在**算 goto 之前**——否则 goto 会指向不该去的节点（0015 期间 `plan_slots` 还不存在，那时是"图编译直接失败"，现在则是"绕过开关"）。
    if plan.scene == "plan" and not _env_flag("AGENT_PLAN_ENABLED", False):
        logger.info(
            "计划轨未启用（AGENT_PLAN_ENABLED=false），按 recommend 处理；query=%s", query[:60]
        )
        plan = _demote_to_recommend(plan)

    if plan.scene in ("recommend", "plan") and not plan.rewritten_query:
        plan.rewritten_query = query


    logger.info(
        "入口路由: scene=%s intents=%s names=%s type=%s",
        plan.scene, plan.sub_intents, plan.shop_names, plan.shop_type,
    )
    _capture_route(plan)
    goto: Literal["daily_chat", "grounding", "capability_chat", "plan_slots"] = (
        "plan_slots" if plan.scene == "plan"
        else "capability_chat" if plan.scene == "capability"
        else "daily_chat" if plan.scene == "daily"
        else "grounding"
    )
    return Command(
        update={"plan": plan.model_dump(), "route_degraded": degraded}, goto=goto
    )


async def _stream_join(stream: Any, parts: list[str]) -> Any:
    """消费一个 `astream`：把**文本**追加进 `parts`，并返回**带 usage 的那条 chunk**：不能只留最后一个 chunk（没有 `usage_metadata` ⇒ 静默拿到 0），也不直接返回聚合体（`chunk + chunk` 时 usage 会累加虚高）⇒ 单独跟踪"最近一次带 usage 的 chunk"，都没有才退回聚合体。
    还有一半在模型侧：`stream_usage=True` 必须显式下发（`llm._build_one`）——轻量模型（`qwen3-max-preview`）不显式要就完全不返回 usage。
    """
    aggregate = None
    with_usage = None
    async for chunk in stream:
        text = as_text(getattr(chunk, "content", ""))
        if text:
            parts.append(text)
        if getattr(chunk, "usage_metadata", None):
            with_usage = chunk
        try:
            aggregate = chunk if aggregate is None else aggregate + chunk
        except Exception as exc:  # noqa: BLE001 - 聚合只为兜底拿 usage，失败不能影响出话
            logger.debug("流式 chunk 聚合失败（只影响 usage 采集）: %s", exc)
            aggregate = chunk
    response = with_usage if with_usage is not None else aggregate
    if response is not None:
        record_usage_from_message(response)
    return response


async def daily_chat(state: HmdpState) -> dict:
    """日常闲聊：轻量模型 + 最近历史（**窗口按 token 预算划定**），流式直出后直接 END。"""
    record_node("daily_chat")

    # 入口**已降级**（入口模型不可用）⇒ 不要再白调一次注定失败的模型：模型正是刚才失败的那个，
    #    重调只会再等一轮超时，然后落到下面的异常分支说出**另一句**"我这边出了点小状况"——同一根因两种说法，会被误当成两个故障。
    #    这里复用 `ROUTE_DEGRADED_NOTICE`（与 `nearby_worker` 同**一句**）且**不调模型**；返回 `answer` + `messages` 即走非流式出话路径（与 `capability_chat` 同款）。
    if state.get("route_degraded"):
        logger.info("daily_chat: 入口降级中，直接回服务繁忙提示（不再调模型）")
        return {
            "answer": ROUTE_DEGRADED_NOTICE,
            "sources": [],
            "messages": [AIMessage(content=ROUTE_DEGRADED_NOTICE)],
        }

    messages = [
        SystemMessage(daily_system()),
        *_summary_block(state),
        *_with_time_context(_recent_chat_messages(state.get("messages"), token_budget())),
    ]
    parts: list[str] = []
    try:
        record_model_call("daily_chat")
        response = await _stream_join(get_light_model().astream(messages), parts)
    except Exception as exc:
        logger.error("日常闲聊生成失败: %s", exc)
        answer = "抱歉，我这边出了点小状况，能再说一次吗？"
        return {"answer": answer, "sources": [], "messages": [AIMessage(content=answer)]}

    answer = "".join(parts).strip() or "嗯，我在这儿，有什么想聊的？"
    return {
        "answer": answer,
        "sources": [],
        "messages": [AIMessage(content=answer)],
        # usage 锚点（`specs/0013` T5 / §11.1）：daily 轨也要喂锚点，否则闲聊型会话
        # 永远只能靠本地估算判压缩。用 `_stream_join` 返回的**带 usage 的 chunk**
        # （最后一个 chunk 没有 usage，daily 轨的模型还必须在 `llm._build_one` 显式索要）。
        **anchor_patch(response, state.get("messages") or []),
    }


async def capability_chat(state: HmdpState) -> dict:
    """能力咨询：「你能做什么 / 有什么功能」的**确定性**回答（不经模型，避免自由发挥把推荐能力说没）。
    文案唯一真相源在 `prompts.CAPABILITY_ANSWER`；由 `entry_router` 的正则短路直接路由到本节点。"""
    record_node("capability_chat")
    return {
        "answer": CAPABILITY_ANSWER,
        "sources": [],
        "messages": [AIMessage(content=CAPABILITY_ANSWER)],
    }


# ===========================================================================
# 第二层（Plan 轨）：抽槽 → 澄清（interrupt）→（`plans/0017`：行程）
# ===========================================================================
class SlotFill(BaseModel):
    """抽槽的结构化输出。字段**全部可选**：用户没提到的槽位模型就不该填（宁缺勿猜）。"""

    model_config = ConfigDict(extra="ignore")

    time: str | None = None
    area: str | None = None
    people: int | None = None
    budget: int | None = None
    scenes: list[str] | None = None
    #: 每个活动 → **真实店铺类型**的归类结果（`specs/0031`）：与 `scenes` **下标一一对应**，
    #: 归不到就留**空串**（**保位置**，不压缩 —— 同 `specs/0030` 的 `shop_terms_raw` 口径）。
    #: 抽槽**本来就要调一次模型**，顺带归类 ⇒ 主路径**零额外调用**。
    scene_types: list[str] = Field(default_factory=list)

    @field_validator("scene_types", mode="before")
    @classmethod
    def _coerce_scene_types(cls, v: Any) -> Any:
        """容忍模型把 `scene_types` 回成**非 list / JSON 字符串**（同 `_coerce_scenes` 口径）。
        只修正形状、绝不因此作废整次抽槽：非 list / `None` ⇒ `[]`；逐项 `strip()`，非 str ⇒ 空串（保位置 ⇒ 与 `scenes` 下标继续对齐）。
        """
        if v is None:
            return []
        if isinstance(v, str):
            text = v.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                except ValueError:
                    return []
                v = parsed if isinstance(parsed, list) else []
            else:
                return []
        if not isinstance(v, list):
            return []
        return [x.strip() if isinstance(x, str) else "" for x in v]

    @field_validator("scenes", mode="before")
    @classmethod
    def _coerce_scenes(cls, v: Any) -> Any:
        """容忍模型把**数组**回成 **JSON 字符串**（真机 2026-10-12 复现，见 `bug/BUG-20261012-*`）：`with_structured_output(json_schema)` 在服务端不强制形状约束 ⇒ `scenes` 回成字符串 ⇒ Pydantic 抛 `list_type` ⇒ 整次抽槽作废（连已抽对的 `time`/`area` 一起丢）⇒ 用户被把 5 个槽位重问一遍。
        这里把"看起来像 JSON 数组的字符串"解析回列表；解析不出只丢弃该字段，绝不因一个字段的形状问题让整条抽槽失败（同 `_clean_sub_intents` 口径）。
        """
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, str):
            text = v.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                except ValueError:
                    return None
                # 只认"解析出来确实是 list"；元素非字符串的交给 Pydantic 再做一次校验
                return parsed if isinstance(parsed, list) else None
            return None  # 非数组形态的字符串（如 "逛街、吃饭"）不猜，丢弃该字段
        return None


async def _extract_slots(text: str, asking: list[str]) -> dict[str, Any]:
    """从用户这句话里抽槽（**只在 `interrupt()` 之后调用**——之前调会被重放、白烧一次模型）。
    `asking` 既作"这轮在问什么"的提示，也作采纳范围的上限（调用方再用 `_accept_filled` 收窄回 `asking`）；本函数不自己做那层收窄，只管"模型抽出了什么、值域合不合法"；返回结果已过 `sanitize_slots`，非法值直接丢弃。
    """
    human = (
        f"这轮正在问的槽位：{', '.join(asking) or '（无）'}\n"
        f"用户的原话：{text}"
    )
    try:
        record_model_call("clarify_slots")
        # 用**抽槽专用模型**而不是轻量模型（2026-10-06 实测）：`qwen-turbo` 在
        #    `SlotFill` 上 >25s **不返回**（3/3 复现），而它在入口路由/品类归类/行程/ReAct
        #    五个通道全都合格且快 10~30 倍 ⇒ 只把本通道隔离出去，其余仍走快模型。
        result = await structured(get_slot_model(), SlotFill).ainvoke(
            [SystemMessage(SLOT_SYSTEM), HumanMessage(human)]
        )
    except Exception as exc:
        logger.warning("抽槽失败（按未填处理）: %s", exc)
        return {}
    raw = (
        result.model_dump(exclude_none=True)
        if isinstance(result, SlotFill)
        else dict(result or {})
    )
    return sanitize_slots(raw)


def _new_task_draft(filled: dict[str, Any]) -> dict[str, Any]:
    """建一份 `task_draft`（结构见 `specs/0012` §11.2）：`slots` 里五个必填键都先占位，这样 `missing_slots` 只需判 `is_valid`，不必区分"键不存在"与"值为空"。
    `scene_types`（`specs/0031`）不是必填占位键：有归类结果才写进 `slots`（与 `scenes` 对齐）。
    """
    now = int(time.time())
    slots: dict[str, Any] = {name: filled.get(name) for name in REQUIRED_SLOTS}
    scene_types = aligned_scene_types(filled)
    if scene_types:
        slots[SCENE_TYPES_KEY] = scene_types
    return {
        "task_id": f"t-{now}",
        "clarify_retry": 0,
        "idle_turns": 0,
        "slots": slots,
        "stops": [],
        "created_at": now,
    }


def _accept_filled(filled: dict[str, Any], asked: list[str], answer: str) -> dict[str, Any]:
    """过滤澄清续答的抽槽结果（纯函数；`clarify_slots` 在 `interrupt()` **之后**调用）。三条规则（2026-10-08 真机依据）：① 只接受本轮正在问的槽位——旧行为允许模型顺带补别的槽位，实测用户答「2个人」时把 time/area/budget/scenes 一并填成"2个人"，合并后上一轮问对的值全被冲掉（错数据的代价是整条行程排错，两害相权取其轻）；② 值必须在用户原话里有痕迹（「1人」被填成 time/area 正是没有痕迹要求的后果）；③ 一个数字只能支撑一个数值槽（`budget=1` 是复用同一个 1 的幻觉，DB 实证 id 483）。
    值域（`plan_slots.is_valid`）在这里再过一遍（纯函数会被单测直接调，不假设调用方先过 `sanitize_slots`）；`scene_types`（`specs/0031` T3）不是槽位、不参与痕迹匹配，但必须与保留下来的 `scenes` 同步，否则下标错位会配出荒谬归类（静默且难查）。
    """
    text = str(answer or "")
    numbers = _reason_numbers(text)          # 按数字边界抽取，与推荐理由用同一套
    # ⓪ 值域不合格的一律不要（哪怕"原话里确实有这几个字"）：`"1人"` 能被当 time/area 用，
    #    正是因为只做"值非空"校验；这里再收一道，不依赖调用方是否已过 `sanitize_slots`。
    #    `scene_types` 不是槽位 ⇒ 先取出再过滤（否则会被 `slot_is_valid` 判不合法而丢掉）。
    scene_types_in = aligned_scene_types(filled)
    filled = {k: v for k, v in (filled or {}).items() if slot_is_valid(k, v)}
    out: dict[str, Any] = {}
    used: set[str] = set()
    # ① 数值槽：数字必须出现在原话里，且**一个数字只给一个槽**（先到先得，people 优先）
    for slot in ("people", "budget"):
        if slot not in filled:
            continue
        token = str(filled[slot])
        if token in numbers and token not in used:
            used.add(token)
            out[slot] = filled[slot]
    # ② 文本槽：原值必须能在原话里找到
    for slot in ("time", "area"):
        if slot not in filled:
            continue
        value = str(filled[slot]).strip()
        if value and value in text:
            out[slot] = filled[slot]
    # ③ 列表槽：逐项都要能在原话里找到（有痕迹的那几项保留）
    kept_scenes = filled.get("scenes") if isinstance(filled.get("scenes"), list) else []
    if kept_scenes:
        items = [s for s in kept_scenes if str(s).strip() and str(s).strip() in text]
        if items:
            out["scenes"] = items
            if scene_types_in:
                # 与保留项**同下标**取归类（`scene_types_in` 已与**原** `scenes` 等长，由调用方保证）
                out[SCENE_TYPES_KEY] = [
                    scene_types_in[i] for i, s in enumerate(kept_scenes)
                    if str(s).strip() and str(s).strip() in text
                ]
    # ④ 只留本轮正在问的；`scene_types` 依附 `scenes`，scenes 被采纳就跟着带上
    kept = {k: v for k, v in out.items() if k in asked}
    if "scenes" in kept and SCENE_TYPES_KEY in out:
        kept[SCENE_TYPES_KEY] = out[SCENE_TYPES_KEY]
    return kept


async def plan_slots(state: HmdpState) -> Command[Literal["clarify_slots"]]:
    """计划轨入口：把用户原始 query 抽成 `task_draft.slots`，交给 `clarify_slots` 决定问不问。"""
    record_node("plan_slots")
    query = (state.get("query") or "").strip()
    filled = await _extract_slots(query, list(REQUIRED_SLOTS))
    draft = _new_task_draft(filled)
    logger.info("计划轨建草稿：首轮抽到 %s，缺 %s", sorted(filled), missing_slots(draft))
    return Command(update={"task_draft": draft}, goto="clarify_slots")


def _normalize_structured_slot(slot: str, value: Any) -> Any:
    """把结构化澄清值归一成 `plan_slots` 能校验的形态（`specs/0025`）。
    `time`：结构化 `{start,end[,date]}` → `"HH:MM"` 或区间，带 `date` 前缀也照样通过值校验；`people`/`budget`：数值或数字串 → `int`，非法原样返回；`scenes`：字符串按常见分隔符拆成列表；其余槽位原样返回。
    """
    if slot == "time":
        if isinstance(value, dict):
            start = str(value.get("start") or "").strip()
            end = str(value.get("end") or "").strip()
            date = str(value.get("date") or "").strip()
            prefix = f"{date} " if date else ""
            if start and end and start != end:
                return f"{prefix}{start}-{end}"
            point = start or end
            return f"{prefix}{point}" if point else value
        return value
    if slot in ("people", "budget"):
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return value
    if slot == "scenes":
        if isinstance(value, str):
            items = [p.strip() for p in re.split(r"[，,、;；/\s]+", value) if p.strip()]
            return items or [value.strip()]
        return value
    return value


def _structured_filled(raw: dict, missing: list[str]) -> dict:
    """结构化澄清答案 → 可用槽位（`specs/0025`）。
    与文本通道的区别：结构化值来自**前端控件**（没有"原话痕迹"），故跳过 `_accept_filled` 的文本痕迹匹配，只保留 `plan_slots.is_valid` 的值域校验；且只收本轮正在问的槽位，不给未问槽位放行（沿用 `_accept_filled` 规则 ④ 的保守口径）。
    """
    out: dict[str, Any] = {}
    for slot in missing:
        if slot not in (raw or {}):
            continue
        value = _normalize_structured_slot(slot, raw[slot])
        if slot_is_valid(slot, value):
            out[slot] = value
    return out


def _structured_answer_text(raw: dict, filled: dict) -> str:
    """结构化答案转一句**人类可读**的话，仅用于回填 `messages`（不参与抽槽/校验）。"""
    parts: list[str] = []
    for slot in REQUIRED_SLOTS:
        if slot in filled:
            value = filled[slot]
            parts.append("、".join(str(x) for x in value) if isinstance(value, list) else str(value))
    return "，".join(parts) if parts else "（已提交选项）"


async def clarify_slots(
    state: HmdpState,
) -> Command[Literal["clarify_slots", "entry_router", END]]:
    """全图**唯一**的 `interrupt()` 点：既当"提问者"也当"收答者"（`specs/0012` §12.3.1）。
    `interrupt()` 之前只允许纯函数——本节点会被框架从头重放，副作用执行两次、模型调用白烧一次（§12.3.2）；重试/弃权判断必须在 `interrupt()` 之后，否则拿到答案也不会弃权 ⇒ 死循环；续跑时 `Command(resume=…)` 不会把用户输入写进 `state.messages`，必须自己回填。
    """
    record_node("clarify_slots")
    draft = dict(state.get("task_draft") or {})
    retry = int(draft.get("clarify_retry") or 0)

    missing = missing_slots(draft)
    if not missing:
        # 槽齐：不挂起，直接进入取数 → 编排。
        return Command(goto="plan_worker")

    if retry > max_clarify_retry():
        # 弃权出口（§11.3）：重试用尽 ⇒ 清草稿、把这句当**新回合**交回入口重路由。
        # 于是"用户挂起后走开、回来发了一句别的话"最多被劫持一次就自纠，无需在别处加启发式判断。
        logger.info("澄清重试超限（%s > %s），弃权并交回入口", retry, max_clarify_retry())
        return Command(
            goto="entry_router",
            update={"task_draft": None, "query": str(state.get("query") or "")},
        )

    payload = build_clarification(missing, draft.get("task_id") or "", retry=retry)
    answer = interrupt(payload)          # ← 挂起；续跑时**直接返回**用户输入（不再抛异常）

    # —— 以下只在"拿到答案之后"执行（每轮续跑至多一次）——
    # 结构化通道（`specs/0025`）：前端澄清卡按标准结构回传 ⇒ 直接映射槽位、跳过抽槽模型；
    # 判据用 `_CLARIFY_ANSWER_ACTIONS`（动作类型白名单）——与预约 `action` 同款"结构化入口"语义；原文本通道保留为兜底。
    if isinstance(answer, dict) and str(answer.get("type") or "").strip() in _CLARIFY_ANSWER_ACTIONS:
        raw_slots = answer.get("slots")
        raw_slots = raw_slots if isinstance(raw_slots, dict) else {}
        filled = _structured_filled(raw_slots, missing)
        answer_message = HumanMessage(content=_structured_answer_text(raw_slots, filled))
        logger.info("澄清结构化通道：采纳 %s（本轮问 %s）", sorted(filled), missing)
    else:
        text_answer = "" if answer is None else str(answer)
        # 抽槽结果必须过 `_accept_filled`：模型会把没问的槽位也"顺带"填上（实测把
        #    「2个人」填进 time/area/budget/scenes），直接合并会**覆盖**上一轮问对的值。
        filled = _accept_filled(await _extract_slots(text_answer, missing), missing, text_answer)
        # 见 docstring：续跑不会自动把用户输入放进 messages，必须回填（否则模型看不到用户答了什么）
        answer_message = HumanMessage(content=text_answer)

    new_draft = {
        **draft,
        "slots": {**(draft.get("slots") or {}), **filled},
        "clarify_retry": retry + 1,
    }
    if missing_slots(new_draft):
        return Command(
            goto="clarify_slots",
            update={"task_draft": new_draft, "messages": [answer_message]},
        )
    return Command(
        goto="plan_worker",
        update={"task_draft": new_draft, "messages": [answer_message]},
    )


#: 场景词 → 后端**店铺类型 / 搜索关键词**：用户说的是"下午茶""看电影"，后端按"咖啡/KTV"这类类型找店——不做这层映射，`fetch_shops_nearby("下午茶")` 必然为空，而且报出来的还是"附近没有店"（把"问错了词"伪装成"确实没有"）。
#: 顺序敏感：先匹配到的先用（"下午茶"必须排在"奶茶"之前）。
_SCENE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "下午茶": ("咖啡", "奶茶", "甜品"),
    "甜品": ("甜品", "蛋糕"),
    "咖啡": ("咖啡",),
    "奶茶": ("奶茶",),
    "午餐": ("美食", "餐厅"),
    "午饭": ("美食", "餐厅"),
    "晚餐": ("美食", "餐厅"),
    "吃饭": ("美食", "餐厅"),
    "夜宵": ("烧烤", "美食"),
    "火锅": ("火锅",),
    "烧烤": ("烧烤",),
    "电影": ("电影院",),
    "看展": ("展览",),
    "演出": ("演出",),
    "KTV": ("KTV",),
    "桌游": ("桌游",),
    "密室": ("密室",),
    "逛街": ("购物",),
    "购物": ("购物",),
    "商场": ("购物",),
    "散步": ("公园",),
    "公园": ("公园",),
    "景点": ("景点",),
    # 口语活动词 → 真实类型名（`bug/BUG-20261012-plan-scene-keywords-missing-activity-types.md`）：
    # 这批等价原先只有入口模型认得，而计划轨取候选只查这张表——命中不了就把口语词原样当搜索词 ⇒ 该类型永远取不到（真机"唱歌/做发型"两站拿不到对口店）。
    # 一律追加在末尾：`_scene_search_terms` 取"先匹配到的键"，插到前面会抢走既有键（如"下午茶"必须排在"奶茶"之前）。
    "唱歌": ("KTV",),
    "k歌": ("KTV",),
    "发型": ("丽人·美发",),
    "头发": ("丽人·美发",),
    "理发": ("丽人·美发",),
    "美发": ("丽人·美发",),
    "美甲": ("美睫·美甲",),
    "美睫": ("美睫·美甲",),
    "推拿": ("按摩·足疗",),
    "足疗": ("按摩·足疗",),
    "美容": ("美容SPA",),
    "spa": ("美容SPA",),
    "遛娃": ("亲子游乐",),
    "儿童乐园": ("亲子游乐",),
    "轰趴": ("轰趴馆",),
    "健身": ("健身运动",),
    "瑜伽": ("健身运动",),
    "酒吧": ("酒吧",),
    "小酒馆": ("酒吧",),
}

#: `第N站换一家` 的确定性触发（`specs/0012` §12.5）。**只认显式的"第N站"** ——
#: 本期不做多站替换、不做全局重排，模糊匹配会把"换一家对比"这类误判成替换。
_REPLACE_STOP = re.compile(r"第([一二两三四五1-5])站")


def _replace_stop_seq(query: str, stops: list) -> int | None:
    """query 是否要求替换某一站；返回站号（从 1 起）；否则 `None`（纯函数）。"""
    if not stops:
        return None
    match = _REPLACE_STOP.search(query or "")
    if not match:
        return None
    idx = _ORDINAL.get(match.group(1))
    if idx is None:
        return None
    seq = idx + 1
    return seq if 1 <= seq <= len(stops) else None


def _scene_search_terms(scene: str) -> tuple[str, ...]:
    """场景 → 依次尝试的搜索词（先查映射表，再回退成场景词本身）。
    匹配大小写不敏感（`KTV`/`ktv`、`K歌`/`k歌`、`spa`/`SPA` 都要认）；回退时仍返回场景词原样（保留大小写），否则 `_scene_candidates` 拿不到用户原始说法。
    """
    text = str(scene or "").strip()
    lowered = text.lower()
    for key, terms in _SCENE_KEYWORDS.items():
        if key.lower() in lowered:
            return terms
    return (text,) if text else ()


def _scene_terms_for(
    scene: str, scene_type: str, type_pool: list[str]
) -> tuple[str, ...] | None:
    """场景词 → 依次尝试的搜索词（**纯函数**，`specs/0031` T4 的三级解析）：① `scene_type` 非空且在真实类型池内 ⇒ `(scene_type,)`（零额外调用）；② 否则 `_scene_search_terms(scene)` 命中字典（结果不等于原词）⇒ 用它（零成本快路径）；③ 都没有 ⇒ 返回 `None`（信号：调用方走慢路径，失败再退回按原词搜）。
    ② 的判据是"字典结果 != (原词,)"，这样调用方才能区分"字典已经给了答案"与"字典帮不上忙"。
    """
    text = str(scene or "").strip()
    if not text:
        return ()
    st = str(scene_type or "").strip()
    if st and st in _pool_of(type_pool):
        return (st,)
    terms = _scene_search_terms(text)
    if terms != (text,):
        return terms
    return None


#: `ENTRY_SYSTEM` 已声明等价、但 `_SCENE_KEYWORDS` / `_PLAN_SCENE_GROUPS` 两处真源都没覆盖的口语活动词 → **归一后的真实类型**：真机多活动问句（"吃饭 → 唱歌 → 做发型"）用的正是这些口语，而 recommend 轨只装得下一个 `shop_type` ⇒ 会被静默丢弃（KTV 明明有店却没查过）。
#: 两处共用：① 提升判据把它并入活动词表；② "未覆盖"判定用它做覆盖比较。只增这两族；新词先登记缺口、别在这里堆表。
_ACTIVITY_ALIAS_TYPE: dict[str, str] = {
    "唱歌": "KTV", "k歌": "KTV",
    "发型": "丽人·美发", "头发": "丽人·美发", "理发": "丽人·美发", "造型": "丽人·美发",
}
_PROMOTION_ACTIVITY_ALIASES: tuple[str, ...] = tuple(_ACTIVITY_ALIAS_TYPE)

#: "两家比价"（E3）的**确定性**辅助判据。`ENTRY_SYSTEM` 里 E3 的例子是"A 和 B 哪家更便宜"；
#: 模型的 `sub_intents` 已会带 `compare`，这里只作纯函数侧的兜底（单测不依赖模型）。
#: 刻意**不含**"还是"——"火锅还是烧烤"不是比价，而是并列的双品类。
_PROMOTION_COMPARE = re.compile(r"哪家|哪一家|哪一个|比较|对比|vs", re.IGNORECASE)


def _activity_terms() -> frozenset[str]:
    """提升/未覆盖判定用的**活动/品类词**词表（纯静态，零网络零模型）。
    真源三处拼接（`specs/0029` §4「复用既有真源、不新造手写大表」）：① `_SCENE_KEYWORDS` 的 key 与 value；② `_PLAN_SCENE_GROUPS` 的全部场景词；③ `_PROMOTION_ACTIVITY_ALIASES`（极少量动词词组）。
    """
    terms: set[str] = set(_PROMOTION_ACTIVITY_ALIASES)
    for key, vals in _SCENE_KEYWORDS.items():
        terms.add(key)
        terms.update(vals)
    for _, kws in _PLAN_SCENE_GROUPS:
        terms.update(kws)
    return frozenset(t for t in terms if t)


def _activity_terms_in(query: str, vocab: frozenset[str] | None = None) -> tuple[str, ...]:
    """`query` 里命中的**活动词**（去重、排序稳定；**纯函数**）。
    统一 `.lower()` 后子串匹配（`_PLAN_SCENE_GROUPS` 里含小写 `"ktv"`，而用户可能写 `KTV`/`ktv`/`K歌`）。
    """
    q = str(query or "").lower()
    if not q:
        return ()
    words = vocab if vocab is not None else _activity_terms()
    return tuple(sorted(t for t in words if t and t.lower() in q))


def _should_promote_to_plan(query: str, plan: RoutePlan | None) -> tuple[str, ...]:
    """多活动问句的**确定性提升**判据（纯函数，零模型零网络；`specs/0029` §2 目标 1）。
    返回命中的活动词（空 = 不提升；非空 = 应改判 plan）以便复现；命中需同时满足：① `plan.scene == "recommend"`；② 开关 `AGENT_PLAN_ENABLED` 为真；③ 未点名店铺；④ 非两家比价（E3）；⑤ 命中 ≥2 个活动/品类词；⑥ 出现衔接词或区域词；保守原则：宁可漏提升，不可误提升。
    """
    if plan is None or plan.scene != "recommend":
        return ()
    if not _env_flag("AGENT_PLAN_ENABLED", False):
        return ()
    if plan.shop_names:
        return ()
    if "compare" in (plan.sub_intents or []) or _PROMOTION_COMPARE.search(str(query or "")):
        return ()
    hits = _activity_terms_in(query)
    if len(hits) < 2:
        return ()
    q = str(query or "").lower()
    if not (_PLAN_TIME_SEQ.search(q) or _PLAN_AREA.search(q)):
        return ()
    return hits


#: 汇总层的**未覆盖**标注（`specs/0029` §2 目标 3）：把"**没查**"与"**没有**"分开。
#: 真机踩过：正文答"暂时没有KTV唱歌和做发型相关的店铺信息"，而 KTV 只是**从未被查询**。
UNCOVERED_ACTIVITIES_NOTE = (
    "【未覆盖】以下活动本轮尚未查询：{terms}"
    "（这是**我们还没查**、不是「没有」——请如实说明「还没查 / 暂未查询」，禁止说成「没有 / 不存在」）。"
)


def _uncovered_activities(query: str, plan: RoutePlan | None) -> tuple[str, ...]:
    """检出"≥2 活动词但本轮**没进 plan**"时，**被结构性丢弃**的活动词（纯函数）：直接消灭"把没查说成没有"——真机正文曾答"暂时没有KTV唱歌和做发型相关的店铺信息"，而 KTV 只是从未被查询。
    判定刻意保守 = 只认"结构性装不下"的口语活动词（`_ACTIVITY_ALIAS_TYPE` 的 key，recommend 轨只按一个 `plan.shop_type` 取数 ⇒ 类型不是它就一定没查）；其余活动词无法确定性判断覆盖，宁可漏报也不误报；只在"命中 ≥2 个活动词 ∧ 未进 plan"时返回。
    """
    if plan is not None and plan.scene == "plan":
        return ()
    out: list[str] = []
    # `specs/0030`：多活动**被上限截断**的那些（`AGENT_NEARBY_MAX_TYPES`）**一次都没查** ⇒ 也算"未覆盖"。
    _targets, truncated = _resolve_type_targets(plan)
    out.extend(truncated)
    terms = _activity_terms_in(query)
    if len(terms) >= 2:
        covered = str((plan.shop_type if plan else "") or "").strip()
        for t in terms:
            mapped = _ACTIVITY_ALIAS_TYPE.get(t)
            if not mapped:                 # 无法确定性判断覆盖 ⇒ 保守不标注
                continue
            if covered and mapped == covered:
                continue
            out.append(t)
    # 去重保序（截断项在前，保持"最该说未查的"先出现）
    seen: set[str] = set()
    deduped: list[str] = []
    for term in out:
        if term and term not in seen:
            seen.add(term)
            deduped.append(term)
    return tuple(deduped)



#: 口语品类词 → **真实店铺类型名**（模型归类，见 `_map_category_by_llm`）。
#: 历史包袱（`plans/0019` 曾手写 `_CATEGORY_TO_TYPE`，40 余条）：DB 类型只有 10 个，用户口语「火锅/烧烤/日料…」没有一个是类型名，不映射就会把"问错了词"伪装成"附近没有店"；归类是开放集问题，交给模型，代码只做池内白名单校验。
#: 与 `_SCENE_KEYWORDS` 的区别：那份是计划轨的"场景→搜索关键词"（可含非类型词），这份的值必须能通过 `fetch_shops_nearby` 的类型名匹配。
class CategoryMatch(BaseModel):
    """口语品类 → 真实类型的**归类**结果：`extra="ignore"` 而非 `"forbid"`——池外类型由 `_sanitize_mapped_types` 显式丢弃（要可观测的降级，而不是整条校验失败把归类打回）。
    字段名只在提示词里描述、此处保持极简——归类清单每次随 `type_pool` 变化，写进 schema 反而多一份要同步的地方。
    """

    model_config = ConfigDict(extra="ignore")

    type_names: list[str] = Field(
        default_factory=list,
        description="匹配到的真实店铺类型名，按贴合度从高到低；清单里都不沾边则留空",
    )


class SubcategoryPick(BaseModel):
    """细分品类筛选结果：候选里**真正满足**用户品类要求的店铺 id。"""

    model_config = ConfigDict(extra="ignore")

    ids: list[int] = Field(
        default_factory=list,
        description="满足品类要求的候选店铺 id；一家都不满足则留空",
    )
    same_type: bool = Field(
        default=False,
        description=(
            "用户要找的品类是否**就是该店铺类型的泛称/等价说法**"
            "（如「饭店」≈「美食」）；是则 `ids` 留空并置 true"
        ),
    )


def _is_generic_type_word(raw: str, type_name: str) -> bool:
    """`raw` 是不是 `type_name` 的**泛指等价词**（"饭店/餐厅/餐馆" ≈ "美食"）？（纯函数）
    需要它（`bug/BUG-20261013-nearby-generic-type-word-vetoed`）：泛指同义词字面不等会走否决 ⇒ 轻量模型判"这 5 家都不算饭店"⇒ 整批被拒（真机「有什么饭店吗」答没有，而「美食」有 5 家）；泛指词不比归类词更细 ⇒ 应等价于 `raw == type_name`，不筛不否决；真源两处：① `_TYPE_NAME_KEYWORDS` 别名表；② `_scene_search_terms(raw)` 含 `type_name`。
    """
    if not raw or not type_name or raw == type_name:
        return False
    if raw in _TYPE_NAME_KEYWORDS.get(type_name, ()):
        return True
    return type_name in _scene_search_terms(raw)


async def _narrow_by_subcategory(
    shops: list, type_name: str, type_pool: list[str], mapped: tuple[str, ...],
    user_id: str = "", raw: str = "",
) -> list:
    """按**细分品类**收窄候选（如"日料" vs 「美食」下的 5 家店）：优先用用户原词 `raw` 筛（`specs/0028`），没原词才退 `type_name`——入口模型会把"日料"归类成「美食」，拿归类词筛等于"用户要的就是整个类型"，5 家全放行（正是本缺陷）。
    只在"筛选词比归类更细"时才筛（`raw == type_name`/`raw` 是真实类型/泛指等价词 ⇒ 不筛；无 `raw` 且 `mapped` 空 ⇒ 不筛）；可否决边界保守：仅有 `raw` 且 `raw != type_name` 时筛空才返回 `[]`；`same_type=true`（`specs/0031` Q3）同样放行全量。
    """
    text = str(raw or "").strip()
    if text:
        # 新路径：原词在手 ⇒ 是否"更细"由原词与归类词比对决定，**不依赖 `mapped`**
        # （ES 命中时会短路归类模型、`mapped=()`，此时原词仍是有效的筛选依据）。
        # 泛指等价词（"饭店"≈"美食"）也走这条短路：它不比归类词更细 ⇒ 不该筛、更不该否决（`bug/BUG-20261013-nearby-generic-type-word-vetoed`）。
        if text == type_name or text in _pool_of(type_pool) or _is_generic_type_word(text, type_name):
            return shops
        term, vetoable = text, True
    else:
        # 旧路径：无原词 ⇒ 保持改造前判据（无归类结果则放行）
        if not mapped:
            return shops
        term, vetoable = str(type_name or "").strip(), False
    if not shops or not term or term in _pool_of(type_pool):
        return shops
    lines = "\n".join(
        f"  - id={_shop_id_of(s)} 店名={s.get('name')}" for s in shops if _shop_id_of(s) is not None
    )
    if not lines:
        return shops
    human = f"用户要找的品类：{term}\n候选店铺：\n{lines}"
    try:
        record_model_call("subcategory_filter")
        result = await structured(get_light_model(), SubcategoryPick).ainvoke(
            [SystemMessage(SUBCATEGORY_FILTER_SYSTEM), HumanMessage(human)]
        )
    except Exception as exc:
        # 筛选失败**保留全量候选**（宁可不筛，也不要把已查到的店丢掉）
        logger.warning("细分品类筛选失败（保留全量候选）: %s", exc)
        return shops
    keep = {
        int(i)
        for i in (getattr(result, "ids", None) or [])
        if isinstance(i, int) or (isinstance(i, str) and i.strip().isdigit())
    }
    if not keep:
        # 模型判定"用户要找的就是这个类型的**泛称/等价说法**"（`specs/0031` Q3）⇒ 放行全量、不否决。
        #    这是"泛指词"的**治本**判据：不依赖 `_is_generic_type_word` 的别名表能否覆盖（开放集问题），
        #    由**既有的这次**筛选调用顺带给出。`_is_generic_type_word` 保留为命中即短路的**快路径**。
        if getattr(result, "same_type", False):
            logger.info("细分品类筛选：%s 判定为该类型的泛称/等价说法 ⇒ 放行全量", term)
            return shops
        if vetoable:
            # 原词比归类更细且筛空 ⇒ **否决**：库里这个细分品类一家都不沾边，
            # 再放行 5 家美食店就是把"不相关"当"推荐"（`specs/0028` 目标 3）。
            logger.info("细分品类筛选：%s 在 %d 家候选里无匹配 ⇒ 否决全部（不发不相关卡片）", term, len(shops))
            return []
        logger.info("细分品类筛选：%s 在 %d 家候选里无匹配，保留全量交由汇总层如实转达", term, len(shops))
        return shops
    narrowed = [s for s in shops if _shop_id_of(s) in keep]
    logger.info("细分品类筛选：%s %d -> %d 家", term, len(shops), len(narrowed))
    return narrowed or shops


def _pool_of(type_pool: list[str]) -> set[str]:
    return {str(p).strip() for p in type_pool if str(p or "").strip()}


def _sanitize_mapped_types(names: Any, type_pool: list[str]) -> tuple[str, ...]:
    """模型归类结果 → 池内真实类型名（去重、保序）。
    池外一律丢弃：模型可能返回清单里没有的词，拿它去 `fetch_shops_nearby` 只会得到「暂时没有「X」这一类店铺」（把配置问题说成业务为空）；丢弃后退回"按名搜"回退，行为不丢。
    """
    pool = _pool_of(type_pool)
    out: list[str] = []
    for raw in (names or []) if isinstance(names, (list, tuple)) else []:
        name = str(raw or "").strip()
        if name and name in pool and name not in out:
            out.append(name)
    return tuple(out)


# ---- 细分品类/食材召回：经 search-service 用博客内容反查店铺（`specs/0026` / `plans/0041`）----
def _blog_recall_size() -> int:
    """博客反查时透传给 search-service 的 `size`（命中博客条数上限）。默认 5。**调用时读 env。**"""
    try:
        return max(1, int(os.getenv("AGENT_BLOG_RECALL_SIZE", "5")))
    except (TypeError, ValueError):
        return 5


def _blog_recall_max() -> int:
    """单轮最多补全的店铺数上限（防止命中很多 `shopId` 时逐条详情把下游打满）。默认 5。**调用时读 env。**"""
    try:
        return max(1, int(os.getenv("AGENT_BLOG_RECALL_MAX", "5")))
    except (TypeError, ValueError):
        return 5


def _is_fine_grained_term(term: str, type_pool: list[str]) -> bool:
    """判断 `term` 是否是**细分商品/菜品/食材词**（而非真实店铺类型名）：确定性规则（不调模型——保证单测零联网、判据可复现），空串或命中真实类型清单（`type_pool`，这个维度已搜过且为空）→ False，其余 → True。
    真正的粗筛是合取里的"原链路召回为空"，本函数只需排除"已搜过的真实类型"（`specs/0026` §2 目标 2）。
    """
    text = str(term or "").strip()
    if not text:
        return False
    return text not in _pool_of(type_pool)


async def _blog_recall_ids(term: str, type_pool: list[str], user_id: str = "") -> list[int]:
    """细分词补充召回：**判定 + 经 search-service 博客反查**，返回命中的店铺 id（去重保序、已截断）；触发是合取：细分词 ∧ 开关开（`AGENT_BLOG_RECALL_ENABLED`，默认开）。
    全程**不外抛**：开关关/词空/非细分词/无命中/search-service 异常 ⇒ 一律返回 `[]`，调用方保持原有"业务为空"分流，主链路不受影响（`specs/0026` §2 目标 6 降级）。
    """
    if not _env_flag("AGENT_BLOG_RECALL_ENABLED", True):
        return []
    if not _is_fine_grained_term(term, type_pool):
        return []
    text = str(term or "").strip()
    if not text:
        return []

    res = await asyncio.to_thread(search_blogs_by_key, text, _blog_recall_size())
    if not isinstance(res, list):
        # 工具按 `list | str` 契约在技术性失败时返回**用户向文案** ⇒ 降级回原链路，绝不外抛
        logger.warning("博客反查店铺失败（降级回原链路）：term=%s", text)
        return []

    seen: set[int] = set()
    out: list[int] = []
    for sid in res:
        if isinstance(sid, int) and sid not in seen:
            seen.add(sid)
            out.append(sid)
    max_n = _blog_recall_max()
    if len(out) > max_n:
        logger.info("博客反查店铺命中 %s 家，按上限 %s 截断", len(out), max_n)
        out = out[:max_n]
    return out


async def _blog_recall_details(shop_ids: list[int], user_id: str = "") -> list[dict]:
    """把博客反查到的 `shop_ids` 逐条补**店铺详情**（≤ `AGENT_BLOG_RECALL_MAX`），产出候选 dict。
    无批量接口（shop-service 仅有 `GET /shop/{id}`）⇒ 逐条 `fetch_shop_detail`，但受并发许可门 `AGENT_FANOUT_CONCURRENCY` 约束，突发时不打满 shop-service（`specs/0026` R2）；任一店失败/不存在（返回 str）只丢弃该条，绝不让整批召回失败。
    """
    if not shop_ids:
        return []

    async def _one(sid: int) -> dict | None:
        async with _fanout_gate():
            detail = await asyncio.to_thread(fetch_shop_detail, sid, user_id)
        # 取数失败/不存在时 fetch_shop_detail 返回**字符串** ⇒ 丢弃该条
        return detail if isinstance(detail, dict) else None

    results = await asyncio.gather(*[_one(sid) for sid in shop_ids], return_exceptions=True)
    out = [r for r in results if isinstance(r, dict)]
    if out:
        logger.info("博客反查店铺补召回 %d 家（命中 %d 个 id）", len(out), len(shop_ids))
    return out


def _merge_shop_candidates(primary: Any, extra: Any) -> list[dict]:
    """把 ES 补召回结果**并入**主链路候选：按 `shop_id` 去重、**保序**（纯函数，`specs/0026` v2）。
    并入而非替换（规格 §2 目标 2/4）：ES 结果追加在未出现过的 id 上，不前置/后置/重排——排序仍由下游既有逻辑统一做；保序是硬要求：主链路候选序不得被并入打乱（`plans/0045` R1）。
    """
    out: list[dict] = []
    seen: set[int] = set()
    for shop in list(primary or []) + list(extra or []):
        if not isinstance(shop, dict):
            continue
        sid = _shop_id_of(shop)
        if sid is None or sid in seen:
            continue
        seen.add(sid)
        out.append(shop)
    return out


async def _map_category_by_llm(
    type_name: str, type_pool: list[str], user_id: str = ""
) -> tuple[str, ...]:
    """口语品类词 → 真实类型名候选（**模型归类**；失败返回 `()`，调用方退原词）；原词本身已是真实类型（如"KTV"/"美食"）时不调模型——省一次 LLM 且没有归错的风险。"""
    text = str(type_name or "").strip()
    if not text or not _pool_of(type_pool):
        return ()
    if text in _pool_of(type_pool):
        return (text,)

    human = (
        f"真实店铺类型清单：{'、'.join(sorted(_pool_of(type_pool)))}\n"
        f"用户的说法：{text}"
    )
    try:
        record_model_call("category_map")
        result = await structured(get_light_model(), CategoryMatch).ainvoke(
            [SystemMessage(CATEGORY_CLASSIFY_SYSTEM), HumanMessage(human)]
        )
    except Exception as exc:
        # 归类失败**不能**让整轮失败：调用方会退原词 + 按名搜，行为与"没做映射"一致。
        logger.warning("品类归类失败（按原词回退）: %s", exc)
        return ()
    names = getattr(result, "type_names", None) if result is not None else None
    mapped = _sanitize_mapped_types(names, type_pool)
    logger.info("品类归类：%s -> %s", text, mapped or "（无匹配，退回按名搜）")
    return mapped


def _type_search_terms(
    type_name: str, type_pool: list[str], mapped: tuple[str, ...] = ()
) -> tuple[str, ...]:
    """`type_name` → 依次尝试的**真实类型名**候选（纯函数，可测）；每项都必须在 `type_pool`（后端真实类型列表）里，否则被过滤——归类结果与真实类型表不同步时退回原词，而不是查一个不存在的类型（把配置问题说成业务为空）。
    :param type_name: 用户/模型给的品类词；:param type_pool: 后端真实类型名列表（`fetch_shop_types` 结果）；:param mapped: `_map_category_by_llm` 归类结果；:returns: 至少含原词。
    """
    text = str(type_name or "").strip()
    if not text:
        return ()
    pool = _pool_of(type_pool)

    # 原词本身就是真实类型（如"KTV"/"美食"）⇒ 不用归类
    if text in pool:
        return (text,)

    # 归类结果在前（原词保底在最后）：归类词都查不到时，仍按原词走原有的回退/报错逻辑
    terms: list[str] = [t for t in mapped if t in pool and t != text]
    if terms:
        return tuple(terms) + (text,)
    return (text,)


def _start_minutes(time_text: str) -> int:
    """起始时刻（分钟）：由 `slots.time` 的**口语词**推导，确定性、可测。
    刻意不解析"今天下午 3 点"这类自由文本里的具体钟点——解析错了会给出假精确的时段；宁可给符合时段的整点，用户可自行调整。
    """
    text = str(time_text or "")
    for word, minute in (
        ("早上", 9 * 60), ("上午", 10 * 60), ("中午", 12 * 60),
        ("下午", 14 * 60), ("傍晚", 17 * 60), ("晚上", 18 * 60), ("夜里", 19 * 60),
    ):
        if word in text:
            return minute
    return 14 * 60


def _hhmm(minutes: int) -> str:
    m = max(0, int(minutes))
    return f"{m // 60:02d}:{m % 60:02d}"


def _straight_distance_m(a: dict, b: dict) -> int | None:
    """两点的**直线距离**（米）；任一缺坐标返回 `None`。
    本期刻意用直线（`specs/0012` §12.1）：真实路网距离由前端地图 SDK 画，后端不做地理编码；字段名 `distance_m`（不叫 `route_distance`），避免被当成真实路线距离呈现（R7）。
    """
    try:
        x1, y1 = float(a.get("x")), float(a.get("y"))
        x2, y2 = float(b.get("x")), float(b.get("y"))
    except (TypeError, ValueError):
        return None
    dy = (y2 - y1) * 110_540.0
    dx = (x2 - x1) * 111_320.0 * math.cos(math.radians((y1 + y2) / 2))
    return int(math.hypot(dx, dy))


def _compose_legs(stops: list[dict]) -> list[dict]:
    """相邻站的接驳段（数量恒为 `len(stops) - 1`）。"""
    legs: list[dict] = []
    for prev, cur in zip(stops, stops[1:]):
        dist = _straight_distance_m(prev, cur)
        legs.append(
            {
                "from": prev["seq"],
                "to": cur["seq"],
                "mode": "walk",
                "distance_m": dist,
                "duration_min": None if dist is None else max(1, round(dist / PLAN_WALK_M_PER_MIN)),
            }
        )
    return legs


async def _scene_candidates(
    scene: str, x, y, user_id: str, limit: int,
    *, with_rag: bool = True, deadline: float | None = None,
    scene_type: str = "", type_pool: list[str] | None = None,
) -> list[dict]:
    """按场景取候选店，并补**详情 + 券 +（可选）细节检索**，产出「一店一条」记录；复用现有 `fetch_*` 不新增取数实现；`rag` 可选且吃余量（`plans/0020` §15.2）：细节检索带 Cross-Encoder 重排、是全链路唯一秒级阻塞的取数块，只吃 `deadline` 余量，超时/失败只降 `chunks` 不连坐 detail/vouchers。
    三级解析（`specs/0031` T4）：`scene_type`（池内）→ 字典快路径 → 慢路径调 `_map_category_by_llm` 兜底，失败退原词，静态字典降级为"快捷方式"。
    """
    terms = _scene_terms_for(scene, scene_type, type_pool or [])
    if terms is None:
        # ③ 慢路径：字典没命中 ⇒ 归类模型兜底。失败/池空 ⇒ 退回按**原词**搜（行为与改造前一致）。
        mapped = await _map_category_by_llm(scene, type_pool or [], user_id)
        terms = mapped or (str(scene or "").strip(),)
    shops: Any = []
    for term in terms:
        res = await asyncio.to_thread(fetch_shops_nearby, term, x, y, 1, user_id)
        if not isinstance(res, list) or not res:
            # 与 `nearby_worker` 同款回退：`/shop/of/type` 对部分类型返回空，而按名称能命中
            res = await asyncio.to_thread(fetch_shops_by_name, term, 1, user_id)
        if isinstance(res, list) and res:
            shops = res
            break
    # 补充召回（`specs/0026` **v2**）：场景词（如"手冲咖啡"）**并行并入**——不再要求主链路为空；
    #    计划轨没有真实类型池可对照 ⇒ 传空池；开关关/未命中/search-service 失败 ⇒ 原样返回（零副作用）。
    #    并入点必须在 `_enrich` 之前（`plans/0045` R4）：这里只给 `{"id": …}`，详情/券/检索块由 `_enrich` 补 ⇒ 同一家店不会被取两次。
    recall_ids = await _blog_recall_ids(scene, (), user_id)
    if recall_ids:
        shops = _merge_shop_candidates(
            shops if isinstance(shops, list) else [], [{"id": sid} for sid in recall_ids]
        )
    if not isinstance(shops, list) or not shops:
        return []

    picked = [s for s in shops if isinstance(s, dict) and _shop_id_of(s) is not None][:limit]

    async def _enrich(shop: dict) -> dict:
        sid = _shop_id_of(shop)
        detail, vouchers = await asyncio.gather(
            asyncio.to_thread(fetch_shop_detail, sid, user_id),
            asyncio.to_thread(fetch_shop_vouchers, sid, user_id),
            return_exceptions=True,
        )
        detail = detail if isinstance(detail, dict) else {}
        if isinstance(vouchers, BaseException):
            vouchers = ""
        vlist, verr = _split_list_or_error(vouchers)
        name = detail.get("name") or shop.get("name")

        chunks: list = []
        chunks_err = ""
        if with_rag:
            budget = _remaining(deadline, RAG_TIMEOUT)
            if budget <= 0:
                chunks_err = f"{RAG_TIMEOUT_MSG}（未获预算）"
            else:
                try:
                    got = await asyncio.wait_for(
                        asyncio.to_thread(fetch_rag_chunks, sid, scene, RAG_TOP_K, deadline),
                        timeout=budget,
                    )
                    chunks, chunks_err = _split_list_or_error(got or [])
                except asyncio.TimeoutError:
                    chunks_err = f"{RAG_TIMEOUT_MSG}（{budget:.1f}s）"
                except Exception as exc:  # noqa: BLE001 检索失败不得影响行程本体
                    logger.warning("计划轨细节检索失败 shop_id=%s（只降 chunks）: %s", sid, exc)
                    chunks_err = f"细节检索失败：{exc}"

        return _new_shop_record(
            sid,
            name,
            tasks=("detail", "voucher", "rag") if with_rag else ("detail", "voucher"),
            # 详情里的坐标/地址优先，缺了再用候选列表里的（两者都来自真实接口）
            **_shop_dims({**shop, **detail}),
            vouchers=vlist,
            vouchers_error=verr,
            chunks=chunks,
            chunks_error=chunks_err,
            order_cards=[prepare_order_card(v, sid) for v in vlist] if vlist else [],
        )

    return list(await asyncio.gather(*[_enrich(s) for s in picked]))


def _scene_type_pairs(slots: dict) -> list[tuple[str, str]]:
    """从草稿槽位取「场景词 → 归类类型」**配对**列表（纯函数，`specs/0031` T3/T4）。
    `scene_types` 与 `scenes` 等长时逐项配对，否则一律视作无归类（配空串）⇒ 行为退回改造前；长度不等绝不按前缀强行配对（错位比无归类更糟），截断由调用方对配对列表整体切片。
    """
    scenes = [
        str(s).strip() for s in (slots.get("scenes") or [])
        if isinstance(s, str) and str(s).strip()
    ]
    types = aligned_scene_types(slots)
    if len(types) != len(scenes):
        types = [""] * len(scenes)
    return list(zip(scenes, types))


async def plan_worker(state: HmdpState) -> dict:
    """计划轨取数：按 `task_draft.slots.scenes`**并发**取候选（每场景一条输出）；不用 `Send`（`specs/0012` §2 目标 3 的 v3 修正）：`Send` 只能由条件边发出，复用 nearby/shop_worker 会让两条轨出边打架。
    本节点在内部并发调同一批 `fetch_*`，取数实现零新增。
    """
    record_node("plan_worker")
    draft = state.get("task_draft") or {}
    slots = draft.get("slots") or {}
    # `scene_types` 与 `scenes` 必须**同步截断**（`specs/0031` T3）：对**配对列表**整体切片，
    #    不能分别切片（那会错位）。无 `scene_types` / 不等长 ⇒ 配空串（走字典快路径 + 归类兜底）。
    pairs = _scene_type_pairs(slots)[:PLAN_MAX_SCENES]
    scenes = [s for s, _ in pairs]
    user_id = state.get("user_id") or ""
    x, y = state.get("x"), state.get("y")

    if not scenes:
        return {
            "worker_outputs": [
                {"worker": "plan", "scene": "", "items": [],
                 "empty": "还没听出你想做什么，能再说说吗？比如先喝个下午茶再吃晚饭。"}
            ]
        }

    # 真实类型池：`_scene_terms_for` 的池内校验与归类兜底都要用 ⇒ **取一次**（`specs/0031` Q1(a)）。
    #    取不到（返回字符串/空）⇒ 池为空 ⇒ ① 跳过、慢路径不调模型（`_map_category_by_llm` 内建短路）。
    known_types = await asyncio.to_thread(fetch_shop_types, user_id)
    type_pool = (
        [str(t.get("name") or "") for t in known_types if isinstance(t, dict)]
        if isinstance(known_types, list)
        else []
    )

    # 一个 deadline 贯穿两层（`specs/0010` §3）：整条计划轨取数吃 `WORKER_TIMEOUT`，
    #    每站的细节检索只吃**余量**（`_remaining(deadline, RAG_TIMEOUT)`）。
    plan_deadline = time.monotonic() + WORKER_TIMEOUT
    results = await asyncio.gather(
        *[
            _scene_candidates(sc, x, y, user_id, PLAN_CANDIDATES_PER_SCENE,
                              with_rag=True, deadline=plan_deadline,
                              scene_type=st, type_pool=type_pool)
            for sc, st in pairs
        ],
        return_exceptions=True,
    )
    outputs: list[dict] = []
    for scene, res in zip(scenes, results):
        if isinstance(res, BaseException):
            # 取数失败 ⇒ **如实**记缺口，绝不用别的场景的店凑数（T1 的"失败时怎么做"）
            logger.warning("计划轨取数失败（场景=%s）: %s", scene, res)
            items: list[dict] = []
        else:
            items = list(res or [])
        entry: dict[str, Any] = {
            "worker": "plan", "scene": scene, "type_name": scene, "items": items,
        }
        if not items:
            entry["empty"] = f"「{scene}」暂时没查到合适的店"
        outputs.append(entry)

    logger.info("计划轨取候选：场景=%s 命中=%s", scenes, [len(o["items"]) for o in outputs])
    return {"worker_outputs": outputs}


def _plan_groups(state: HmdpState) -> list[tuple[str, list[dict]]]:
    """取回「场景 → 候选记录」分组（只认 `worker == "plan"` 的输出）。"""
    groups: list[tuple[str, list[dict]]] = []
    for wo in state.get("worker_outputs") or []:
        if isinstance(wo, dict) and wo.get("worker") == "plan":
            groups.append(
                (str(wo.get("scene") or ""),
                 [r for r in (wo.get("items") or []) if isinstance(r, dict)])
            )
    return groups


def _index_records(groups: list[tuple[str, list[dict]]]) -> dict[int, dict]:
    """候选白名单：`shop_id → 记录`。行程里的每一站都必须能在这里找到（R1 防幻觉门店）。"""
    index: dict[int, dict] = {}
    for _, records in groups:
        for rec in records:
            sid = _shop_id_of(rec)
            if sid is not None and sid not in index:
                index[sid] = rec
    return index


class ItineraryPlan(BaseModel):
    """模型编排结果：**只有"挑哪几家、什么顺序、标题"**，事实字段一律由代码回填。"""

    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    stops: list[dict] = Field(default_factory=list)


#: 偏好维度 → 中文准则（`specs/0024`）：plan 轨排序准则行与 recommend 轨提示词共用同一份文案。
_PREFERENCE_LABELS: dict[str, str] = {
    "price": "人均价格低",
    "score": "评分高",
    "distance": "距离近",
    "sold": "销量高",
    # ---- 反向（`specs/00xx` 问题 B）：软偏好同样可能出现"最贵/最远"这类说法 ----
    "price_desc": "人均价格高（最贵）",
    "score_asc": "评分低",
    "distance_desc": "距离远（最远）",
    "sold_asc": "销量低",
}


def _preference_dimensions(preferences: Any) -> list[str]:
    """把 `preferences` 抽成**去重保序**的合法维度列表（纯函数，两条轨共用）。
    入参兼容 `PreferenceItem` 对象、`[{"dimension": ...}]`（dict）与裸字符串数组；非法/缺失维度一律丢弃（与 `state._clean_preferences` 同口径，脏值不臆造排序）。
    """
    out: list[str] = []
    for item in preferences or []:
        if isinstance(item, dict):
            raw = item.get("dimension")
        elif hasattr(item, "dimension"):  # `PreferenceItem`（BaseModel）
            raw = getattr(item, "dimension")
        else:
            raw = item
        dim = str(raw or "").strip().lower()
        if dim in _PREFERENCE_LABELS and dim not in out:
            out.append(dim)
    return out


async def _ask_itinerary(
    groups: list[tuple[str, list[dict]]], slots: dict, max_stops: int,
    *, preferences: Any = (), query: str = "",
) -> ItineraryPlan | None:
    """让模型挑站与排序（**只给它候选白名单**）；失败返回 `None`，由规则兜底；`preferences`/`query`（`specs/0024` T2/#6/#8）把用户**软偏好**与**原话**显式带进 human 供定序（原先只看"时间/区域/人数/预算"，顺序靠模型自由裁量）。"""
    lines: list[str] = []
    for scene, records in groups:
        if not records:
            continue
        lines.append(f"【{scene}】")
        for rec in records:
            bits = [f"shop_id={_shop_id_of(rec)}", f"店名={rec.get('name')}"]
            if rec.get("avgPrice"):
                bits.append(f"人均={rec['avgPrice']}元")
            score = rec.get("score")
            if isinstance(score, (int, float)) and score:
                bits.append(f"评分={score / 10:.1f}")
            if rec.get("distance") is not None:
                bits.append(f"距离={rec['distance']}m")
            lines.append("  - " + " ".join(bits))
    if not lines:
        return None

    # 排序准则（`specs/0024` §2 目标 3）：把偏好维度按优先级显式写进 human，不让模型自由裁量。
    dims = _preference_dimensions(preferences)
    if dims:
        criteria = "；".join(f"{i}. {_PREFERENCE_LABELS[d]}" for i, d in enumerate(dims, 1))
        order_line = f"排序偏好（按优先级从高到低）：{criteria}。\n"
    else:
        order_line = "排序偏好：无（按你认为合理的顺序排）。\n"
    # 顺序证据（§2 目标 4 / #6）：附上用户原话，供"同一场景内多店"定序（场景名本身不含先后）。
    if str(query or "").strip():
        order_line += f"用户原话：{str(query).strip()}\n"

    human = (
        f"时间：{slots.get('time')}；区域：{slots.get('area')}；人数：{slots.get('people')}；"
        f"总预算：{slots.get('budget')} 元；最多 {max_stops} 站。\n"
        + order_line
        + "候选店铺：\n" + "\n".join(lines)
    )
    try:
        record_model_call("itinerary")
        # 走 **planner 模型**（`specs/0014` T1）：编排与执行可分离。
        #    `AGENT_PLANNER_MODEL` 留空 ⇒ `get_planner_model()` 就是轻量模型 ⇒ 行为与现状一致。
        result = await structured(get_planner_model(), ItineraryPlan).ainvoke(
            [SystemMessage(ITINERARY_SYSTEM), HumanMessage(human)]
        )
    except Exception as exc:
        logger.warning("行程编排模型失败（改用规则兜底）: %s", exc)
        return None
    return result if isinstance(result, ItineraryPlan) else None


def _activity_of(sid: int, groups: list[tuple[str, list[dict]]]) -> str:
    """某家店属于哪个场景（模型没给 `activity` 时用它回填）。"""
    for scene, records in groups:
        if any(_shop_id_of(r) == sid for r in records):
            return scene
    return ""


def _stop_reason(rec: dict, *, limit: int = 60) -> str:
    """给行程站生成一句**只含已知事实**的推荐理由（`plans/0020` §15.1）；**由代码生成而非只让模型写**：模型会把资料没有的也写进去（"环境好""性价比高"），而 `SUMMARY_SYSTEM` 要求只输出明确事实；代码先给**可核对**底稿（人均/评分/距离/有券），模型给更好 `reason` 可覆盖（见 `plans/0020` T3），覆盖不掉至少还有底稿。
    只用**已取回**字段；缺一个少说一个，绝不用默认值/推测值填充。"""
    bits: list[str] = []
    avg = rec.get("avgPrice")
    if avg:
        bits.append(f"人均{avg}元")
    score = rec.get("score")
    if isinstance(score, (int, float)) and score:
        bits.append(f"评分{score / 10:.1f}")
    dis = rec.get("distance")
    if isinstance(dis, (int, float)) and dis >= 0:
        bits.append(f"距离{dis}m")
    if rec.get("vouchers"):
        bits.append("有可用券")
    reason = "、".join(bits)
    if limit > 0 and len(reason) > limit:
        reason = reason[: max(0, limit - 1)] + "…"
    return reason


#: 每站最多带几条证据（`AGENT_ITINERARY_EVIDENCE_MAX`）—— 前端要逐条展示，太多反而没人看
ITINERARY_EVIDENCE_MAX = int(os.getenv("AGENT_ITINERARY_EVIDENCE_MAX", "2"))


def _evidence_int(value: Any) -> int | None:
    """把取回资料里的 id 归一化成 `int`（Java JSON 可能给字符串数字）；显式排除 `bool`（`isinstance(True, int)` 为真）与一切非数字—— id 是"跳转落地页"的键，错一个就跳到**别人的文章**（前端看正常只是内容不对）。"""
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _evidence_ref(item: dict, shop_id: int | None) -> tuple[int | None, str]:
    """evidence 的**跳转规则**（`plans/0025`）：返回 `(id, target)`；有博客 id⇒`(blog_id, "blog")` 打开该篇，没有而店铺 id 有⇒`(shop_id, "shop")` 打开店详情（退化仍有入口），连店铺 id 都没有⇒`(None, "")` 纯文本、**不伪装成可点**。
    `id` 一律来自**取回的资料**（`chunks`/`blogs` 是 `fetch_*` 返回值，模型只读正文连 id 都看不到），绝不来自模型输出。"""
    blog_id = _evidence_int(item.get("id"))
    if blog_id is not None:
        return blog_id, "blog"
    if shop_id is not None:
        return shop_id, "shop"
    # 不能返回 `(None, "shop")`：前端会照着 target 拼出 `#/shop/null`
    #    —— 一个"能点开但必然 404"的入口，比不可点更糟（测试 `test_no_id_at_all_is_plain_text` 抓的）。
    return None, ""


def _stop_evidence(rec: dict, *, limit: int | None = None, snippet_chars: int = 60) -> list[dict]:
    """从该店**取回的原文**里取可回溯的证据（`plans/0020` §15.1）；只放**取回来的原文片段**（前端可点回原文），**不放**模型转述（二次转述失真，"有依据"比"依据好看"重要）。
    素材口径与推荐理由**必须一致**（2026-10-08）：同用「检索块 `chunks` ∪ 攻略 `blogs`」，优先 `chunks`（与 query 更相关）空时回退 `blogs`；每条带 `id`+`target`（跳转规则 `plans/0025`），前端「推荐来源(N)」据此跳转（只跳规则不猜 id）。
    两份素材都空时返回空列表：「**没有证据**」与「**有证据**」必须能被下游/前端区分，否则把"没查到"当成"没有评价"。"""
    cap = ITINERARY_EVIDENCE_MAX if limit is None else limit
    source = list(rec.get("chunks") or [])
    if not source:
        source = list(rec.get("blogs") or [])
    shop_id = _shop_id_of(rec)
    out: list[dict] = []
    for c in source:
        if not isinstance(c, dict):
            continue
        text = (c.get("content") or "").strip()
        if not text:
            continue
        ref_id, target = _evidence_ref(c, shop_id)
        out.append({
            "kind": "blog",
            "title": c.get("title") or "",
            "snippet": (text[:snippet_chars] + "…") if len(text) > snippet_chars else text,
            "id": ref_id,
            "target": target,
        })
        if len(out) >= cap:
            break
    return out


def _merge_reason(base: str, model_reason: str, rec: dict, *, limit: int = 80) -> str:
    """把模型那句推荐理由**校验后**接到代码底稿后（`plans/0020` T3）；`SUMMARY_SYSTEM` 写"只输出明确事实"但提示词非保证，实测模型编"环境好/性价比高" ⇒ 直接放行展示编造语比没理由更糟。
    校验：①每数字须在 `base`（真取回的人均/评分/距离/有券）里对上（按数字边界比集合，非子串 `in`，2026-10-08 自测抓过 `30` 砍 `3` 误放）；②命中贬义/无依据套话（`_REASON_UNSUPPORTED_WORDS`）⇒整句丢；③超 `limit` 截断。
    净效果：模型只能"组织已有事实的表达"、不能引入新事实；校验不过退回底稿（安全降级）。`base` 空仍校验但跳过规则①，只用规则②③。"""
    text = (model_reason or "").strip()
    if not text:
        return base

    # 规则 2：套话黑名单（命中即整句丢，不做"只删这个词"的修补 —— 句子会变得不通顺）
    if any(w in text for w in _REASON_UNSUPPORTED_WORDS):
        return base

    # 规则 1：数字必须都在底稿里对上（2026-10-08 自测抓出：不能子串 `in`、不能 `rstrip("0")` 把 `30` 砍成 `3` 放行，真值 80；应抽底稿全部数字做集合比对，边界天然正确）。
    # `base` 为空时**跳过**这条：无底稿无从核对，照比会把每句都拒、等于把 `reason` 永久关掉。
    if base:
        said = _reason_numbers(text)
        known = _reason_numbers(base)
        if not said <= known:
            logger.info("行程推荐理由含底稿之外的数字 %s，已丢弃该句（防编造）",
                        sorted(said - known))
            return base

    merged = f"{base}；{text}" if base else text
    if len(merged) > limit:
        merged = merged[: max(0, limit - 1)] + "…"
    return merged


def _reason_numbers(text: str) -> set[str]:
    """文本里出现的全部数字（按数字边界切，不是子串）。"""
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def _reason_numbers_covered(said: set[str], known: set[str]) -> set[str]:
    """返回 `said` 中**无法在 `known` 里找到数值相容对应**的数字（空集 = 全部可核对）；不用字符串精确相等（`plans/0020` 真机复验：模型把 `距离1023.062m` 写 `距离1023米` 被旧逻辑误杀为编造）。
    相容规则：某数字 `s` 与某底稿数字 `b` 差 ≤ `max(1.0, 0.5%·max(|s|,|b|))` 即相容（大数容忍取整、小数/价格卡紧）；差远超仍判编造（如评分 3.7 写 9.9）。"""
    def _f(x: str) -> float | None:
        try:
            return float(x)
        except ValueError:
            return None
    known_floats = [v for v in (_f(k) for k in known) if v is not None]
    uncovered: set[str] = set()
    for s in said:
        sv = _f(s)
        if sv is None:
            uncovered.add(s)
            continue
        if not any(abs(sv - bv) <= max(1.0, 0.005 * max(abs(sv), abs(bv))) for bv in known_floats):
            uncovered.add(s)
    return uncovered


#: 模型推荐理由里的**无依据套话**黑名单（`plans/0020` T3）；整句丢而非"删掉这些词"：删完句子不通顺（"适合约会"剩"推荐这家，"像坏掉），**留着**又展示编造评价 ⇒ 宁可退回代码底稿（信息少但为真）。
_REASON_UNSUPPORTED_WORDS = (
    "环境", "氛围", "味道", "口味", "性价比", "排长队", "排队", "划算", "值得",
    "约会", "必吃", "打卡", "网红", "本地人", "口碑", "服务好", "干净", "地道",
)


#: recommend 轨每店推荐理由：LLM 生成的字数与单店预算（`plans/0020` §15.1 / #120）。
#: 预算是**单店**上限；一轮 recommend 最多 `NEARBY_TOP`(=5) 家 ⇒ 用 `asyncio.gather` 并发，
#: 总耗时 ≈ 最慢那一家，而不是 5 倍串行。
REASON_LLM_MAX = int(os.getenv("AGENT_REASON_LLM_MAX", "60"))
REASON_LLM_TIMEOUT = float(os.getenv("AGENT_REASON_LLM_TIMEOUT", "3.0"))

#: recommend 轨推荐理由的系统提示词（#120 / #121）：只许用给定信息，不许编造评价；目标不是"复述字段"而是**一句像朋友介绍的推荐语**（2026-10-08 反馈：要查完 RAG 让模型生成推荐话语，非截取片段）。
#: 旧提示词只写"不要编造" ⇒ 模型最安全地把人均/评分原样念一遍；必须**同时**给正向要求（把原文真实亮点说出来）与负向约束（不许新增数字/评价）。
RECOMMEND_REASON_SYSTEM = (
    "你是本地生活推荐助手。我会给你一家店的【已知事实】（人均/评分/距离/是否有券）"
    "和【顾客原文】（来自该店的博客与攻略片段）。请写**一句**不超过指定字数的中文推荐语，"
    "语气像朋友介绍这家店，把顾客原文里**真实出现过**的亮点说出来"
    "（例如有没有包间、营业到几点、酒水/价格、位置、适合的场合）。"
    "硬约束：① 只能用我给的信息；② 推荐语里**不许出现我没给过的数字**；"
    "③ 提到的评价（环境/服务/味道等）必须能在【顾客原文】里找到原话依据；"
    "④ **不要罗列字段**（不要写成「人均x元，评分y分」这样的清单）；"
    "⑤ 若信息确实不足，**直接留空** —— 不要输出「空字符串」「无」「暂无」这类占位词，"
    "也不要输出 JSON。"
)


#: 模型把"留空"**说**出来时的占位词（真机实测会回字面量 `空字符串`）；必须单独挡：这些词会**当推荐语展示**给用户（"推荐理由：空字符串"）比没理由更糟，同理还有偶发吐回的 JSON（`{"recommend": ""}`）。
_REASON_PLACEHOLDERS = frozenset(
    {"空字符串", "空", "无", "无内容", "暂无", "没有", "n/a", "na", "null", "none", "-", "—", "（无）", "(无)"}
)


def _is_placeholder_reason(text: str) -> bool:
    """模型那句是不是"空的占位表示"（占位词 / 裸 JSON）—— 是则按**空**处理，退回底稿。"""
    stripped = (text or "").strip().strip("。.！!～~ \"'")
    if not stripped:
        return True
    if stripped in _REASON_PLACEHOLDERS or stripped.lower() in _REASON_PLACEHOLDERS:
        return True
    # 裸 JSON（`{"recommend": ""}` / `[]`）：模型偶尔把上一版结构化输出的壳吐回来
    return stripped[0] in "{[" and stripped[-1] in "}]"


def _reason_material(rec: dict) -> str:
    """该店**可引用的原文素材**：博客检索块（`chunks`）并集攻略（`blogs`），按原文拼成多行。
    `blogs` 必须一起算进来：入口模型对「这家店怎么样」经常只判 `detail` 不判 `rag` ⇒ `chunks` 为空，只喂 `chunks` 会让模型只能复述人均/评分这种数据碎片。
    """
    lines: list[str] = []
    seen: set[str] = set()
    for key in ("chunks", "blogs"):
        for c in rec.get(key) or []:
            if not isinstance(c, dict):
                continue
            title = str(c.get("title") or "").strip()
            content = str(c.get("content") or "").strip()
            if not content:
                continue
            line = f"- {title}：{content}" if title else f"- {content}"
            if line in seen:
                continue
            seen.add(line)
            lines.append(line)
            if len(lines) >= 6:
                return "\n".join(lines)
    return "\n".join(lines)


def _ungrounded_words(text: str, material: str) -> list[str]:
    """`text` 里出现、但**原文素材里找不到依据**的套话词（空列表 = 全部有依据）；黑名单加"依据豁免"：本项目博客原文里真的写着这些词，旧行为把"模型忠实转述原文"判成编造、整句丢掉，与用户诉求正好相反。
    判据：词（或其前两字）在素材里出现过即视为有依据；素材为空时不豁免任何词（保守丢弃）。
    """
    if not material:
        return [w for w in _REASON_UNSUPPORTED_WORDS if w in text]
    hits: list[str] = []
    for w in _REASON_UNSUPPORTED_WORDS:
        if w not in text:
            continue
        if w in material or (len(w) > 2 and w[:2] in material):
            continue
        hits.append(w)
    return hits


def _ground_reason(base: str, model_text: str, rec: dict, *, limit: int = 60) -> str:
    """把 LLM 生成的推荐理由做**防编造校验**（#120），返回可直接展示的推荐语。
    与 `_merge_reason`（plan 轨拼底稿）不同：recommend 轨不拼底稿，校验不过返回空串由前端隐藏，不退回"人均80元、评分3.7"；规则：占位答案/套话黑名单且素材无依据/数字不在底稿∪素材里 ⇒ 空串（数值相容比较）。
    """
    text = (model_text or "").strip()
    # 先挡"占位答案"：真机实测模型会把"留空"这件事**说**出来（回一个字面量 `空字符串`），
    #    或把上一版结构化输出的壳吐回来（`{"recommend": ""}`）——
    #    照原样放行就会在卡片上显示「推荐理由：空字符串」。
    if _is_placeholder_reason(text):
        return ""
    material = _reason_material(rec)
    if _ungrounded_words(text, material):
        return ""
    if base:
        said = _reason_numbers(text)
        known = _reason_numbers(base) | _reason_numbers(material)
        uncovered = _reason_numbers_covered(said, known)
        if uncovered:
            logger.info("推荐理由含底稿/原文之外的数字 %s，已丢弃该句（防编造）", sorted(uncovered))
            return ""
    if len(text) > limit:
        text = text[: max(0, limit - 1)] + "…"
    return text


async def _llm_reason_text(rec: dict, base: str, timeout: float) -> str:
    """真正去打模型的那一层（`_compose_reason` 的下游）。任何失败/超时都返回空串 ⇒ 上游退回 base。"""
    material = _reason_material(rec)
    human = (
        f"【已知事实】\n{base or '（无）'}\n\n"
        f"【顾客原文】\n{material or '（未取到任何原文）'}\n\n"
        f"请基于以上信息，写一句不超过 {REASON_LLM_MAX} 字的推荐理由。"
    )
    try:
        resp = await asyncio.wait_for(
            get_light_model().ainvoke([
                SystemMessage(RECOMMEND_REASON_SYSTEM),
                HumanMessage(human),
            ]),
            timeout=timeout,
        )
        return as_text(getattr(resp, "content", "")).strip()
    except asyncio.TimeoutError:
        logger.warning("推荐理由 LLM 超时（%.1fs），降级底稿", timeout)
        return ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("推荐理由 LLM 生成失败，降级底稿: %s", exc)
        return ""


async def _compose_reason(rec: dict, *, timeout: float = REASON_LLM_TIMEOUT) -> str:
    """recommend 轨每店的推荐理由：调用轻量模型，基于该店原文素材 + 真实字段生成一句（#120）；只把【已知事实】与【顾客原文】喂给模型，不喂别的上下文 ⇒ 模型无从编造。
    输出过 `_ground_reason` 校验（含底稿外数字/无依据套话/占位词 ⇒ 空串）；净效果：有素材才有理由，没素材就不显示理由，而不是退回数据碎片。
    """
    # 没有原文素材 ⇒ 直接不产理由，连模型都不调（2026-10-08 真机）：用户诉求的前提是"查到数据"；
    #    素材为空时模型只有人均/评分，写出来必然是字段复述，而数字复述照样能过 `_ground_reason` 校验 ⇒ 必须在入口分流。
    #    顺带省掉一次注定没用的模型调用（真机：入口只判 `voucher` 意图时就是这种情况）。
    if not _reason_material(rec):
        return ""
    base = _stop_reason(rec)
    model_text = await _llm_reason_text(rec, base, timeout)
    return _ground_reason(base, model_text, rec)


def _placeholder_shop_name(shop_id: Any) -> str:
    """记录工厂在"名字未知"时填的占位名——与 `_new_shop_record` **同一个规则**；占位名是**非空字符串**，`or 兜底`会直接短路 ⇒ 判定必须按**字面是否等于占位**来做。"""
    return f"店铺{shop_id}"


def _shop_display_name(rec: dict, sid: Any, name_by_id: dict) -> str:
    """记录里的展示用店名：真名优先，占位名/空值才回退 `name_by_id`（见 `_placeholder_shop_name`）。"""
    name = str(rec.get("name") or "").strip()
    if name and name != _placeholder_shop_name(sid):
        return name
    return str(name_by_id.get(sid) or "").strip() or name or _placeholder_shop_name(sid)


async def _topup_reason_material(recs: list[dict], user_id: str) -> list[dict]:
    """给"一点原文素材都没有"的店**补一次攻略**（只读、并行、失败即放弃）：收窄前置只把 Top-1 扇出给 `shop_worker`，其余候选既没有攻略也没有检索块 ⇒ 不补就会出现"5 张卡片只有 1 张有推荐理由"。
    只对确实没素材的店补一次 `fetch_shop_blogs`（约 50ms、`asyncio.gather` 并行）；只补 `blogs` 不碰 `chunks`（避免取数口径出现第二处真相）。
    """
    if not user_id:
        return recs
    targets = [r for r in recs if not _reason_material(r)][:NEARBY_TOP]
    if not targets:
        return recs

    async def _one(rec: dict) -> None:
        sid = _shop_id_of(rec)
        if sid is None:
            return
        try:
            res = await asyncio.to_thread(fetch_shop_blogs, sid, 1, user_id)
        except Exception as exc:  # noqa: BLE001  补素材失败只影响"有没有理由"
            logger.warning("推荐理由素材补齐失败 shop_id=%s: %s", sid, exc)
            return
        blogs, err = _split_list_or_error(res or [])
        if blogs:
            rec["blogs"] = blogs
            rec["blogs_error"] = ""

    await asyncio.gather(*(_one(r) for r in targets))
    logger.info("推荐理由素材补齐：%s/%s 家缺少原文，已尝试补攻略", len(targets), len(recs))
    return recs


async def _recommend_shops(state: "HmdpState") -> list[dict]:
    """从 `worker_outputs` 抽店铺记录，给每店生成 LLM 推荐理由，产出卡片载荷（#117 / #120）；抽卡走与 react/渲染同一条读取路径 `_react_candidates`（拍平 `items`、按 `id` 归并），`reason` 走 `_compose_reason`，`evidence` 走 `_stop_evidence`。
    并发生成 reason：`NEARBY_TOP` 家同时打轻量模型，`asyncio.gather` ⇒ 总耗时≈最慢一家。
    """
    outputs = list(state.get("worker_outputs") or [])
    # 抽卡必须走与 react / 渲染**同一条**读取路径 `_react_candidates`：每个 `worker_outputs`
    # 条目都是 `{"worker": ..., "items": [...]}` 的**包装**，店记录嵌在 `items` 里；旧版直接遍历顶层
    # 并用 `if o.get("worker"): continue` 全跳过 ⇒ recommend 卡片恒为 0 张（`plans/0020` 真机复验发现）。
    recs = _react_candidates(outputs)

    if not recs:
        return []
    # 收窄前置的副作用：只有 Top-1 被扇出取数 ⇒ 其余候选没有攻略/检索块（见 `_topup_reason_material`）
    recs = await _topup_reason_material(recs, str(state.get("user_id") or ""))
    # 店名的**兜底来源**：本轮焦点店铺（`grounding` 反查数据库得到的名字）——真机实测（2026-10-08）：
    #    入口只判出 `voucher` 意图时取数路径根本不取详情 ⇒ 记录里没有 `name` ⇒ 卡片显示成「店铺10」；
    #    只用它补**空**名字，绝不覆盖记录里已有的名字（记录才是权威）。
    name_by_id = {
        _shop_id_of(f): str(f.get("name") or "")
        for f in (state.get("focus_shops") or [])
        if isinstance(f, dict) and _shop_id_of(f) is not None and str(f.get("name") or "").strip()
    }
    reasons = await asyncio.gather(*(_compose_reason(r) for r in recs))
    shops: list[dict] = []
    for r, reason in zip(recs, reasons):
        sid = _shop_id_of(r)
        score = r.get("score")
        score_disp = (score / 10) if isinstance(score, (int, float)) and score else None
        shops.append({
            "shop_id": sid,
            "name": _shop_display_name(r, sid, name_by_id),
            "address": r.get("address") or "",
            "x": r.get("x"),
            "y": r.get("y"),
            "score": score_disp,
            "avgPrice": r.get("avgPrice"),
            "distance": r.get("distance"),
            "vouchers": list(r.get("vouchers") or []),
            "order_cards": list(r.get("order_cards") or []),
            "reason": reason,
            "evidence": _stop_evidence(r),
            "error": r.get("error") or "",
        })
    return shops


def _compose_stops(
    plan: ItineraryPlan | None, groups: list[tuple[str, list[dict]]], slots: dict, max_stops: int
) -> list[dict]:
    """把"模型挑的"与"真实记录"合成 `stops`。
    白名单过滤（R1）：`shop_id` 不在候选集里的直接丢弃（模型会编店），其余事实字段一律从记录回填；`reason` 是唯一允许模型参与的字段（`plans/0020` T3），由 `_merge_reason` 拿底稿校验，含底稿外数字整句丢弃。
    """
    index = _index_records(groups)
    picks: list[tuple[int, str, str, str, str]] = []
    for item in (plan.stops if plan else []) or []:
        if not isinstance(item, dict):
            continue
        try:
            sid = int(item.get("shop_id"))
        except (TypeError, ValueError):
            continue
        if sid not in index:
            logger.warning("行程编排给出候选集之外的 shop_id=%s，已丢弃（防幻觉门店）", sid)
            continue
        picks.append(
            (
                sid,
                str(item.get("activity") or "").strip() or _activity_of(sid, groups),
                str(item.get("start") or "").strip(),
                str(item.get("end") or "").strip(),
                str(item.get("reason") or "").strip(),
            )
        )

    if not picks:
        # 规则兜底：每个场景取第一家（模型不可用时仍能出行程，而不是空手而归）
        for scene, records in groups:
            if records:
                picks.append((_shop_id_of(records[0]), scene, "", "", ""))

    seen: set[int] = set()
    unique: list[tuple[int, str, str, str, str]] = []
    for pick in picks:
        if pick[0] in seen:
            continue
        seen.add(pick[0])
        unique.append(pick)
    unique = unique[:max_stops]

    stops: list[dict] = []
    cursor = _start_minutes(slots.get("time"))
    for seq, (sid, activity, start, end, why) in enumerate(unique, 1):
        rec = index[sid]
        stops.append(
            {
                "seq": seq,
                "shop_id": sid,
                "name": rec.get("name") or f"店铺{sid}",
                "address": rec.get("address") or "",
                "x": rec.get("x"),
                "y": rec.get("y"),
                "start": start or _hhmm(cursor),
                "end": end or _hhmm(cursor + PLAN_STOP_MINUTES),
                "activity": activity,
                # 推荐理由：代码底稿（只含已知事实）+ 模型那句**通过校验后**的补充
                "reason": _merge_reason(_stop_reason(rec), why, rec),
                "evidence": _stop_evidence(rec),
                "vouchers": list(rec.get("vouchers") or []),
                "order_cards": list(rec.get("order_cards") or []),
            }
        )
        cursor += PLAN_STOP_MINUTES
    return stops


def _default_title(slots: dict) -> str:
    area = str(slots.get("area") or "").strip()
    return f"{area}行程" if area else "本地生活行程"


def _mention_order(query: str, shop_names: list[str]) -> list[str]:
    """用户**原话**中按先后提到的店名序列（纯函数；`specs/0024` #6 / T3）。
    只收原话里真的出现、按首次出现位置排序的名字（"同一场景内多店定序"的证据），没提到的不进结果。
    """
    text = str(query or "")
    hits: list[tuple[int, str]] = []
    for raw in shop_names or []:
        name = str(raw or "").strip()
        if not name:
            continue
        pos = text.find(name)
        if pos >= 0:
            hits.append((pos, name))
    hits.sort(key=lambda pair: pair[0])
    out: list[str] = []
    for _pos, name in hits:
        if name not in out:
            out.append(name)
    return out


def _same_scene_reorder(stops: list[dict], order: list[str]) -> tuple[list[dict], bool]:
    """同一 `activity` 内按用户原话先后重排（纯函数，`specs/0024` §8 Q3）：只在同一 activity 内 ≥2 站都命中 `order` 且当前顺序不一致时动手，其余站与跨 activity 顺序一律不动。
    时段原样带走，`seq` 重编号、`legs` 由调用方重算；:returns: `(stops, changed)`，无改动返回入参本身。
    """
    if not stops or len(order) < 2:
        return stops, False

    def _rank(stop: dict) -> int | None:
        """该站在 `order` 里的位次（店名包含匹配）；不在 `order` 里返回 None。"""
        name = str(stop.get("name") or "")
        for i, hint in enumerate(order):
            if hint and (hint in name or name in hint):
                return i
        return None

    groups: dict[str, list[int]] = {}
    for idx, stop in enumerate(stops):
        groups.setdefault(str(stop.get("activity") or ""), []).append(idx)

    new = list(stops)
    changed = False
    for _activity, idxs in groups.items():
        hinted = [(i, r) for i, r in ((i, _rank(stops[i])) for i in idxs) if r is not None]
        if len(hinted) < 2:
            continue
        slots_ = [i for i, _r in hinted]                                  # 命中站当前所在的位置
        target = [i for i, _r in sorted(hinted, key=lambda pair: pair[1])]  # 按原话先后重排后的顺序
        if target == slots_:
            continue
        for slot, src in zip(slots_, target):
            new[slot] = stops[src]
        changed = True

    if not changed:
        return stops, False
    copied = [dict(s) for s in new]
    for i, stop in enumerate(copied, 1):
        stop["seq"] = i
    return copied, True


def _itinerary_payload(
    draft: dict, slots: dict, plan: ItineraryPlan | None, stops: list[dict],
    groups: list[tuple[str, list[dict]]],
) -> dict:
    """组 `itinerary` 载荷（`specs/0012` §12.1）。"""
    if plan is not None and plan.title and plan.title.strip():
        title = plan.title.strip()
    else:
        title = _default_title(slots)
    missing_scenes = [scene for scene, records in groups if not records]
    return {
        "task_id": draft.get("task_id") or "",
        "title": title,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "slots": dict(slots),
        "stops": stops,
        "legs": _compose_legs(stops),
        "missing_scenes": missing_scenes,
    }


async def itinerary(state: HmdpState) -> Command[Literal["suggest"]]:
    """把候选编排成一份行程（`specs/0012` §12.1）：模型挑站排序，代码持有一切事实字段。"""
    record_node("itinerary")
    draft = dict(state.get("task_draft") or {})
    slots = draft.get("slots") or {}
    route_plan = plan_of(state)
    query_text = state.get("query") or ""
    groups = _plan_groups(state)
    # 偏好与用户原话一起喂给编排模型（`specs/0024` T2）：原先顺序完全靠模型自由裁量。
    plan = await _ask_itinerary(
        groups, slots, ITINERARY_MAX_STOPS,
        preferences=route_plan.preferences if route_plan else [],
        query=query_text,
    )
    stops = _compose_stops(plan, groups, slots, ITINERARY_MAX_STOPS)
    # 同场景顺序校验（`specs/0024` T3）：用户原话里点名了先后时，纠正"同一 activity 内"的排反；
    # 跨场景顺序与未点名站一律不动（见 `_same_scene_reorder`）。
    order = _mention_order(query_text, route_plan.shop_names if route_plan else [])
    stops, reordered = _same_scene_reorder(stops, order)
    if reordered:
        logger.info("行程同场景顺序按用户原话校正：原话命中顺序=%s", order)
    payload = _itinerary_payload(draft, slots, plan, stops, groups)
    logger.info("行程编排完成：%s 站（缺场景=%s）", len(stops), payload["missing_scenes"])
    return Command(
        goto="suggest",
        update={"itinerary": payload, "task_draft": {**draft, "stops": stops}},
    )


async def replace_stop(state: HmdpState) -> Command[Literal["suggest"]]:
    """单站替换（`specs/0012` §12.5）：**只重排指定站**，其余站的 `shop_id` / 时段原样不动。
    不做整条重排/多站替换/全局最优——那会把"换个第二站"变成"重新规划一次"，用户已认可的前后站全被打散。
    """
    record_node("replace_stop")
    draft = dict(state.get("task_draft") or {})
    stops = [dict(s) for s in (draft.get("stops") or [])]
    query = state.get("query") or ""
    slots = draft.get("slots") or {}
    current = state.get("itinerary") or {}

    seq = _replace_stop_seq(query, stops)
    if seq is None:
        # 没有可替换的目标：原样交给 `suggest`（照常产摘要），不报错、不改动任何一站
        return Command(
            goto="suggest",
            update={"itinerary": {**current, "slots": slots, "stops": stops,
                                  "legs": _compose_legs(stops)}},
        )

    target = stops[seq - 1]
    used = {s.get("shop_id") for s in stops}
    activity = str(target.get("activity") or "")
    candidates = await _scene_candidates(
        activity, state.get("x"), state.get("y"), state.get("user_id") or "",
        PLAN_CANDIDATES_PER_SCENE + 2,
    )
    replacement = next((r for r in candidates if _shop_id_of(r) not in used), None)

    if replacement is None:
        payload = {
            **current, "slots": slots, "stops": stops, "legs": _compose_legs(stops),
            "note": f"没找到能替换第{seq}站的其他店，先保持原样。",
        }
        logger.info("单站替换：第%s 站没有可替换的店", seq)
        return Command(goto="suggest", update={"itinerary": payload})

    new_stop = {
        "seq": seq,
        "shop_id": _shop_id_of(replacement),
        "name": replacement.get("name") or f"店铺{_shop_id_of(replacement)}",
        "address": replacement.get("address") or "",
        "x": replacement.get("x"),
        "y": replacement.get("y"),
        # 替换后的站也要有理由/证据（否则前端会出现"有的站有理由、有的没有"）
        "reason": _stop_reason(replacement),
        "evidence": _stop_evidence(replacement),
        # 时段**保持原值**：本期不做"新站时长溢出则顺延后序"的联动（少一个会出错的自由度）
        "start": target.get("start"),
        "end": target.get("end"),
        "activity": activity,
        "vouchers": list(replacement.get("vouchers") or []),
        "order_cards": list(replacement.get("order_cards") or []),
    }
    replaced = [new_stop if s.get("seq") == seq else s for s in stops]
    payload = {
        **current, "slots": slots, "stops": replaced, "legs": _compose_legs(replaced),
        "note": f"已把第{seq}站换成「{new_stop['name']}」。",
    }
    logger.info("单站替换：第%s 站 -> shop_id=%s", seq, new_stop["shop_id"])
    return Command(
        goto="suggest",
        update={"itinerary": payload, "task_draft": {**draft, "stops": replaced}},
    )


#: 建议项总条数上限（`specs/0012` §12.2）：一屏按钮不宜过多。
MAX_SUGGESTIONS = 8


def _booking_time_of(date_text: str, hhmm: str) -> datetime | None:
    """把行程的 `date` + `stops[].start` 合成 Java 侧要的 `yyyy-MM-dd HH:mm:ss`（纯函数）。
    用本地墙钟与 `datetime.now()` 比对；合成失败返回 `None`，调用方不产预约建议项——宁可少给一个按钮，也不能给一个点了必报 400 的按钮。"""
    text = str(hhmm or "").strip()
    if not re.fullmatch(r"\d{1,2}:\d{2}", text):
        return None
    day = str(date_text or "").strip() or datetime.now().strftime("%Y-%m-%d")
    try:
        return datetime.strptime(f"{day} {text}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def _bookable_time(date_text: str, hhmm: str, *, lead_minutes: int = 30) -> str | None:
    """可预约的到店时间；**已过期则返回 `None`**（人类决策 D3：生成期就规避，不等用户点了才 400）；`lead_minutes`：至少比"现在"晚这么多分钟。"""
    moment = _booking_time_of(date_text, hhmm)
    if moment is None:
        return None
    if moment < datetime.now() + timedelta(minutes=lead_minutes):
        return None
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def _build_suggestions(stops: list[dict], slots: dict, date_text: str = "") -> list[dict]:
    """按行程产生**结构化**建议项（纯函数）：建议项不进自然语言正文（§12.2），可点击、可埋点；`booking` 项先按 `_bookable_time` 过滤过期时刻（D3），否则用户点下去只会吃 Java 侧 400。"""
    out: list[dict] = []
    people = slots.get("people")
    for stop in stops[:2]:
        if stop.get("vouchers"):
            out.append(
                {"type": "voucher", "label": "看看有什么券", "args": {"shop_id": stop["shop_id"]}}
            )
    for stop in stops[:2]:
        bookable = _bookable_time(date_text, stop.get("start") or "")
        if not bookable:
            continue          # 时刻不可预约（格式不认识或已过期）⇒ 不给这个按钮
        out.append(
            {
                "type": "booking",
                "label": "帮我预约",
                "args": {
                    "shop_id": stop["shop_id"],
                    "time": stop.get("start") or "",
                    "people": people,
                    # 完整到店时间（Java 侧要秒级）：前端点这个按钮时**原样回传**，
                    #    不要自己拼 —— 拼错格式就是 400。
                    "booking_time": bookable,
                },
            }
        )
    if len(stops) >= 2:
        out.append(
            {
                "type": "compare",
                "label": "换一家对比",
                "args": {"shop_ids": [stops[0]["shop_id"], stops[1]["shop_id"]]},
            }
        )
    for stop in stops[:3]:
        out.append(
            {
                "type": "replace_stop",
                "label": f"第{stop['seq']}站换一家",
                "args": {"stop_seq": stop["seq"]},
            }
        )
    return out[:MAX_SUGGESTIONS]


def _itinerary_summary_messages(state: HmdpState) -> list:
    """把 `itinerary` 载荷事实拼成喂模型的 messages（`specs/0019` T1，纯函数）：事实只来自载荷，函数内不调模型、不读其它 State（否则无法离线单测"喂了什么"）；只给 stops 不给候选全集（不给 `shop_id`、不带候选清单）——给得越多模型越有机会把没排进行程的店说成"已安排"（幻觉主要入口）。"""
    payload = state.get("itinerary") or {}
    slots = payload.get("slots") or {}
    title = str(payload.get("title") or "")
    date_text = str(payload.get("date") or "")

    lines: list[str] = []
    head = f"行程标题：{title}" if title else "行程标题：（未命名）"
    if date_text:
        head += f"（日期 {date_text}）"
    lines.append(head)

    slot_bits: list[str] = []
    if slots.get("time"):
        slot_bits.append(f"时间 {slots.get('time')}")
    if slots.get("area"):
        slot_bits.append(f"区域 {slots.get('area')}")
    if slots.get("people"):
        slot_bits.append(f"人数 {slots.get('people')} 人")
    if slots.get("budget"):
        slot_bits.append(f"预算约 {slots.get('budget')} 元")
    lines.append("行程口径：" + ("、".join(slot_bits) if slot_bits else "（未提供）"))

    stops = payload.get("stops") or []
    if stops:
        for stop in stops:
            bits = [f"第{stop.get('seq')}站", str(stop.get("name") or "")]
            span = "-".join(x for x in (str(stop.get("start") or ""), str(stop.get("end") or "")) if x)
            if span:
                bits.append(span)
            if stop.get("activity"):
                bits.append(f"活动：{stop['activity']}")
            if stop.get("reason"):
                bits.append(f"理由：{stop['reason']}")
            lines.append("；".join(bits))
    else:
        lines.append("（这份行程没有排到任何店）")

    missing = payload.get("missing_scenes") or []
    if missing:
        lines.append("未找到的场景：" + "、".join(str(m) for m in missing))
    note = str(payload.get("note") or "").strip()
    if note:
        lines.append(f"备注：{note}")

    return [
        SystemMessage(plan_summary_system()),
        HumanMessage("以下是一份已经排好的行程（事实以此为准）：\n" + "\n".join(lines)),
    ]


def _itinerary_digest(payload: dict) -> str:
    """行程的进记忆摘要（`specs/0012` §11.1：含"共 N 站"而非逐站明细）；正文超长被 `_digest_for_memory` 显式截断（不静默丢）。"""
    stops = payload.get("stops") or []
    if not stops:
        return payload.get("note") or "这次没找到合适的店，换个区域或品类我再试试。"
    names = " → ".join(str(s.get("name")) for s in stops[:3])
    tail = " 等" if len(stops) > 3 else ""
    slots = payload.get("slots") or {}
    budget = slots.get("budget")
    extra = f"预算约 {budget} 元。" if budget else ""
    missing = payload.get("missing_scenes") or []
    note = f"（{'、'.join(str(m) for m in missing)}暂时没找到合适的店）" if missing else ""
    lead = str(payload.get("note") or "")
    prose = f"{lead}已为你排好行程「{payload.get('title')}」，共 {len(stops)} 站：{names}{tail}。{extra}{note}"
    return _digest_for_memory(prose)


async def _stream_plan_answer(messages: list) -> tuple[str, Any]:
    """计划轨终态答复的**纯文本流式**调用（`specs/0019` T2）：与 `_summarize_plain_stream` 同构但刻意不共用（复用会为 plan 轨动 recommend 的签名/计账口径，两条轨彼此连坐）。
    只保留 plan 轨需要的部分：逐 token 累加 + `_parse_possible_json` 防御 + 返回带 usage 的 chunk（供 `anchor_patch` 用）。
    """
    parts: list[str] = []
    record_model_call("plan_summary")
    response = await _stream_join(get_main_model().astream(messages), parts)
    raw = "".join(parts).strip()
    parsed_answer, _ = _parse_possible_json(raw)
    return parsed_answer or raw, response


async def suggest(state: HmdpState) -> dict:
    """计划轨**终态**：产建议项 + 行程摘要 + 状态收尾（`specs/0012` §12.2 / §11.1）。
    答复由模型流式生成（`specs/0019`）：事实只来自 `itinerary` 载荷（防幻觉在上游 `_compose_stops` 做完），模型只说人话；失败/返回空回落 `_itinerary_digest` 模板；终态清理与 `summarize_answer` 共用 `_terminal_cleanup`。
    """
    record_node("suggest")
    payload = state.get("itinerary") or {}
    stops = payload.get("stops") or []
    # `date` 必须传：`_bookable_time` 靠它合成完整到店时间，缺了会退化成"今天"，
    #    于是**跨天行程**的过期判断全错（D3）。
    suggestions = _build_suggestions(stops, payload.get("slots") or {}, str(payload.get("date") or ""))

    answer = ""
    response = None
    try:
        answer, response = await _stream_plan_answer(_itinerary_summary_messages(state))
    except Exception as exc:  # noqa: BLE001 - 模型失败不能影响终态（有模板兜底）
        logger.warning("计划轨答复生成失败，回落模板: %s", exc)

    if not answer:
        # 回落模板（`specs/0019` §8 #3 决策）：模型不可用 / 返回空时保住"有话说"，
        #    不抛错、不空手。`_itinerary_digest` 全程保留，回滚即恢复原行为。
        logger.info("计划轨答复为空，回落 _itinerary_digest 模板")
        answer = _itinerary_digest(payload)

    return {
        "suggestions": suggestions,
        "answer": answer,
        "sources": [],
        # 写回记忆的是**摘要**（`specs/0019` 验收 6）：`answer` 全文只用于 SSE 与 DB
        #    `content`，进 message list 的是 `_digest_for_memory`（上限 `PLAN_ANSWER_MAX_CHARS`=400）
        #    —— 否则多轮 plan 会话在上下文里累积长正文，撑预算且触发频繁压缩。
        "messages": [AIMessage(content=_digest_for_memory(answer))],
        **_terminal_cleanup(state),
        **anchor_patch(response, state.get("messages") or []),
    }


# 计划轨：预约写路径（`plans/0019`）——全图**第二个** `interrupt()` 点
#: 预约意图的**确定性**触发（与 `capability` / `replace_stop` 同款短路）：写路径入口判定**绝不交给模型**（判错会"没确认就下单"或当下单意图当推荐）；**只认祈使/意愿句式**（"请/帮我/我要"发起标记或句尾祈使），不认"提到预约二字"的问规则，光秃秃"预约"不算。
_BOOKING_INTENT = re.compile(
    r"("
    r"(请|帮我|给我|帮忙|替我|我要|我想|想要|来|再)\s*"
    r"(?:请|帮我|给我|帮忙|替我|我要|我想|想要)?\s*"
    r"(预约|预订|订位|订个位|留个位|留位|订桌|订个桌)"
    r"|(预约|预订|订位|订桌)(?:一下|下|吧|好吗|行吗|嘛)"
    # 口语里"留个位""订个桌"**省略了意愿词**本身就是祈使句（"留个位"不会有第二种意思），
    # 所以单列一支；它们含"个"字，不会撞上"预约要提前多久"这类问句。
    r"|(留个位|留位|订个位|订个桌|订桌)"
    r")"
)

#: 确认动作的**白名单**（结构化，非自然语言）。白名单之外的一切值 —— 包括 `None`、
#: 乱串、模型生成的文本 —— **一律当取消**。写操作的默认分支必须是"不写"。
_CONFIRM_ACTIONS = frozenset({"confirm", "ok", "yes", "submit"})

#: 修改预约信息的三种说法（`plans/0021` §16.1）；存在的全部理由（`plans/0021` F3）：原本"白名单外一律取消" ⇒ 前端新增"改时间"按钮回传 `{type:"modify"}` 时不是忽略而是**把预约取消**并播"已取消"（用户点改时间得已取消）。
#: 它必须排在 `booking_confirm` 里"一律取消"分支的**之前**（顺序反了功能形同虚设）。
_MODIFY_ACTIONS = frozenset({"modify", "edit", "change"})

#: 撤销动作（`plans/0026`）：确认成功后卡片上**只剩这一个**按钮，所以它必须被**显式识别**，
#: 否则会掉进"白名单之外一律取消"——用户点「取消预约」得到"预约已取消，没有提交任何信息"
#: （听起来对，其实是**没有真正撤单**）。语义与 `_MODIFY_ACTIONS` 同一个坑。
_CANCEL_BOOKING_ACTIONS = frozenset({"cancel_booking", "cancel-booking", "cancel_order"})

#: 澄清卡「一次选完再提交」的动作类型（`plans/0026` / `specs/0025`）：结构化载荷在 `slots`。
#: 已于 `specs/0025` **接线**：`clarify_slots` 节点据此判定"结构化通道" ⇒ 那条路**不调模型**，
#:    直接映射槽位、只过 `plan_slots.is_valid` 的值域校验（区间的 `time` 先归一成 `HH:MM`）。
_CLARIFY_ANSWER_ACTIONS = frozenset({"clarify", "clarify_answer", "slot_fill"})

#: 自然语言的撤单意图。必须**同时**出现撤单动词**与**预约名词（短窗口内）：
#: 光一个"取消"会撞上"取消行程/取消筛选"这类完全无关的说法。
_CANCEL_BOOKING_INTENT = re.compile(r"(取消|撤销|退掉|取消掉)[^，。！？]{0,6}(预约|预订|订单|单子)")

#: 用户在确认卡上**可改**的字段（`plans/0021` §16.4）。
#: 由**后端**给出而不是前端硬编码：将来增减字段不需要前端跟版，两边不会漂移。
BOOKING_EDITABLE_FIELDS = ("booking_time", "people", "remark")

#: 同一轮预约里最多允许改几次（`plans/0021` §16.2）。超过就交回入口。
#: 没有它 ⇒ 前端 bug 反复发 `modify` 会**无限挂起**，用户永远结束不了这一轮。
BOOKING_MODIFY_MAX = int(os.getenv("AGENT_BOOKING_MODIFY_MAX", "3"))


def _parse_booking_time(raw: str) -> float:
    """把时间串解析成 epoch 秒；解析不出来返回 0；接受 `YYYY-MM-DD HH:MM`（也接受 `T` 分隔与秒级，前端 `datetime-local` 形状）。**刻意不猜**「今天/明天」这类相对表述：那是用户输入层职责，这里猜会把"改时间"变成"改成另一个时间"，比解析失败更糟。"""
    text = (raw or "").strip().replace("T", " ")
    if not text:
        return 0.0
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return time.mktime(time.strptime(text, fmt))
        except ValueError:
            continue
    return 0.0


def _apply_booking_modification(draft: dict, args: Any, *, now: float | None = None
                                ) -> tuple[dict, str]:
    """把 `action.args` **局部覆盖**到草稿上（纯函数，`plans/0021` §16.1/§16.2）；**局部覆盖非整体替换**——前端只带 `{"people":4}` 时未出现的键保持原值，否则时间/备注被清空。
    返回 `(新草稿, 错误原因)`：原因为空串表示通过，校验失败时新草稿等于原草稿（调用方据此不写单、重新挂起给可读原因）；`remark` 超长**截断到 255**不拒绝。
    校验（§16.2）：`booking_time` 早于当前⇒拒绝；`people`≤0或非整数⇒拒绝；`args` 全空/无可识别键⇒拒绝（视为无效修改）。"""
    out = dict(draft)
    if not isinstance(args, dict):
        return out, "没看懂要改什么，请重新选择要修改的内容。"

    reference = time.time() if now is None else now
    touched = False
    reason = ""

    if "people" in args:
        raw = args.get("people")
        try:
            people = int(str(raw).strip())
        except (TypeError, ValueError):
            people = 0
        if people <= 0:
            reason = f"人数得是正整数（现在收到的是 {raw}），请重新填。"
        else:
            out["people"] = people
            touched = True

    if "booking_time" in args:
        raw = str(args.get("booking_time") or "").strip()
        parsed = _parse_booking_time(raw)
        if not parsed:
            reason = f"时间格式看不懂（收到 {raw}），请按「2026-10-09 18:30」这样的格式填。"
        elif parsed <= reference:
            reason = "这个时间已经过去了，换一个吧。"
        else:
            # 存**统一格式**（`%Y-%m-%d %H:%M:%S`）：前端传 `2026-10-09T18:30`、
            #    写单路径传别的形状时，下游 `create_booking` 会拿到不一样的字符串。
            out["booking_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(parsed))
            touched = True

    if "remark" in args:
        # 超长**截断**而不是拒绝：备注是自由文本，用户粘一大段很正常；
        #    拒绝会让用户觉得"改个备注失败了"，比截断差得多。
        out["remark"] = str(args.get("remark") or "")[:255]
        touched = True

    if not touched and not reason:
        reason = "没有可修改的内容，请告诉我想改时间、人数还是备注。"

    return out, reason


def _cancellable_order_id(state: HmdpState) -> int | None:
    """可撤的单号：**只认 State 里"下单成功"的那条结果**（`booking_result`）；撤单是写操作，单号**必须**从服务端记录查出，前端 action 带的 `order_id` 一律不看（否则传别人单号就能撤，与 `_booking_draft_from_action` 不信任 action 同名同纪律）。
    `booking_result` 带 `result` 落 State（进 checkpoint）：仅 `success` 算"有一单在"，`cancelled`/`cancel_done`/`cancel_failed` 不再可撤，同时充当**幂等闸门**（重复取消不重复下单/撤单）。"""
    res = state.get("booking_result") or {}
    if not isinstance(res, dict) or str(res.get("result") or "") != "success":
        return None
    try:
        return int(res.get("order_id"))
    except (TypeError, ValueError):
        return None


#: 预约意图里那些**不是店名**的部分（提店名时要剥掉）。
_BOOKING_PHRASE_NOISE = (
    "帮我预约", "帮我预定", "帮我订", "给我预约", "给我订", "请你预约", "请帮我预约",
    "我要预约", "我想预约", "想要预约", "我想订", "我要订", "帮忙预约", "替我预约", "替我订",
    "预约一下", "预定一下", "帮我订个位", "预订一下", "订个位", "留个位", "订个桌", "订桌",
    "预约", "预订", "订位", "留位",
    # 句尾的意愿/礼貌词：剥掉后剩下的才是店名
    "可以吗", "行吗", "好吗", "吧", "谢谢", "一下", "下", "呢", "哦", "啊",
)


def _booking_shop_candidate(query: str) -> str:
    """从"预约"这句话里**提出店名候选**（纯规则，`plans/0026`）；病灶：入口预约短路**只看上下文** ⇒ 用上一轮焦点 103茶餐厅，用户被问错门店。
    这里只"把不是店名的部分剥掉"**不做任何猜测**：剥空返回空串由调用方决定问用户或用唯一候选；"店是否存在"由 `_lookup_one_shop`（查库）回答，规则**不编店名**。"""
    s = (query or "").strip()
    if not s:
        return ""
    for w in _BOOKING_PHRASE_NOISE:
        s = s.replace(w, "")
    # 常见标点与语气词收尾：剥完剩下的才可能是店名
    s = s.strip(" \t，,。.！!？?~～、:：;；\"'“”‘’()（）[]【】")
    return s.strip()


def _booking_draft_from_shop(shop: Any, slots: dict[str, Any]) -> dict[str, Any] | None:
    """用**查库得到的店铺对象**组草稿（纯函数；`plans/0026`）；店名/地址一律取**查库结果**（非用户原话/模型输出，否则"预约米其林"就能让 agent 这么播报）。
    到店时间不在这里猜：留空由 `booking_confirm` 走补时间澄清。"""
    sid = _shop_id_of(shop)
    if sid is None:
        return None
    return {
        "shop_id": sid,
        "shop_name": (shop.get("name") if isinstance(shop, dict) else "") or "",
        "address": (shop.get("address") if isinstance(shop, dict) else "") or "",
        "booking_time": "",
        "people": slots.get("people"),
        "remark": "",
    }


def _booking_candidates(state: HmdpState) -> list[dict]:
    """"可能想约的店"候选：焦点店铺 ∪ 行程站点（按 id 去重、保持先后）；用途仅判断**是否唯一**，唯一才直接用，多个一律问用户（`plans/0026`）。"""
    out: list[dict] = []
    seen: set[int] = set()
    for src in (state.get("focus_shops") or [], (state.get("itinerary") or {}).get("stops") or []):
        for item in src:
            if not isinstance(item, dict):
                continue
            sid = _shop_id_of(item)
            if sid is None or sid in seen:
                continue
            seen.add(sid)
            out.append(item)
    return out


def _booking_draft_from_action(state: HmdpState) -> dict[str, Any] | None:
    """从本轮的**结构化 action** 构造预约草稿（纯函数）；数据来源仅前端点的 `booking` 建议项（经 `action.args` 带回）。
    店名/地址**一律回查行程 `stops`**（`shop_id` 命中才取）—— **不信任 action 里的店名地址**，否则传 `shop_name="米其林"` 就能让 agent 这么播报；时间也**不信任 action**，必须**重新过 `_bookable_time`**（按钮上一轮生成、用户可能挂几分钟才点已过期的，`plans/0019` D3）。"""
    action = state.get("action") or {}
    if not isinstance(action, dict):
        return None
    args = action.get("args") or {}
    if not isinstance(args, dict):
        return None

    try:
        shop_id = int(args.get("shop_id"))
    except (TypeError, ValueError):
        return None

    itinerary = state.get("itinerary") or {}
    stops = {}
    for item in itinerary.get("stops") or []:
        try:
            stops[int(item.get("shop_id"))] = item
        except (TypeError, ValueError):
            continue
    stop = stops.get(shop_id)
    if not stop:
        logger.warning("预约 action 的 shop_id=%s 不在行程里，已拒绝（防幻觉门店）", shop_id)
        return None

    bookable = _bookable_time(
        str(itinerary.get("date") or ""), str(args.get("time") or stop.get("start") or "")
    )
    if not bookable:
        logger.info("预约 action 的时刻已过期或格式不对（shop_id=%s），不挂起", shop_id)
        return None

    return {
        "shop_id": shop_id,
        "shop_name": stop.get("name") or "",
        "address": stop.get("address") or "",
        "booking_time": bookable,
        "people": args.get("people") or (state.get("task_draft") or {}).get("slots", {}).get("people"),
        "remark": str(args.get("remark") or "")[:255],
    }


def _booking_draft_from_focus(state: HmdpState, prefer_shop_id: int | None = None) -> dict[str, Any] | None:
    """从**上下文里的店铺**补一份预约草稿（自然语言入口"帮我预约下"，`plans/0022`）；2026-10-08 真机 id 476/477：「帮我预约下」命中 `_BOOKING_INTENT` 但 `_booking_draft_from_action` 只认前端 booking 建议项 ⇒ 返回 `None` ⇒ 交回模型答"不能预约"，用户觉得没上下文。
    补一条**只读、确定性**来源：本轮焦点店铺（`focus_shops`，`grounding` 反查数据库名称可信）或行程 stops；到店时间/人数不猜，返回空由 `booking_confirm` 挂起（写路径绝不猜参数）。
    选店顺序：**先 `prefer_shop_id`（本轮 `grounding` 刚解析）** → `last_grounded_shop_id` → `focus_shops[0]`；`prefer_shop_id` 须优先于上下文否则"我要预约星聚会KTV"约成上一轮那家（真机 2026-10-08）；该店不在 `focus_shops`（拿不到可信名称）⇒ 返回 `None`，不拿别的店顶包。"""
    slots = (state.get("task_draft") or {}).get("slots") or {}
    focus = [s for s in (state.get("focus_shops") or []) if isinstance(s, dict)]
    if prefer_shop_id is not None:
        for shop in focus:
            if _shop_id_of(shop) == prefer_shop_id:
                return {
                    "shop_id": prefer_shop_id,
                    "shop_name": shop.get("name") or "",
                    "address": shop.get("address") or "",
                    "booking_time": "",           # 未知 ⇒ 走澄清补齐
                    "people": slots.get("people"),
                    "remark": "",
                }
        logger.info("预约草稿：本轮解析到 shop_id=%s，但它不在焦点列表里 ⇒ 不猜店名，交回追问", prefer_shop_id)
        return None
    # 优先"最近一次点名的那家店"（见 `state.last_grounded_shop_id`），
    # 找不到再退回 focus 首位（那是"最近一家"，只在用户没说清哪家时才合理）。
    last_id = state.get("last_grounded_shop_id")
    if last_id is not None:
        for shop in focus:
            if _shop_id_of(shop) == last_id:
                return {
                    "shop_id": last_id,
                    "shop_name": shop.get("name") or "",
                    "address": shop.get("address") or "",
                    "booking_time": "",           # 未知 ⇒ 走澄清补齐
                    "people": slots.get("people"),
                    "remark": "",
                }
    if focus:
        shop = focus[0]
        sid = _shop_id_of(shop)
        if sid is not None:
            return {
                "shop_id": sid,
                "shop_name": shop.get("name") or "",
                "address": shop.get("address") or "",
                "booking_time": "",           # 未知 ⇒ 走澄清补齐
                "people": slots.get("people"),
                "remark": "",
            }
    stops = [s for s in ((state.get("itinerary") or {}).get("stops") or []) if isinstance(s, dict)]
    if stops:
        stop = stops[0]
        sid = _shop_id_of(stop)
        if sid is not None:
            return {
                "shop_id": sid,
                "shop_name": stop.get("name") or "",
                "address": stop.get("address") or "",
                "booking_time": "",
                "people": slots.get("people"),
                "remark": "",
            }
    return None


#: 预约补槽卡给出几个时间候选（按整点，从下一个整点起）
BOOKING_FILL_OPTION_HOURS = 4


def _booking_fill_payload(draft: dict) -> dict[str, Any]:
    """预约**缺到店时间**时的澄清卡（纯函数；`interrupt()` 之前调用）；形状与计划轨 `clarification` **完全一致**（`specs/0012` §12.3）前端零改动可渲染，选项仅**引导**以 `_parse_booking_fill` 解析为准。
    只问**到店时间**：人数拿不到时沿用 `task_draft.slots.people`，再不然由 `booking_payload` 默认值兜底（确认卡把 `people` 列可改字段）；多问一个必填槽多一轮挂起，而"确认卡上改人数"只有一步。"""
    now = datetime.now()
    # 起点必须**严格晚于 `now + 30min`**（与 `_bookable_time` 同口径）：
    #    写成 `(now + 1h).replace(minute=0)` 在 `now=18:36` 时得到 `19:00` ——
    #    只差 24 分钟，用户点了选项反而被判"时间已过期"，是最糟的一种交互。
    anchor = now + timedelta(minutes=30)
    base = anchor.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    options: list[dict[str, Any]] = []
    for i in range(BOOKING_FILL_OPTION_HOURS):
        moment = base + timedelta(hours=i)
        prefix = "今天 " if moment.date() == now.date() else "明天 "
        options.append({
            # 值给**完整时刻**：前端点选项时原样回传，`_parse_booking_fill` 直接认
            "value": moment.strftime("%Y-%m-%d %H:%M"),
            "label": prefix + moment.strftime("%H:%M"),
        })
    name = draft.get("shop_name") or "这家店"
    return {
        "type": "clarification",
        "mode": "booking_fill",
        "shop_id": draft.get("shop_id"),
        "shop_name": draft.get("shop_name") or "",
        "questions": [
            {
                "slot": "booking_time",
                "label": "到店时间",
                "required": True,
                "value_type": "text",
                "unit": "",
                "hint": "比如「今天18:00」",
                "options": options,
            }
        ],
        "text": f"想预约「{name}」，告诉我到店时间就行（比如「今天18:00」，也可以点下面的时间）。",
    }


def _parse_booking_fill(text: str, *, now: datetime | None = None) -> str:
    """从补槽答案里解析到店时刻（`YYYY-MM-DD HH:MM:SS`）；解析不出来返回空串（纯函数）；刻意**不调模型**（写路径参数解析不该有二次"模型理解"，与 `_apply_booking_modification` 不猜相对时间同源）。
    支持三种写法：完整时刻（选项回传 `2026-10-09 18:00`）、`18:00`、`18点`；日期认「今天/明天/后天」（缺省今天）。**已过期一律返回空串**，由调用方收尾。"""
    raw = now or datetime.now()
    s = str(text or "").strip()
    if not s:
        return ""
    day = raw.date()
    if "明天" in s or "明日" in s:
        day = day + timedelta(days=1)
    elif "后天" in s:
        day = day + timedelta(days=2)

    hh = mm = None
    full = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})[ T](\d{1,2}):(\d{2})", s)
    if full:
        day = date(int(full.group(1)), int(full.group(2)), int(full.group(3)))
        hh, mm = int(full.group(4)), int(full.group(5))
    else:
        clock = re.search(r"(\d{1,2})\s*[:：]\s*(\d{2})", s)
        if clock:
            hh, mm = int(clock.group(1)), int(clock.group(2))
        else:
            oclock = re.search(r"(\d{1,2})\s*点", s)
            if oclock:
                hh, mm = int(oclock.group(1)), 0
    if hh is None or not (0 <= hh <= 23 and 0 <= mm <= 59):
        return ""
    moment = datetime(day.year, day.month, day.day, hh, mm)
    if moment < raw + timedelta(minutes=30):
        return ""
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def booking_payload(draft: dict) -> dict[str, Any]:
    """预约确认卡载荷（纯函数；`interrupt()` 之前调用，必须无副作用）。"""
    return {
        "type": "booking_confirm",
        "mode": "confirm",
        "shop_id": draft.get("shop_id"),
        "shop_name": draft.get("shop_name") or "",
        "address": draft.get("address") or "",
        "booking_time": draft.get("booking_time") or "",
        "people": draft.get("people"),
        "remark": draft.get("remark") or "",
        # `editable` 由**后端**给出（`plans/0021` §16.4），前端据此渲染可改字段。
        #    刻意**每次新建列表**而不是返回模块常量：载荷会被 JSON 序列化、
        #    前端也可能改它 ⇒ 共用同一个 list 会有"谁改了它"的跨请求污染。
        "editable": list(BOOKING_EDITABLE_FIELDS),
        "text": (
            f"确认预约「{draft.get('shop_name') or '这家店'}」"
            f" {draft.get('booking_time') or ''}"
            f"（{draft.get('people') or 2} 人）吗？"
        ),
    }


def _booking_result_answer(state: HmdpState, result: str, *, created: bool = True) -> str:
    """把预约结果转成一句话（`result` ∈ success / cancelled / failed）；刻意**不调模型**（下单已发生，让模型润色只引入"看起来成功"风险）。
    `created=False` = **幂等命中既有单、这次没新建** ⇒ 必须换一套说法，否则两者都播报"已为你预约"、用户以为刚下了一单（`plans/0019` 真机验收发现）。"""
    name = (state.get("booking_draft") or {}).get("shop_name") or "这家店"
    when = (state.get("booking_draft") or {}).get("booking_time") or ""
    if result == "success":
        if not created:
            return f"你已经预约过 {name}（{when}）了，这次没有重复提交。"
        return f"已为你预约 {name}（{when}），到店时提一下预约信息即可。"
    if result == "cancelled":
        return "好的，预约已取消，没有提交任何信息。"
    return f"{BOOKING_FAILED}（{name}）"


async def _booking_fill_step(
    state: HmdpState, draft: dict
) -> Command[Literal["booking_confirm", END]]:
    """挂起「补到店时间」的澄清并解析答复（`plans/0022`）；只补**一轮**：解析不出就明确收尾、**不无限挂起**（写路径宁可让用户重说，绝不靠猜约到别的时间，与 `_apply_booking_modification` 不猜相对时间同源）。
    `interrupt()` 之前只做纯计算（组 payload），副作用在之后 —— 本节点会被框架从头重放。"""
    if state.get("booking_fill_tried"):
        logger.info("预约补槽仍未拿到可用时间，收尾（不猜时间）")
        answer = "没听清你想约的时间。想预约的话，直接告诉我「今天18:00」这样的时间就行。"
        return Command(
            goto=END,
            update={
                "answer": answer,
                "sources": [],
                "messages": [AIMessage(content=answer)],
                "booking_draft": None,
                "booking_fill_tried": False,   # 终态清场：不许带到下一次预约
                **_terminal_cleanup(state),
            },
        )

    payload = _booking_fill_payload(draft)
    answer = interrupt(payload)          # ← 挂起；续跑时**直接返回**用户输入
    reply = HumanMessage(content=f"[预约补时间] {answer}")
    booking_time = _parse_booking_fill(str(answer or ""))
    updated = dict(draft)
    if booking_time:
        updated["booking_time"] = booking_time
    logger.info(
        "预约补槽：answer=%r → booking_time=%s",
        str(answer or "")[:40], booking_time or "(未解析出，交回后收尾)",
    )
    return Command(
        goto="booking_confirm",
        update={
            "booking_draft": updated,
            "booking_fill_tried": True,
            "messages": [reply],
        },
    )


async def booking_confirm(state: HmdpState) -> Command[Literal["entry_router", END]]:
    """预约确认：挂起等用户点确认，确认后才调 Java 侧写接口（`plans/0019`）；与 `clarify_slots` 同纪律：`interrupt()` **之前**只允许纯函数（读 State/校验/组 payload），节点会被框架**从头重放**，副作用（HTTP 写）在**之后**。
    重试/弃权判断**必须在 `interrupt()` 之后**，否则拿到 action 不生效；**未登录不挂起**：无 `user_id` 直接交回入口给可读提示（人类决策 D2），挂起提交不了的单比说清楚更糟。"""
    record_node("booking_confirm")
    draft = dict(state.get("booking_draft") or {})
    _action0 = state.get("action") or {}
    _atype0 = str(_action0.get("type") or "") if isinstance(_action0, dict) else ""

    # **确认之后**的取消（`plans/0027`）：草稿已清空、单号在 `booking_result` 里；卡片上「取消预约」走此路径，没有它按钮只能"假装取消"或没入口。`order_id` **只从 State 取**，绝不接受前端传 id（否则传别人单号就能撤）。
    if _atype0 == "cancel" and not draft:
        prev = dict(state.get("booking_result") or {})
        order_id = _evidence_int(prev.get("order_id"))
        if order_id is None:
            logger.info("收到取消动作但没有可撤的单号（booking_result=%s），按未预约处理", prev)
            answer = "当前没有可以取消的预约。"
            return Command(
                goto=END,
                update={"answer": answer, "sources": [],
                        "messages": [AIMessage(content=answer)],
                        "booking_result": {"result": "none", "order_id": None},
                        **_terminal_cleanup(state)},
            )
        # 写操作：同步 httpx ⇒ 必须挪出事件循环（与 `create_booking` 同纪律）
        res = await asyncio.to_thread(cancel_booking, order_id, str(state.get("user_id") or ""))
        ok = bool(res.get("ok") and res.get("cancelled"))
        result = "cancelled" if ok else "failed"
        logger.info("取消预约 order_id=%s ⇒ %s", order_id, result)
        answer = ("好的，预约已取消，没有提交任何信息。" if ok
                  else str(res.get("message") or BOOKING_CANCEL_FAILED))
        return Command(
            goto=END,
            update={
                "answer": answer,
                "sources": [],
                "messages": [HumanMessage(content="[取消预约]"), AIMessage(content=answer)],
                "booking_result": {"result": result, "order_id": order_id},
                **_terminal_cleanup(state),
            },
        )
    if not draft and state.get("booking_pending"):
        # 预约轮走 `entry_router → grounding → 这里`（店名本轮才解析出来），草稿按**本轮 grounding 结果**建而非入口用上一轮上下文。
        # **解析不到就保持空**：绝不退回上下文 —— 那等于把用户点名的店换成上一轮那家（真机 2026-10-08：点名星聚会却问 103茶餐厅）。
        ids = [i for i in (state.get("shop_ids") or []) if isinstance(i, int)]
        if ids:
            draft = _booking_draft_from_focus(state, prefer_shop_id=ids[0]) or {}
        if draft:
            logger.info("预约草稿按本轮 grounding 结果构造 shop_id=%s", draft.get("shop_id"))

    # —— 以下是"挂起前的纯校验"：任何不满足都**不挂起**，直接交回入口重路由 ——
    if not str(state.get("user_id") or "").strip():
        logger.info("预约请求缺 user_id（未登录），不挂起、不下单")
        return Command(
            goto="entry_router",
            update={"answer": BOOKING_NEEDS_LOGIN, "booking_draft": None},
        )
    if not draft.get("shop_id") or not draft.get("booking_time"):
        # 🆕 只有店、没有到店时间（自然语言入口「帮我预约下」，`plans/0022`）⇒
        #    **先挂起澄清补齐时间**，而不是静默交回入口 ——
        #    静默交回的后果实测过：那句话落回模型手里，被答成"我暂时不能帮你预约"（id 477）。
        if draft.get("shop_id") and not draft.get("booking_time"):
            return await _booking_fill_step(state, draft)
        logger.info("预约参数不完整（shop_id/booking_time 缺失），不挂起")
        # 同样 `goto=END`：交回入口会因"续跑不更新 `query`"而**重新命中预约关键词短路** ⇒ 残缺草稿被重建又回到挂起（真机 2026-10-08 循环）；这里给可读的话收尾比"静默重路由"清楚。
        # 话说得**具体**：用户点名店却没解析到时说"没找到这家店"，而非"预约信息不完整"（后者让他怀疑自己没说清）。
        _plan = plan_of(state)
        asked_names = [n for n in ((_plan.shop_names if _plan else None) or []) if n]
        if asked_names and not draft.get("shop_id"):
            answer = f"没找到「{asked_names[0]}」这家店，换个名字或换个说法我再帮你找。"
        else:
            answer = "预约信息不完整，先告诉我想预约哪家店吧。"
        return Command(
            goto=END,
            update={
                "answer": answer,
                "sources": [],
                "messages": [AIMessage(content=answer)],
                "booking_draft": None,
                **_terminal_cleanup(state),
            },
        )

    payload = booking_payload(draft)
    # 上一次 `modify` 被拒的原因（`plans/0021` §16.2）—— 带进**这次**挂起的卡片。
    #    校验失败时我们让节点**从头再跑一遍**重新挂起（不 goto END：那等于把卡收了），
    #    这时若不把原因带上去，用户会看到一张**一模一样**的卡，不知道自己改错了什么。
    pending_why = str(state.get("booking_modify_error") or "")
    if pending_why:
        payload["text"] = f"{payload['text']}\n（{pending_why}）"

    action = interrupt(payload)      # ← 挂起；续跑时**直接返回**前端的结构化 action
    action = {} if action is None else action
    # 见 docstring：续跑不会自动把用户输入写进 state.messages，必须自己回填
    reply = HumanMessage(content=f"[预约确认] {payload['text']}")

    action_type = (str(action.get("type") or "") if isinstance(action, dict) else "")

    # —— `modify` 分支**必须排在下面"一律取消"之前**（`plans/0021` F3 / T2）——
    #    顺序反了的话功能"看起来实现了"但永远走不到：modify 会被当取消，
    #    用户点"改时间"得到"预约已取消"。`test_booking_modify` 有源码级顺序守卫。
    if action_type in _MODIFY_ACTIONS:
        done = int(state.get("booking_modify_count") or 0)
        if done >= BOOKING_MODIFY_MAX:
            # §16.2：超过上限不再挂起（否则前端 bug 反复 modify ⇒ 无限挂起）
            logger.info("预约修改次数超上限（%s/%s），交回入口", done, BOOKING_MODIFY_MAX)
            too_many = f"预约信息已经改过 {done} 次了，直接告诉我新的时间或人数就行。"
            return Command(
                # **必须 `goto=END`，不能再 `goto entry_router`**（2026-10-08 真机循环）：续跑时**不更新 `state["query"]`**仍上一句"帮我预约下" ⇒ 入口预约关键词短路**再次命中**，用无时间草稿新建又回补时间挂起，用户看不到"已改 N 次"。
                # END 之后下一条消息天然新回合（query 更新），入口行为正常。
                goto=END,
                update={
                    "answer": too_many,
                    "sources": [],
                    "messages": [reply, AIMessage(content=too_many)],
                    "booking_draft": None,
                    "booking_modify_error": "",   # 终态清场：陈旧原因不许带到下一次预约
                    "booking_fill_tried": False,  # 同上：补时间标记也属于"这一张卡"
                    **_terminal_cleanup(state),
                },
            )

        merged, why = _apply_booking_modification(
            draft, action.get("args") if isinstance(action, dict) else None
        )
        if why:
            # 校验不过：**不写单**。把原因写进 State 并**重入本节点**重新挂起。
            # 刻意**不 goto END**（那等于收卡、用户没机会改），也**不二次 interrupt**（同节点调第二次不可靠），而是节点从头再跑：`booking_modify_error` 把原因带进新卡片。
            logger.info("预约修改被拒：%s", why)
            return Command(
                goto="booking_confirm",
                update={
                    "booking_draft": dict(draft),      # 原值不变
                    "booking_modify_count": done + 1,  # 改错也算次数：否则能无限试错
                    "booking_modify_error": why,
                    "messages": [reply, HumanMessage(content=f"[预约修改被拒] {why}")],
                },
            )

        logger.info("预约信息已修改（原 %s / %s 人 → 新 %s / %s 人），重新挂起确认卡",
                    draft.get("booking_time"), draft.get("people"),
                    merged.get("booking_time"), merged.get("people"))
        return Command(
            goto="booking_confirm",
            update={
                "booking_draft": merged,
                "booking_modify_count": done + 1,
                "booking_modify_error": "",            # 改成功了，清掉上次的原因
                # **不写 booking_result**：这轮既没确认也没取消，
                #    写任何结果都会让前端以为"预约已结束了"。
                "messages": [reply, HumanMessage(content="[预约修改] 已按你的要求更新")],
            },
        )

    # —— 以下只在"拿到 action 之后"执行（每轮续跑至多一次）——
    # 白名单之外一律取消：写操作的默认分支必须是"不写"（D1）。
    if action_type not in _CONFIRM_ACTIONS:
        logger.info("预约 action 非确认（type=%r），按取消处理", action_type or type(action).__name__)
        return Command(
            goto=END,
            update={
                "answer": _booking_result_answer(state, "cancelled"),
                "sources": [],
                "messages": [reply, AIMessage(content="好的，预约已取消，没有提交任何信息。")],
                # 取消也要写结果：否则刷新后前端**看不到**"已取消"，
                #    会以为还挂着一个待确认的卡（落 `payload` 列供回看）。
                "booking_result": {"result": "cancelled", "order_id": None},
                "booking_draft": None,
                "booking_modify_error": "",   # 终态清场：陈旧原因不许带到下一次预约
                    "booking_fill_tried": False,  # 同上：补时间标记也属于"这一张卡"
                **_terminal_cleanup(state),
            },
        )

    # 写操作**只在这一处**发生；`create_booking` 内部是同步 httpx ⇒ 必须挪出事件循环。
    result = await asyncio.to_thread(
        create_booking,
        int(draft["shop_id"]),
        str(draft["booking_time"]),
        draft.get("people"),
        str(draft.get("remark") or ""),
        str(state.get("user_id") or ""),
    )
    outcome = "success" if result.get("ok") else "failed"
    if not result.get("ok"):
        logger.error("预约失败 shop_id=%s: %s", draft.get("shop_id"), result.get("message"))
    # `created` 区分"新建"与"幂等命中既有单"：后者**没有新建**，文案与结果载荷都要如实体现。
    created = bool(result.get("created", True))
    if outcome == "success" and not created:
        logger.info("预约幂等命中（未新建）shop_id=%s order_id=%s", draft.get("shop_id"), result.get("order_id"))
    answer = _booking_result_answer(state, outcome, created=created)
    return Command(
        goto=END,
        update={
            "answer": answer,
            "sources": [],
            "messages": [reply, AIMessage(content=answer)],
            # 结构化结果落 payload（刷新后可回看"约了什么"）；`created` 一并落，前端可据此提示"已预约过"
            "booking_result": {
                "result": outcome,
                "order_id": result.get("order_id"),
                "created": created,
                # **连当时那张卡一起落**（`plans/0027`）：确认之后 `booking_draft` 被清空了，
                #    只留一个 order_id 的话，刷新后前端**还原不出这张卡** ⇒ 用户既看不到
                #    "已预约"，也没有「取消预约」的入口。落的是**已提交的参数原文**，不含猜测。
                **({"booking": dict(draft)} if outcome == "success" else {}),
            },
            "booking_draft": None,
            "booking_modify_error": "",   # 终态清场：陈旧原因不许带到下一次预约
                    "booking_fill_tried": False,  # 同上：补时间标记也属于"这一张卡"
            **_terminal_cleanup(state),
        },
    )


# ===========================================================================
# 第二层：Grounding——把用户的表述解析成 shop_ids
# ===========================================================================
def _ordinal_index(query: str) -> int | None:
    """单数指代的**下标**（与 `_indices_to_ids` 同语义：`-1` = 倒数第一）。

    支持「第N家」（正下标）、「倒数第N家」、「最后一家 / 最后那家」（`-1`）。
    注意本函数**只负责单数**；「最后两家」是复数，家数/区间由 `_plural_count` + 这里返回的
    **起点下标**共同表达（见 `grounding` 里 `_plural_span`）。
    """
    match = re.search(r"第([一二两三四五1-5])家", query)
    if match:
        pos = _ORDINAL.get(match.group(1))
        if pos is None:
            return None
        # 「倒数第N家」⇒ 负数下标（Python 语义：-1 是最后一家）
        if re.search(r"倒数第" + re.escape(match.group(1)) + r"家", query):
            return -(pos + 1)
        return pos
    if "最后一家" in query or "最后那家" in query or "最后这家" in query:
        return -1
    # 「最后两家 / 最后三家」⇒ 区间起点（负数），家数由 `_plural_count` 给
    match = re.search(r"最后[这那]?([二两三四五2-5])[家个]", query)
    if match:
        idx = _ORDINAL.get(match.group(1))
        return None if idx is None else -(idx + 1)
    return None


def _span_by_start_count(start: int, count: int, total: int) -> list[int]:
    """把「起点下标 + 家数」折成**绝对下标列表**（`grounding` 正则兜底用）。

    - **负起点**：`start=-2, count=2, total=3` ⇒ `[-2,-1]` ⇒ `[1,2]`（最后两家）。
    - **越界裁剪**：`total=3, start=1, count=5` ⇒ `[1,2]`（能给几家给几家，不报错）。
    - 家数 ≤ 0 或 `total` ≤ 0 ⇒ `[]`（调用方会退化到"取焦点第一家"以外的兜底路径）。
    """
    if total <= 0 or count <= 0:
        return []
    begin = start + total if start < 0 else start
    begin = max(0, min(begin, total - 1))
    return list(range(begin, min(begin + count, total)))


def _indices_to_ids(indices: Any, focus: list[dict]) -> list[int]:
    """把入口抽出的**指代下标**翻译成 focus 里的店 id（`specs/00xx`）。
    - **负索引**：`-1` = 倒数第一（Python 语义，与 `_ordinal_index` 的 `-1` 一致），`-2` = 倒数第二；
      「最后两家」⇒ `[-2, -1]` ⇒ 天然取到末尾两家，无需单独表达"反向区间"。
    - **越界丢弃**：`focus` 只有 3 家而模型给了 `[0,1,5]` ⇒ 丢掉 5（**不报错、不回退全量**，
      能解析几家就几家；全越界 ⇒ 返回 `[]` ⇒ 调用方回退正则兜底）。
    - **去重保序**：重复下标（模型有时会重复输出）只保留首次出现。
    - 非 `int` 项丢弃（validator 已清洗过一轮，这里是**双保险**：本函数也可能被别的路径直接调）。
    :returns: 店 id 列表；无有效项返回 `[]`。
    """
    out: list[int] = []
    seen: set[int] = set()
    total = len(focus)
    for raw in indices or []:
        if isinstance(raw, bool) or not isinstance(raw, int):
            continue
        idx = raw + total if raw < 0 else raw      # 负索引归一成绝对下标
        if idx < 0 or idx >= total:
            continue                                # 越界丢弃
        shop = focus[idx]
        sid = shop.get("id") if isinstance(shop, dict) else None
        if sid is None:
            continue
        sid = int(sid)
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


# ---- 「按店名反查」的三种结局 ----
#: `fetch_shops_by_name` 的契约是 `list | str`（失败与业务为空都返回字符串）；旧实现把三者一律压成 `hit = None`，于是"这家店不存在"与"店铺服务挂了"在 State 上完全不可区分，最后双双落到 `route_after_grounding` 的 catch-all → `clarify`，反问用户"你想了解哪家店铺呢"（而用户明明已经说了店名）。见 `bug/BUG-20260928-grounding-named-shop-to-clarify.md`。
_LOOKUP_HIT = "hit"
_LOOKUP_NOT_FOUND = "not_found"
_LOOKUP_ERROR = "error"

#: 店铺类下游调用技术性失败（连不上 / 非 200 / 超时 / 坏 JSON）时的唯一用户文案。
#: 它经 `render_context` 进上下文、被汇总模型原样转达给用户，而原始错误带 URL、异常类型、`WinError 10061` —— 用户看不懂且等于把内部拓扑说出去，所以原始错误只允许进日志（`logger.error`），本条替换发生在所有出口；`grounding` 与 `nearby_worker` 共用这一句。
SHOP_LOOKUP_FAILED = "店铺信息暂时查不到，请稍后再试。"


async def _lookup_one_shop(name: str, user_id: str) -> tuple[dict | None, str, str]:
    """按店名反查单家店，并保留失败原因。Returns: `(命中店铺 or None, 结局（`_LOOKUP_*`）, 失败原因原文)`。
    第三个返回值只允许进日志，不得进入上下文或用户答复 —— 用户需要知道"店铺信息暂时查不到"，不需要看到 URL 和异常类型。"""
    reasons: list[str] = []  # 非「没找到」的失败原因（接口/传输层），优先级高于「没找到」
    item = await asyncio.to_thread(fetch_shops_by_name, name, 1, user_id)
    if isinstance(item, list) and item:
        return item[0], _LOOKUP_HIT, ""
    if isinstance(item, str) and item and item != SHOP_NAME_NOT_FOUND.format(name=name):
        reasons.append(item)

    # 整串匹配失败时，去掉「餐厅/火锅/店」等通用品类后缀重试。
    # 后端 /shop/of/name 按店名子串匹配，而「新白鹿餐厅」的店名实为「新白鹿(...)」。
    norm = _normalize_shop_name(name)
    if norm and norm != name:
        item2 = await asyncio.to_thread(fetch_shops_by_name, norm, 1, user_id)
        if isinstance(item2, list) and item2:
            logger.info("店名规范化重试命中: %r -> %r (id=%s)", name, norm, item2[0].get("id"))
            return item2[0], _LOOKUP_HIT, ""
        if isinstance(item2, str) and item2 and item2 != SHOP_NAME_NOT_FOUND.format(name=norm):
            reasons.append(item2)

    if reasons:
        return None, _LOOKUP_ERROR, reasons[0]
    return None, _LOOKUP_NOT_FOUND, ""


def _grounding_notice(outcome: str, names: list[str], reason: str) -> dict:
    """构造"点名店铺没解析出来"的说明，交给下游当答复 / 当资料用。
    - 结局 = `error` → 独立用户文案（原始错误只进日志，不泄露技术细节）；- 结局 = `not_found` → 直说没找到并给一句引导（由 `clarify` 拼成完整答复）。"""
    label = "、".join(f"「{n}」" for n in names[:MAX_SHOPS])
    if outcome == _LOOKUP_ERROR:
        logger.error("店名反查失败（%s）：%s", label, reason)
        return {
            "worker": "grounding",
            "outcome": _LOOKUP_ERROR,
            "notice": SHOP_LOOKUP_FAILED,
            # 点名列表（`specs/0016` T4）：供 `[死路]` 投影读取"点名却没找到"的店名。
            "names": list(names[:MAX_SHOPS]),
        }
    return {
        "worker": "grounding",
        "outcome": _LOOKUP_NOT_FOUND,
        "notice": f"没找到{label}{'这家店' if len(names[:MAX_SHOPS]) == 1 else '这几家店'}",
        "hint": "你可以换个店名，或者说说想找什么类型的店（比如火锅、咖啡）。",
        # 点名列表（`specs/0016` T4）：未找到的点名店 ⇒ 进 `[死路]`（Q4）。
        "names": list(names[:MAX_SHOPS]),
    }


async def grounding(state: HmdpState) -> dict:
    """解析出用户真正想问的店铺 id。绝大多数情况下是纯规则 + 查库，不调模型。"""
    record_node("grounding")
    plan = plan_of(state)
    query = state.get("query") or ""
    user_id = state.get("user_id") or ""
    focus = list(state.get("focus_shops") or [])

    if plan is None:
        return {"shop_ids": []}
    if plan.need_clarify and not plan.shop_names:
        return {"shop_ids": []}

    ids: list[int] = []
    names = [n for n in (plan.shop_names or []) if n]

    # 1) 指代消解：这家 / 第二家 / 这三家 —— 依赖跨轮焦点，这是 focus_shops 必须入 State 的原因
    #    **入口抽取为主链**（`specs/00xx`）：入口模型已把 `target_indices` 抽出来（0 基、-1=倒数第一），
    #    这里直接用 ⇒ 「最后两家」「前两家」这类正则表达不了的说法也能解析。
    #    抽不到（`[]`）才走下面的正则兜底 —— 正则**不是**主路径，只是"模型没抽/抽不动"时的保险，
    #    这样"指代解析失败"永远不会把整轮打崩（最差退化成旧行为）。
    if not names and focus and plan.target_indices:
        picked_ids = _indices_to_ids(plan.target_indices, focus)
        if picked_ids:
            ids.extend(picked_ids)
            logger.info("指代消解（入口抽取）：indices=%s → shop_ids=%s",
                        list(plan.target_indices), picked_ids)

    # 1b) 兜底：入口没抽到指代 ⇒ 走正则（旧逻辑，仅"单数/复数二选一"的判据被修，见下）
    if not ids and not names and _ANAPHORA.search(query) and focus:
        # 🔴 判据顺序：**先问"是不是单数序号"，再问"是不是复数"**。
        #    旧版先判 `_PLURAL_ANAPHORA`，而「最后一家」里的 `一家` 会被复数正则吃到
        #    （`(?<!第)` 拦不住，因为前面是 `最` 不是 `第`）⇒ 走到 `focus[:1]` ⇒
        #    「那最后一家怎么样」取到**第一家**，而 `_ordinal_index` 明明会返回 `-1`。
        #    现在单数序号有值就不进复数分支，语义不会两头抢。
        idx = _ordinal_index(query)
        count = _plural_count(query)
        if count is not None and count > 1:
            # 复数：家数由 `count` 表达，起点由 `idx` 表达（`最后两家` ⇒ idx=-2, count=2）
            start = idx if idx is not None else 0
            span = _span_by_start_count(start, count, len(focus))
            ids.extend(int(focus[i]["id"]) for i in span if focus[i].get("id") is not None)
        else:
            if idx is None:
                pick = focus[0]
            else:
                pick = focus[idx] if -len(focus) <= idx < len(focus) else focus[0]
            ids.append(int(pick["id"]))

    # 2) query 里直接出现了焦点店铺名
    if not ids and focus:
        for shop in focus:
            name = str(shop.get("name") or "")
            if name and name in query:
                ids.append(int(shop["id"]))
                break

    # 3) 按店名检索（支持一次提到多家店）
    # 命中的店同步沉淀为跨轮焦点：这样「先分别问 A/B/C，再问这三家对比」时，
    # 三家都在 focus_shops 里（累积 reducer 负责去重追加），指代解析才能落到实体上。
    grounded_focus: list[dict] = []
    lookup_outcome = ""
    lookup_reason = ""
    if not ids and names:
        for name in names[:MAX_SHOPS]:
            hit, outcome, reason = await _lookup_one_shop(name, user_id)
            if outcome == _LOOKUP_HIT and hit is not None:
                ids.append(int(hit["id"]))
                focus_item = _to_focus_item(hit)
                if focus_item is not None:
                    grounded_focus.append(focus_item)
                continue
            # 以**第一个**失败为准（用户通常只点一家店）。
            # 只要有任何一家命中，`ordered` 就非空，下游不会用这条说明。
            if not lookup_outcome:
                lookup_outcome, lookup_reason = outcome, reason

    # 去重保序
    ordered: list[int] = []
    seen: set[int] = set()
    for sid in ids:
        if sid not in seen:
            seen.add(sid)
            ordered.append(sid)

    logger.info("Grounding 解析到 shop_ids=%s（历史焦点 %s 家，本轮新增 %s 家）",
                ordered, len(focus), len(grounded_focus))
    result: dict = {"shop_ids": ordered}
    if grounded_focus:
        result["focus_shops"] = grounded_focus
        # 记下"用户最近一次**点名**的那家店"（`plans/0022`）：打字"帮我预约下"要靠它定位，
        #    否则只能取 `focus_shops[0]`（那是"最早出现 + 最近的一家"），与刚聊的店无关。
        #    只在**点名解析成功**时写：附近的列表类问句没有"指定哪家"，不该覆盖它。
        last_id = _shop_id_of(grounded_focus[-1])
        if last_id is not None:
            result["last_grounded_shop_id"] = last_id
    # 用户点名了店、却一个都没解析出来 ⇒ 必须把"为什么"带给下游：
    # 否则 `route_after_grounding` 的 catch-all 只会反问"你想了解哪家店铺呢"，
    # 而用户明明已经给了店名（见 bug/BUG-20260928-grounding-named-shop-to-clarify.md）。
    if names and not ordered and lookup_outcome:
        result["worker_outputs"] = [_grounding_notice(lookup_outcome, names, lookup_reason)]
    return result


def _guess_shop_type(query: str, user_id: str) -> str:
    """不调模型，用店铺类型列表对 query 做包含匹配。"""
    types = fetch_shop_types(user_id)
    if isinstance(types, str) or not types:
        return ""
    for item in types:
        name = str(item.get("name") or "")
        if name and name in query:
            return name
    return ""


# 店名里常被用户带上的通用品类/后缀词。后端 /shop/of/name 按店名子串精确匹配，
# 而真实店名往往不含这些词（如「新白鹿餐厅」→ 店名是「新白鹿(运河上街店)」），
# 直接拿 LLM 抽出的「新白鹿餐厅」去搜会整串匹配失败。这里做规范化兜底重试。
_SHOP_NAME_NOISE = (
    "餐厅", "饭店", "餐馆", "酒楼", "饭馆", "火锅", "烧烤", "串串", "香锅",
    "咖啡", "奶茶", "茶饮", "饮品", "甜品", "面包", "烘焙",
    "店", "门店", "商家", "超市", "商场", "广场", "酒店", "宾馆",
    "KTV", "ktv", "酒吧", "银行", "医院", "学校", "中学", "小学",
)


def _normalize_shop_name(name: str) -> str:
    """去掉店名里的通用品类/后缀词并压掉空格，便于后端 /shop/of/name 按子串匹配（「新白鹿餐厅」→「新白鹿」）；入口模型偶发在数字与中文间插空格会整串落空，剥空则原样返回。"""
    s = "".join((name or "").split())  # 去掉首尾与**内部**空白（含全角空格）
    if not s:
        return s
    for w in _SHOP_NAME_NOISE:
        if w and w in s:
            s = s.replace(w, "")
    s = s.strip()
    return s if s else (name or "").strip()


def _to_focus_item(shop: Any) -> dict | None:
    """把后端返回的店铺对象压成跨轮焦点项；无有效 id 返回 None。
    nearby 结果、按店名检索命中的结果、以及 `_collect_focus` 里的沉淀都共用这一份，避免字段名（后端是 `avgPrice`、State 里统一叫 `avg_price`）在多处各写一遍。
    主键一律走 `_shop_id_of`（`id` → 兼容旧 `shop_id`）—— 全仓唯一的读侧入口。"""
    sid = _shop_id_of(shop)
    if sid is None:
        return None
    return {
        "id": sid,
        "name": shop.get("name"),
        "distance": shop.get("distance"),
        "avg_price": shop.get("avgPrice"),
        "score": shop.get("score"),
    }


# ---- 形状契约：「一店一条」记录（specs/0009 §2.1）----
# 病灶是"一店一条"的形状分三次长出来（`_one_shop_bundle` 的 compare 稿、voucher/rag 分支各加的扁平 items、`react_loop` 又自造一套），消费方只能用 `item.get(…) or []` 宽容读取 ⇒ 错配既不报错也不被测试发现，代价是静默失效（compare 候选恒空 / ReAct 产出渲染成 `【店铺None】`）。见 bug/BUG-20260928-agent-payload-shape-mismatch.md。
# 这里把"记录长什么样"收敛成一个工厂：四个产出方都调它，键集合天然完全相等（不是"⊇"，是"=="），测试才能断言集合相等而不是逐字段打点。

#: 记录里的**显式能力标签**。取值与 `RoutePlan.sub_intents` **同域**（只取其中"消费店铺数据"的那三个），
#: 并且与 `CONSUMES_SHOP_IDS` **同集合**（后者直接由它派生，从结构上杜绝漂移）。
SHOP_RECORD_TASKS: frozenset[str] = frozenset({"detail", "voucher", "rag"})

#: 「一店一条」记录的**必填键**。工厂无条件写满整份 —— 「有没有这份数据」由 `tasks` 声明，
#: **不由"键在不在"反推**（那正是本规格要消灭的形状嗅探）。
SHOP_RECORD_KEYS: frozenset[str] = frozenset(
    {
        "id",                                  # 店 id —— **唯一主键名**（写侧不再出现 shop_id）
        "name",                                # 店名；拿不到时工厂填「店铺{id}」
        "tasks",                               # 显式能力标签（detail / voucher / rag）
        "vouchers", "vouchers_error",           # 券块        （tasks 含 voucher 时有意义）
        "order_cards",                          # 下单卡片     （键名不变，hmdp_agent 的 card 事件依赖它）
        "chunks", "chunks_error",               # 细节检索块   （tasks 含 rag 时有意义）
        "blogs", "blogs_error",                 # 攻略块       （tasks 含 detail 时有意义）
        "error",                                # 该店取数失败的用户向短句
    }
)

#: 记录允许出现的**店维度可选键**（有则放顶层，无则不写该键）。
#: 前四个正是 `_narrow_by_depends_on` 的 `key_map` 维度 —— 收窄的算法前提是它们在顶层，埋在嵌套 `shop` 里会永不收窄。
#: `x`/`y` 是坐标（`specs/0012`：行程/地图按坐标画点、算站间直线距离），不参与收窄。
SHOP_RECORD_OPTIONAL_KEYS: frozenset[str] = frozenset(
    {"area", "address", "distance", "avgPrice", "score", "sold", "openHours", "x", "y"}
)

#: 允许在工厂里**覆盖**的键 = 能力块字段 + `error` + 店维度键（`id`/`name`/`tasks` 由位置参数给）。
_SHOP_RECORD_OVERRIDABLE: frozenset[str] = (
    SHOP_RECORD_KEYS - {"id", "name", "tasks"}
) | SHOP_RECORD_OPTIONAL_KEYS

#: 每个能力标签**拥有**的块字段。用于校验"填了某块的数据、却没声明对应标签"这类手滑。
_TASK_BLOCK_FIELDS: dict[str, frozenset[str]] = {
    "detail": frozenset({"blogs", "blogs_error"}),
    "voucher": frozenset({"vouchers", "vouchers_error", "order_cards"}),
    "rag": frozenset({"chunks", "chunks_error"}),
}


def _new_shop_record(
    shop_id: int,
    name: str | None = None,
    tasks: "tuple[str, ...] | list[str]" = (),
    **fields: Any,
) -> dict[str, Any]:
    """「一店一条」记录的唯一构造函数（specs/0009 §2.1）。
    - 无条件写满 `SHOP_RECORD_KEYS`（未勾选的能力块填空值 `[]` / `""`）⇒ 键集合与 `tasks` 无关地恒定，消费方可以放心 `rec["vouchers"]` 直取；`tasks` 只能是 `SHOP_RECORD_TASKS` 的子集。
    - `fields` 只接受 `_SHOP_RECORD_OVERRIDABLE`（能力块字段 / `error` / 店维度键），未知键抛错；填了某能力块的数据却没在 `tasks` 里声明对应标签 ⇒ 抛错（形状自相矛盾）。"""
    task_list = [str(t) for t in tasks]
    unknown_tasks = set(task_list) - SHOP_RECORD_TASKS
    if unknown_tasks:
        raise ValueError(
            f"未知能力标签 {sorted(unknown_tasks)}；合法标签={sorted(SHOP_RECORD_TASKS)}"
        )

    unknown = set(fields) - _SHOP_RECORD_OVERRIDABLE
    if unknown:
        raise ValueError(
            f"未知的记录键 {sorted(unknown)}；可覆盖键={sorted(_SHOP_RECORD_OVERRIDABLE)}"
        )

    for task, block in _TASK_BLOCK_FIELDS.items():
        if task in task_list:
            continue
        touched = block & set(fields)
        if touched:
            raise ValueError(
                f"填了 {sorted(touched)} 却没在 tasks 里声明「{task}」标签（形状自相矛盾）"
            )

    record: dict[str, Any] = {
        "id": int(shop_id),
        "name": name or f"店铺{shop_id}",
        "tasks": task_list,
        "vouchers": [],
        "vouchers_error": "",
        "order_cards": [],
        "chunks": [],
        "chunks_error": "",
        "blogs": [],
        "blogs_error": "",
        "error": "",
    }
    record.update(fields)
    return record


def _shop_id_of(rec: Any) -> int | None:
    """记录主键的唯一读侧入口：`id` → 兼容旧 `shop_id` → `None`。
    写侧只写 `id`；兼容分支只为 checkpoint 里的旧记录留一条命（`worker_outputs` 是 `add_and_clear` 的短生命周期字段，但"本轮写入后崩溃重启"的窗口内仍可能读到旧形状）。"""
    if not isinstance(rec, dict):
        return None
    val = rec.get("id")
    if val is None:
        val = rec.get("shop_id")  # 兼容旧记录；写侧已不再产出该键
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _shop_dims(src: Any) -> dict[str, Any]:
    """从上游店铺对象里挑出店维度可选键（只挑在 `SHOP_RECORD_OPTIONAL_KEYS` 里且非 `None` 的）。
    上游有两个来源（`fetch_shop_detail` 的扁平 dict、`fetch_shops_nearby` / `shops_by_name` 的元素），键名一致，所以"哪些维度上顶层"只在一处挑，避免各产出点各写一遍而漂移。"""
    if not isinstance(src, dict):
        return {}
    return {k: src[k] for k in SHOP_RECORD_OPTIONAL_KEYS if src.get(k) is not None}


def _shop_header(rec: dict) -> str:
    """记录的小节标题：`【店铺{id} · {name}】`；拿不到真名时退化为 `【店铺{id}】`。
    工厂会给 `name` 填占位名「店铺{id}」，这里必须与占位名等值比较再决定是否追加，否则"只查到券、拿不到店名"的记录会渲染成「【店铺9 · 店铺9】」。"""
    sid = _shop_id_of(rec)
    name = str(rec.get("name") or "")
    if name and name != f"店铺{sid}":
        return f"【店铺{sid} · {name}】"
    return f"【店铺{sid}】"


#: worker 任务的**固定优先级**（决定合并记录里 `tasks` 的顺序）。
#: 必须是**确定序**而不是 set 迭代序：顺序决定渲染小节顺序，
#: 不确定的顺序会让同一批数据两次渲染出的字符串不等（验收 17 的"乱序喂入结果相同"就无从谈起）。
WORKER_TASK_PRIORITY: tuple[str, ...] = ("detail", "voucher", "rag")

#: 每个任务降级时该写哪个块字段（块级降级：只坏这一块，其余块照常）。
# 以下三项**定义**在 `degrade.py`（降级文案单点真源，`specs/0015` T3）。
#    这里保留同名引用：调用点一行都不用改（纯重构），但新增文案必须去 `degrade.py` 加。
_DEGRADED_ERROR_FIELD = degrade._DEGRADED_ERROR_FIELD
_TASK_LABEL = degrade._TASK_LABEL


def _degraded_block_error(task: str) -> str:
    """熔断短路时该块的**用户向**降级说明 —— 实现在 `degrade.py`（单点真源）。"""
    return degrade.degraded_block_error(task)


def _ordered_tasks(tasks: "Any") -> tuple[str, ...]:
    """按 `WORKER_TASK_PRIORITY` 去掉重复与非法值，产出**确定序**的任务元组。"""
    names = {str(t) for t in tasks or ()}
    return tuple(t for t in WORKER_TASK_PRIORITY if t in names)


def _blocked_record(sid: int, tasks: "tuple[str, ...]") -> dict:
    """所有任务都被熔断短路时的记录：只有降级说明、零实际取数（R5）。
    与 `_shop_record_for` 逐块降级产出的形状完全一致（同一个工厂 + 同一批文案），这样"全熔断"与"部分熔断"两条路径在渲染层看来没有区别。"""
    return _new_shop_record(
        sid,
        tasks=tasks,
        **{_DEGRADED_ERROR_FIELD[t]: _degraded_block_error(t) for t in tasks},
    )


def _detail_fields(bundle: dict, with_rag: bool) -> dict[str, Any]:
    """从 `_one_shop_bundle` 的产出里挑出可写进合并记录的字段。
    必须按标签挑、不能整个 `update`：工厂会拒绝"填了某块却没声明对应标签"，而 `bundle` 里每个能力块键都存在（工厂无条件写满）⇒ 整拷会把 `chunks` 带进一条只声明了 `detail` 的记录，直接抛 `ValueError`。"""
    fields: dict[str, Any] = {
        "blogs": bundle.get("blogs") or [],
        "blogs_error": bundle.get("blogs_error") or "",
    }
    for key in SHOP_RECORD_OPTIONAL_KEYS:   # 店维度：收窄的算法前提（必须在顶层）
        if bundle.get(key) is not None:
            fields[key] = bundle[key]
    if with_rag:
        fields["chunks"] = bundle.get("chunks") or []
        fields["chunks_error"] = bundle.get("chunks_error") or ""
    if bundle.get("error"):
        fields["error"] = bundle["error"]
    return fields


def _worker_tasks_of(state: Any) -> tuple[str, ...]:
    """本次 Send 要做的任务集合：`worker_tasks` 优先，回退旧载荷的单值 `task`。
    为什么要两个都读（`plans/0009.2` N1 / R3）：Send 载荷在新旧版本间会共存 —— checkpoint 里已落库的旧会话 worker 入参只有 `task: str`，把它改成 list 会让旧会话反序列化后行为错乱，所以只加新字段、不改老字段类型。"""
    raw = state.get("worker_tasks") if isinstance(state, dict) else None
    names = {str(t) for t in raw} if isinstance(raw, (list, tuple, set)) else set()
    if not names:
        legacy = state.get("task") if isinstance(state, dict) else None
        if legacy:
            names = {str(legacy)}
    return _ordered_tasks(names)


# ---- 第三层：Worker（定位 + 扇出取数）----
# 后端 /shop/of/type 对部分类型（如"美食"）返回空，而 /shop/of/name 按名称能命中；
# 这里把"规范类型名"映射到用户口语化的名称搜关键字，作为回退候选。
_TYPE_NAME_KEYWORDS: dict[str, list[str]] = {
    "美食": ["餐厅", "美食", "饭店", "餐馆"],
    "火锅": ["火锅"],
    "KTV": ["KTV"],
    "咖啡": ["咖啡"],
    "酒吧": ["酒吧"],
    "奶茶": ["奶茶"],
}


def _nearby_business_empty(type_name: str) -> tuple[str, ...]:
    """`fetch_shops_nearby` 的两种业务为空文案（类型不存在 / 该类型附近没数据）：它们是"查过了，确实没有"、不是"没查成"，必须与取数失败区分开，否则会把"服务挂了"说成"附近没有"、让用户误以为已经查过了。
    判据是与工具层共享模板的等值比较，不是前缀嗅探（模板改字这里会自动跟随，而 `startswith` 会静默失效 —— 同 `grounding` 里对 `SHOP_NAME_NOT_FOUND` 的口径）。"""
    return (
        SHOPS_NEARBY_EMPTY.format(type_name=type_name),
        SHOP_TYPE_NOT_FOUND.format(type_name=type_name),
    )


def _narrow_candidates_by_plan(plan: "RoutePlan | None", shops: list) -> list[int]:
    """收窄前置：用候选里已经带全的 4 个维度，在发 Send 之前把目标店定下来。
    否则第 2 批的消费类 Send 已在 `route_after_nearby` 按候选全集发出、收窄只发生在 `react_loop` 内 ⇒"附近最便宜的那家有什么券"会先给 3 家候选各取一次券、进环后又给那家再取一次（共 4 次，3 次白取）；而收窄要的 `avgPrice` / `score` / `distance` / `sold` 在 nearby 候选里早已带全。
    判据保守（R6）：只在"候选 >1 且该维度有取值 且 结果条数 == 请求的 `top_k`"时收窄；维度非法 / 全部缺值 / 候选池不足 ⇒ 返回 `[]`，行为与阶段一完全一致（进环兜底）。
    `specs/00xx`：`top_k` 来自 `plan_top_k`（用户明说的 `top_n` 优先，其次指代 `target_indices` 的长度），
    支持"评分最高的两家"这类多店需求 —— **含没有指代、只说家数**的问句（`plans/0050` 缺口 b）。
    :returns: 收窄后的店 id 列表（**保序**）；未收窄返回 `[]`。
    """
    if not plan or not plan.target_depends_on_result:
        return []
    candidates = [s for s in (shops or []) if isinstance(s, dict) and _shop_id_of(s) is not None]
    if len(candidates) <= 1:
        return []
    top_k = plan_top_k(plan)
    # `_narrow_by_dimensions` 在"全部维度都缺值 / 维度非法"时**原样返回** ⇒ 长度不等于请求的
    # `top_k`（除非候选恰好就这么多），于是这里天然不会臆造排序（与 `react_loop` 里那道收窄
    # 共用同一个算法与同一份维度判据）。
    top = _narrow_by_dimensions(candidates, _plan_ranking_dimensions(plan), top_k)
    if len(top) != top_k:
        return []
    return [sid for sid in (_shop_id_of(s) for s in top) if sid is not None]


def _max_nearby_types() -> int:
    """多活动一次最多解析的**类型/活动**数（`specs/0030`）。默认 3。**调用时读 env。**"""
    try:
        return max(1, int(os.getenv("AGENT_NEARBY_MAX_TYPES", "3")))
    except (TypeError, ValueError):
        return 3


def _resolve_type_targets(plan: "RoutePlan | None") -> tuple[list[tuple[str, str]], list[str]]:
    """`plan.shop_types` / `shop_terms_raw` → `([(type, raw), ...], [被上限截掉的...])`（**纯函数**）；按下标对齐（首位缺原词回退 `plan.shop_type_raw`），按 `(type, raw)` 去重保序（同类型下不同细分说法算不同活动）。
    截断到 `AGENT_NEARBY_MAX_TYPES`，被截部分（原词优先、无原词用类型名）作"未覆盖"返回；`shop_types` 空（旧 checkpoint/入口没填）⇒ 返回 `([], [])` 交回单值路径。"""
    if plan is None:
        return [], []
    types = [str(t or "").strip() for t in (plan.shop_types or [])]
    raws = [str(r or "").strip() for r in (plan.shop_terms_raw or [])]
    first_raw = str(getattr(plan, "shop_type_raw", "") or "").strip()
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for i, type_name in enumerate(types):
        if not type_name:
            continue
        raw = raws[i] if i < len(raws) and raws[i] else (first_raw if i == 0 else "")
        key = (type_name, raw)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((type_name, raw))
    cap = _max_nearby_types()
    kept, dropped = pairs[:cap], pairs[cap:]
    return kept, [raw or type_name for type_name, raw in dropped]


async def nearby_worker(state: HmdpState) -> dict:
    """没有具体店铺时，先按类型定位，产出候选 shop_ids 交给后续扇出。
    `specs/0030`：**多活动**（`plan.shop_types` 多项）按类型**逐个取数**再**聚合**（每类一条 `nearby` 输出 ⇒ 多批候选/卡片）；单目标走**同一条**单类型路径，行为与改造前逐字节一致。"""
    record_node("nearby_worker")
    plan = plan_of(state)
    query = state.get("query") or ""
    user_id = state.get("user_id") or ""
    x, y = state.get("x"), state.get("y")

    # 真实类型清单：归类与"按类型找店"都要用 ⇒ 先取一次（原先只在抽出类型后才取）。
    known_types = await asyncio.to_thread(fetch_shop_types, user_id)
    type_pool = (
        [str(t.get("name") or "") for t in known_types if isinstance(t, dict)]
        if isinstance(known_types, list)
        else []
    )

    targets, truncated = _resolve_type_targets(plan)
    if not targets:
        # ---- 旧路径：入口只给了单值 ⇒ 沿用"追问复用 / 字面匹配 / 整句归类"三步 ----
        type_name = (plan.shop_type if plan else "") or ""
        # 追问「还有吗」：**先**复用上一轮品类（2026-10-07）：「还有吗」无品类词 ⇒ `_guess_shop_type` 与整句归类**必然失败**，不复用就只能答「没判断类型」，正确是"把上一轮品类再列一批"。
        # 只在**追问语义**下复用：首问「还有吗」上文无焦点，不该瞎猜品类。
        if not type_name and _is_more_request(query):
            type_name = str(state.get("last_nearby_type") or "").strip()
            if type_name:
                logger.info("附近找店: 追问「更多」复用上一轮品类 type=%s", type_name)
        if not type_name:
            type_name = await asyncio.to_thread(_guess_shop_type, query, user_id)
        if not type_name:
        # 最后一道兜底：把**整句问话**交归类模型，从整句理解"想找哪类店"；入口抽 `shop_type` 可能合法给空（"好吃的"不是真实类型名），而 `_guess_shop_type` 只做字面包含匹配帮不上忙。
        # 只在前面两步落空才走这里 ⇒ 正常路径零额外模型调用；`type_pool` 空时 `_map_category_by_llm` 直接返回空、不调模型，不会在单测偷联网；它救不了"入口模型不可用"（由 `route_degraded` 分支如实说是我们的问题）。
            picked = await _map_category_by_llm(query, type_pool, user_id)
            type_name = picked[0] if picked else ""
        if not type_name:
        # 两种"抽不出类型"必须说成不同的话（否则把服务故障甩锅给用户）：入口**降级**（模型不可用）⇒ 问题在**我们这边**请稍后再试；正常路由但判不出品类 ⇒ 请用户补充。2026-10-05 真机：轻量模型额度耗尽 ⇒ 此前一律答"没判断类型"，用户白试。
            notice = (
                ROUTE_DEGRADED_NOTICE
                if state.get("route_degraded")
                else NEARBY_TYPE_UNKNOWN_NOTICE
            )
            return {"shop_ids": [], "worker_outputs": [{"worker": "nearby", "error": notice}]}
        targets = [(type_name, str((plan.shop_type_raw if plan else "") or "").strip())]

    if len(targets) == 1:
        return await _nearby_for_target(
            state, plan, query, user_id, x, y, type_pool, targets[0][0], targets[0][1]
        )

    # ---- 多活动（`specs/0030`）：逐类取数 + 聚合（顺序遍历，避免并发翻倍；每类内部仍受并发门约束）----
    logger.info(
        "附近找店: 多活动 %s（被上限截断：%s）",
        [f"{t}/{r or '-'}" for t, r in targets], truncated or "无",
    )
    ids: list[int] = []
    seen_ids: set[int] = set()
    focus: list[dict] = []
    seen_focus: set[Any] = set()
    outputs: list[dict] = []
    for type_name, raw in targets:
        part = await _nearby_for_target(
            state, plan, query, user_id, x, y, type_pool, type_name, raw, more_dedupe=False
        )
        for sid in part.get("shop_ids") or []:
            if sid not in seen_ids:
                seen_ids.add(sid)
                ids.append(sid)
        for item in part.get("focus_shops") or []:
            key = item.get("id") if isinstance(item, dict) else None
            if key in seen_focus:
                continue
            seen_focus.add(key)
            focus.append(item)
        outputs.extend(part.get("worker_outputs") or [])
    return {
        "shop_ids": ids,
        "focus_shops": focus,
        # 供下一轮的「还有吗」复用（那句本身不含品类词）：多活动时取**首个**
        "last_nearby_type": targets[0][0],
        "worker_outputs": outputs,
    }


async def _nearby_for_target(
    state: HmdpState, plan: "RoutePlan | None", query: str, user_id: str,
    x: Any, y: Any, type_pool: list[str], type_name: str, raw_type: str,
    *, more_dedupe: bool = True,
) -> dict:
    """**单类型**取数（`specs/0030` 从 `nearby_worker` 抽出，逐行搬移、语义不变）。"""
    # 品类词 → 真实店铺类型：DB 只有 10 个类型（美食/KTV/丽人·美发/健身运动/按摩·足疗/美容SPA/亲子游乐/酒吧/轰趴馆/美睫·美甲），用户口语（火锅/烧烤/咖啡）**没有一个是类型名**；不映射会让"问错词"伪装成"附近没店"，故归类交模型 `_map_category_by_llm`+池内白名单（**不改 `type_name` 本身**，否则错误文案错位）。整句归类已得真实类型时短路。
    # 细分词**补充召回**（`specs/0026` v2）：**先探 ES**（~50ms）命中则短路归类模型（省 ~1-2s）并并入；未命中走原链路。门：开关 ∧ **无店名**(`plan.shop_names` 空) ∧ 细分词；v2 不再要求"原链路召回为空"。
    # **检索词用「用户原话」**（2026-10-12 修正）：入口把「私房菜」归成 `shop_type=美食`，拿归一类型搜博客无意义；原话交 ES 的 `ik_max_word` 命中。门：无店名 ∧ **原话不含真实类型名**。
    _pool = _pool_of(type_pool)
    # 检索词**三级兜底**（`specs/0028` 目标 4）：用户原词 →（空则）非池内类型 →（空则）整句原话。
    #   原词最准（"日料"）：拿整句进 ES 会带进"吃/推荐"等噪声（实测搜出 3 个 id，而 `key=日料` 只命中 1 个）。
    #   `raw_type` 由调用方按**这一个活动**传入（`specs/0030`：多活动时逐类各带自己的原词）。
    if raw_type:
        es_key = raw_type
    elif type_name and type_name not in _pool:
        es_key = type_name
    else:
        es_key = query
    es_shops: list[dict] = []
    if not (plan and plan.shop_names) and not any(t and t in query for t in _pool):
        es_ids = await _blog_recall_ids(es_key, type_pool, user_id)
        if es_ids:
            es_shops = await _blog_recall_details(es_ids, user_id)
    # 命中 ⇒ 跳过归类模型（`mapped=()`：类型搜只用原词，类型/名称链路照跑、结果并入）
    mapped = () if es_shops else await _map_category_by_llm(type_name, type_pool, user_id)
    shops = None
    for term in _type_search_terms(type_name, type_pool, mapped):
        got = await asyncio.to_thread(fetch_shops_nearby, term, x, y, 1, user_id)
        if isinstance(got, list) and got:
            shops = got
            break
        if shops is None:
            # 记住"没查到"这个结论（可能是类型不存在，也可能是该类型无数据，下面要分类）
            shops = got
    fallback_kw = None
    if isinstance(shops, str) or not shops:
        # 按类型搜不到（类型名不存在，如"火锅"；或该类型在后端无数据）时，
        # 退化为按名称关键字搜——实测 /shop/of/name 对"火锅/KTV/餐厅"等都能命中，
        # 而 /shop/of/type 对部分类型（美食/KTV/后端无数据）返回空。该回退不触碰 Java 代码。
        candidates = [type_name or query] + _TYPE_NAME_KEYWORDS.get(type_name, [])
        by_name = None
        for cand in candidates:
            if not cand:
                continue
            res = await asyncio.to_thread(fetch_shops_by_name, cand, 1, user_id)
            if isinstance(res, list) and res:
                by_name = res
                fallback_kw = cand
                break
        if by_name is not None:
            shops = by_name
        elif isinstance(shops, str) and shops not in _nearby_business_empty(type_name):
            # 取数失败（服务不可达 / 非 200）：原样上抛会把 URL、异常类型一起说给用户 ⇒ 原始错误只进日志，对外统一用 `SHOP_LOOKUP_FAILED`；也不能换成"附近没有"（那会把"根本没查成"伪装成"查过了确实没有"）。
            # 日志仍原样回显 `shops`（本节点实际收到的字符串，生产中即 `SERVICE_UNAVAILABLE`），属分类判据的回放证据，刻意不脱敏（非用户可见）。
            logger.error("附近找店失败（类型=%s）：%s", type_name, shops)
            return {
                "shop_ids": [],
                "worker_outputs": [{"worker": "nearby", "error": SHOP_LOOKUP_FAILED}],
            }
        else:
            # **业务为空**：走 `empty` 而非 `error`（`error` 表示"没查成"），两者在 `render_context` 语义不同；判据用与工具层**共享文案模板等值比较**，不用 `startswith` 嗅探（文案改字即静默失效，同 `grounding` 口径）。
            # `specs/0026` **v2**：v1 曾在此处再做补召回（"空才补"）已删（触发前移到 ES 先探）；此处主链路空但 ES 有命中时**不提前返回**，交下面统一合并。
            if not es_shops:
                empty_text = (
                    shops if isinstance(shops, str) else SHOPS_NEARBY_EMPTY.format(type_name=type_name)
                )
                return {
                    "shop_ids": [],
                    "worker_outputs": [
                        {
                            "worker": "nearby",
                            "type_name": type_name,
                            # 键名统一为 items（specs/0009 §2.2）：与 react_loop 产出同一条读取路径
                            "items": [],
                            "empty": empty_text,
                        }
                    ],
                }
            shops = []  # 主链路空 ⇒ 仅用 ES 候选（下面的统一合并会把它追加进来）

    # ES 命中 ⇒ **并入**（去重、保序；不前置/后置）—— 主链路有候选时是并集，主链路空时即 ES 候选
    if es_shops:
        shops = _merge_shop_candidates(shops if isinstance(shops, list) else [], es_shops)

    # 「还有吗」必须**排除前文已展示过的店**（2026-10-06 真机缺陷）：原先每次追问从头查 page=1 原样返回 ⇒ 重复列或（只命中 1 家时）把"已全列完"说成"查不到"；已展示集取自跨轮累积的 `focus_shops`（`grounding`/`nearby_worker`/`summarize_answer` 都写）。
    # 只在 `_is_more_request` 命中时去重：用户认真重问同一句不能被当"要新一批"返回空；`specs/0030`：**多活动**逐类取数时**不做**去重（`more_dedupe=False`）——那句"还有吗"定性的是上一活动。
    more_request = _is_more_request(query) and more_dedupe
    listing_note = ""
    if more_request:
        shown_ids = {
            sid
            for sid in (_shop_id_of(s) for s in (state.get("focus_shops") or []) if isinstance(s, dict))
            if sid is not None
        }
        fresh = [s for s in shops if _shop_id_of(s) not in shown_ids]
        if not fresh:
            # 去重后无新增 ⇒ **确实已列尽**。走 `empty`（＝"查过了，确实没有更多"），
            # 与 `error`（＝"没查成"）严格区分；文案由 `NEARBY_NO_MORE` 给结论，
            # 汇总层照实转达，不必自己推断"资料里到底有没有答案"。
            logger.info("附近找店：追问「更多」但已无新增（%s 类已展示 %s 家）", type_name, len(shown_ids))
            return {
                "shop_ids": [],
                "worker_outputs": [
                    {
                        "worker": "nearby",
                        "type_name": type_name,
                        "items": [],
                        "empty": NEARBY_NO_MORE.format(type_name=type_name),
                    }
                ],
            }
        shops = fresh
        listing_note = "（用户要的是「更多」，以上是**前文尚未列出过**的店铺，不要重复前文那几家）"

    picked = shops[:NEARBY_TOP]
    before_narrow = len(picked)
    picked = await _narrow_by_subcategory(
        picked, type_name, type_pool, mapped, user_id, raw=raw_type
    )
    if before_narrow and not picked:
        # 细分筛选**真的否决**（原词比归类更细且筛空）⇒ 无候选 ⇒ **不发不相关卡片**，
        # 如实说"没有该细分品类"（`specs/0028` 目标 3 / 验收 4）。
        logger.info(
            "附近找店：raw=%s type=%s 细分品类「%s」无匹配 ⇒ 否决全部（不发卡）",
            raw_type or "-", type_name, raw_type,
        )
        return {
            "shop_ids": [],
            "worker_outputs": [
                {
                    "worker": "nearby",
                    "type_name": type_name,
                    "items": [],
                    "empty": SUBCATEGORY_EMPTY.format(term=raw_type),
                }
            ],
        }
    focus = [item for item in (_to_focus_item(s) for s in picked) if item is not None]
    candidate_ids = [_shop_id_of(s) for s in picked if _shop_id_of(s) is not None]

    # 收窄前置（plans/0009.2 N7 / 决策 1-B；多店见 `specs/00xx`）：目标店在发 Send 之前定下来时，`shop_ids` 直接收窄为那几家 ⇒ 第 2 批只给这几家取数，并把 `narrowed_shop_ids` 写给 `route_after_fanout` 让它跳过整轮 ReAct。
    # 完整候选仍留在 `items` 里（`focus_shops` 也照旧全量沉淀），渲染与跨轮焦点不受影响。
    narrowed = _narrow_candidates_by_plan(plan, picked)
    logger.info("附近找店：raw=%s type=%s 命中 %s 家%s", raw_type or "-", type_name, len(picked),
                f"，按 {plan.depends_on or plan.preferences} 收窄到 {narrowed}" if narrowed else "")

    result: dict[str, Any] = {
        "shop_ids": narrowed or candidate_ids,
        "focus_shops": focus,
        # 供下一轮的「还有吗」复用（那句本身不含品类词）
        "last_nearby_type": type_name,
        "worker_outputs": [
            {
                "worker": "nearby",
                "type_name": type_name,
                # 纯候选：只有店主键 + 店维度（`tasks=()`），没有能力块 —— 由 `tasks` 显式声明，
                # 渲染层据此只出一行候选，不再靠"有没有某字段"猜记录类型。
                "items": [
                    _new_shop_record(_shop_id_of(s), s.get("name"), tasks=(), **_shop_dims(s))
                    for s in picked
                    if _shop_id_of(s) is not None
                ],
                # 仅"追问更多"时有值：告诉汇总层这是**接续**，别与前文重复。
                **({"listing_note": listing_note} if listing_note else {}),
            }
        ],
    }
    if narrowed:
        # 多店收窄（`specs/00xx`）：`narrowed_shop_ids` 是**新的权威字段**；
        #   单店时**同时**写老字段 `narrowed_shop_id`，保证既有读侧（`route_after_fanout` /
        #   `react_loop` 守卫二 / 旧 checkpoint）逐字节不变。
        result["narrowed_shop_ids"] = narrowed
        if len(narrowed) == 1:
            result["narrowed_shop_id"] = narrowed[0]
    return result


async def _one_shop_bundle(
    shop_id: int, query: str, user_id: str, with_rag: bool, deadline: float | None = None
) -> dict:
    """单店的详情 + 攻略（+ 可选细节检索），用于详情和对比，产出统一记录（`_new_shop_record`）。
    两条硬变更（specs/0009 §2.2）：① 不再嵌套 `shop` 子字典，详情字段平铺到记录顶层（`avgPrice` / `score` / `distance` / `sold` 正是 `_narrow_by_depends_on` 的收窄维度，旧稿埋进嵌套 `shop` ⇒ 候选取到也 `all None`、永不收窄）；② 只写 `id`（不再写 `shop_id`）。`with_rag=True` 时记录同时带 `detail` 与 `rag` 两个标签（入口同时判出两者时即此形状；`compare` 自 `specs/0011` 起只投影 `detail`）。
    `deadline`（`time.monotonic()` 绝对时刻）是"一个 deadline 贯穿两层"的内层：`detail` / `blogs` 必须做、不受它约束，只有可选的细节检索段吃 `_remaining(deadline, RAG_TIMEOUT)`，超时只写 `chunks_error` ⇒ `rag` 再慢也丢不掉 55ms 就到手的 `detail`（`specs/0010` §3）。"""
    detail, blogs = await asyncio.gather(
        asyncio.to_thread(fetch_shop_detail, shop_id, user_id),
        asyncio.to_thread(fetch_shop_blogs, shop_id, 1, user_id),
        return_exceptions=True,
    )
    if isinstance(detail, BaseException):
        detail = f"店铺 {shop_id} 详情暂时不可用"

    record = _new_shop_record(
        shop_id,
        detail.get("name") if isinstance(detail, dict) else None,
        tasks=("detail", "rag") if with_rag else ("detail",),
        **_shop_dims(detail),
        error="" if isinstance(detail, dict) else str(detail),
    )
    blog_list = [] if isinstance(blogs, BaseException) else blogs
    # 必须归一化：取数失败/无攻略时 fetch_shop_blogs 返回**字符串**
    record["blogs"], record["blogs_error"] = _split_list_or_error(blog_list or [])
    if with_rag and query:
        budget = _remaining(deadline, RAG_TIMEOUT)
        if budget <= 0:
            # 上游（详情那一步）已把预算用光 ⇒ 不发起检索，但**详情照常保留**
            logger.warning(
                "shop_id=%s %s（未获预算）: 只降 chunks，detail/blogs 照常",
                shop_id,
                RAG_TIMEOUT_MSG,
            )
            record["chunks"], record["chunks_error"] = [], f"{RAG_TIMEOUT_MSG}（未获预算）"
        else:
            try:
                chunks = await asyncio.wait_for(
                    asyncio.to_thread(fetch_rag_chunks, shop_id, query, RAG_TOP_K, deadline),
                    timeout=budget,
                )
                record["chunks"], record["chunks_error"] = _split_list_or_error(chunks or [])
            except asyncio.TimeoutError:
                logger.warning(
                    "shop_id=%s %s（%.1fs 预算）: 只降 chunks，detail/blogs 照常",
                    shop_id,
                    RAG_TIMEOUT_MSG,
                    budget,
                )
                record["chunks"], record["chunks_error"] = [], f"{RAG_TIMEOUT_MSG}（{budget:.1f}s）"
            except Exception as exc:  # noqa: BLE001  细节检索失败不得影响详情
                logger.warning("shop_id=%s 细节检索失败（只降该块）: %s", shop_id, exc)
                record["chunks"], record["chunks_error"] = [], f"细节检索失败：{exc}"
    return record


async def _shop_record_for(
    sid: int, tasks: "tuple[str, ...]", query: str, user_id: str
) -> tuple[dict, dict[str, bool]]:
    """该店的一条记录 + 每个任务的成败（`plans/0009.2` N2/N3）。
    一店一个 Send 之后，同一家店的多份数据必须合并进同一条记录（多 `tasks` 标签），渲染层才能天然"一店一块"（归并是产出端做的，不是渲染层）。
    降级是块级的（决策 3-A）：某任务熔断/超时/取数失败只让它自己那块写 `*_error`，其余块照常（整条降级会把"券查不到"扩大成"这家店查不到"）；取数次数与逐任务分开取完全一致（详情+攻略一次、细节一次、券一次），不因合并多调接口。"""
    fields: dict[str, Any] = {}
    ok_map: dict[str, bool] = {t: True for t in tasks}

    def _blocked(task: str) -> bool:
        """熔断短路：**不发起任何实际调用**（验收 4 / R5）。"""
        if not _breaker_open(task):
            return False
        logger.warning(
            "worker(%s) 已被熔断短路（连续失败达 %s），未发起实际调用", task, FANOUT_FAIL_THRESHOLD
        )
        return True

    active = frozenset(t for t in tasks if not _blocked(t))
    if not active:
        return _blocked_record(sid, tasks), {t: False for t in tasks}
    for task in tasks:
        if task not in active:
            fields[_DEGRADED_ERROR_FIELD[task]] = _degraded_block_error(task)
            ok_map[task] = False

    name: str | None = None
    want_detail, want_rag = "detail" in active, "rag" in active

    if want_detail and want_rag:
        # 详情与细节一起要 ⇒ 复用 `_one_shop_bundle` 的内部并发（detail + blogs + chunks 各一次），而不是"两个任务各取一遍详情"（plans/0009.2 N4）。
        # 自 `specs/0011` 起 `compare` 只投影 `detail` ⇒ 走不到这里；能进这一支的是入口同时判出 `detail` 与 `rag`。
        # 一个 deadline 贯穿两层（`specs/0010` §3）：外层 `_run_with_timeout` 与内层"细节检索段"用同一个绝对截止时刻（只给内层固定预算不够：`d=8s, r=6s, 上限 12s` 时内层不截断、外层照样把整包含 55ms 的 `detail` 丢掉）。
        deadline = _deadline_after(WORKER_TIMEOUT)
        bundle, ok = await _run_with_timeout(
            _one_shop_bundle(sid, query, user_id, with_rag=True, deadline=deadline),
            "detail",
            deadline,
        )
        if ok and isinstance(bundle, dict):
            name = bundle.get("name")
            fields.update(_detail_fields(bundle, with_rag=True))
            ok_map["detail"] = not bool(bundle.get("error"))
            ok_map["rag"] = not bool(bundle.get("chunks_error"))
        else:
            fields["blogs_error"] = fields["chunks_error"] = "取数失败"
            ok_map["detail"] = ok_map["rag"] = False
    elif want_detail:
        bundle, ok = await _run_with_timeout(
            _one_shop_bundle(sid, "", user_id, with_rag=False), "detail"
        )
        if ok and isinstance(bundle, dict):
            name = bundle.get("name")
            fields.update(_detail_fields(bundle, with_rag=False))
            ok_map["detail"] = not bool(bundle.get("error"))
        else:
            fields["blogs_error"] = "取数失败"
            ok_map["detail"] = False
    elif want_rag:
        res, ok = await _run_with_timeout(
            asyncio.to_thread(fetch_rag_chunks, sid, query, RAG_TOP_K), "rag"
        )
        # 必须归一化：检索不到时 fetch_rag_chunks 返回**字符串**（会被逐字迭代）
        chunks, chunks_error = _split_list_or_error(res or [])
        fields["chunks"] = chunks
        fields["chunks_error"] = chunks_error or ("" if ok else "取数失败")
        ok_map["rag"] = ok

    if "voucher" in active:
        res, ok = await _run_with_timeout(
            asyncio.to_thread(fetch_shop_vouchers, sid, user_id), "voucher"
        )
        # 同 rag：取数失败 / 该店无券时 fetch_shop_vouchers 返回**字符串**
        vouchers, vouchers_error = _split_list_or_error(res or [])
        fields["vouchers"] = vouchers
        fields["vouchers_error"] = vouchers_error or ("" if ok else "取数失败")
        # 0006.1：每个券附一张下单卡片（纯函数、零出站；秒杀券 action=goto_activity）。
        # 取数失败时 vouchers 为空 ⇒ 不下发任何卡片。
        fields["order_cards"] = [prepare_order_card(v, sid) for v in vouchers] if vouchers else []
        ok_map["voucher"] = ok

    return _new_shop_record(sid, name, tasks=tasks, **fields), ok_map


async def shop_worker(state: HmdpState) -> dict:
    """扇出 worker：一次 Send = 一家店，把该店的全部意图合并成一条记录。
    粒度在 `plans/0009.2`（阶段二）里从"任务"改成"店铺"：`worker_tasks` 带该店的全部意图（≥1），`worker_shop_ids` 通常恰有 1 个店；同一节点多实例并行，全部完成后汇总节点只执行一次。
    产出是「一店一条」记录（可带多个 `tasks` 标签）⇒ 渲染层天然"一店一块"、不需要归并（归并靠产出端的唯一工厂 `_new_shop_record`）；旧载荷（单值 `task: str`）仍读得懂（`_worker_tasks_of` 双读），`task` 的类型不被改动，checkpoint 里已落库的会话不会错乱。"""
    tasks = _worker_tasks_of(state)
    shop_ids = [int(i) for i in (state.get("worker_shop_ids") or []) if i is not None]
    query = state.get("worker_query") or state.get("query") or ""
    user_id = state.get("user_id") or ""

    if not shop_ids:
        return {}
    if not tasks:
        logger.warning(
            "未知 worker 任务: %s（合法=%s）",
            state.get("worker_tasks") or state.get("task"), sorted(SHOP_RECORD_TASKS),
        )
        return {}

    if all(_breaker_open(t) for t in tasks):
        # 全部任务熔断 ⇒ 一条"只有降级说明"的记录，**零实际调用**（R5 的看门路径）。
        # 放在 `record_node` **之前**（与旧实现一致）—— 短路时不算"跑过这个 worker"。
        return _breaker_degraded_output(tasks, shop_ids)

    # 记账节点名**原样保留** `shop_worker:{task}`（按 task 逐个记，metrics 口径不漂移）
    for task in tasks:
        record_node(f"shop_worker:{task}")

    records: list[dict] = []
    ok_map: dict[str, bool] = {t: True for t in tasks}
    error_text = ""
    try:
        for sid in shop_ids:
            record, per_task = await _shop_record_for(sid, tasks, query, user_id)
            records.append(record)
            for task, ok in per_task.items():
                ok_map[task] = ok_map[task] and ok
    except Exception as exc:  # 单个 worker 失败不能拖垮整条链路
        logger.error("shop_worker(%s) 执行失败: %s", "+".join(tasks), exc)
        error_text = str(exc)
        for task in tasks:
            ok_map[task] = False

    # 熔断计数：成功清零 / 失败累加。超时也算失败，但**不让本轮失败**
    # （决策 3-A：只坏这一块，其余块照常进 summarize）
    for task in tasks:
        _record_worker_result(task, ok_map[task])

    output: dict[str, Any] = {
        "worker": "+".join(tasks),
        "shop_ids": shop_ids,
        "query": query,
        "items": records,
    }
    if error_text:
        output["error"] = error_text
    if not all(ok_map.values()):
        output["failed"] = True
    return {"worker_outputs": [output]}


# ===========================================================================
# 第三层：路由——条件边做主骨架，Send 做局部优化
# ===========================================================================
def _fanout_args(state: HmdpState, **extra: Any) -> dict[str, Any]:
    """构造 Send 的 payload。
    LangGraph 的 Send 只把 arg 作为 worker 的输入 State，父 State 的字段不会透传，所以 worker 需要的一切（用户、坐标、原始问题、路由结果）都必须显式带上。"""
    args: dict[str, Any] = {
        "query": state.get("query") or "",
        "user_id": state.get("user_id") or "",
        "x": state.get("x"),
        "y": state.get("y"),
        "plan": state.get("plan"),
        # `route_degraded` 必须带上（2026-10-05 造故障真机实测漏过）：`nearby_worker` 靠它把"判不出类型"分流成「服务忙，请稍后再试」而非「你没说明白」。**Send 不透传父 State**，漏带 ⇒ 标记恒 None ⇒ 降级仍答旧文案。
        "route_degraded": state.get("route_degraded", False),
        # `focus_shops` / `last_nearby_type` 也必须带上（2026-10-07 进程内重放实测漏过）：`nearby_worker` 靠 `focus_shops` 剔"已展示过的店"、靠 `last_nearby_type` 在追问「还有吗」复用上一轮品类；漏带 ⇒ 两者恒空 ⇒ 追问被"已展示"判据架空或答「没能判断类型」。
        # 单测没抓到是因为它们直接调 `nearby_worker` 并手工拼完整 state、绕过 Send ⇒ 纯函数绿 ≠ 节点真拿得到（须配源码级接线断言）。
        "focus_shops": list(state.get("focus_shops") or []),
        "last_nearby_type": state.get("last_nearby_type") or "",
    }
    args.update(extra)
    return args


#: 引导意图：手上还没有店时可以先跑的意图（它自己会产出真实店 id），与「消费店 id 的意图」（`CONSUMES_SHOP_IDS`，必须先有店）相对。
#: 集合里只有 `nearby`（`compare` 已退场：产出与 `detail` 完全同形，保留只会白多一轮"每店详情+攻略"；而"点名多家"的店来自 `grounding`，不需要引导）；该类意图的记录由消费方按 `items` 统一读取，不另立键名。
BOOTSTRAP_INTENTS: tuple[str, ...] = ("nearby",)

#: 需要**已有** shop_id 才能执行的意图。它们必须等第一批跑完、拿到真实店 id 之后才能发。
#: 直接由 `SHOP_RECORD_TASKS` 派生 —— "记录的能力标签域"与"消费店 id 的意图"必须同集合，
#: 一处定义、两处引用，从结构上杜绝二者漂移（改一处会同时改到）。
CONSUMES_SHOP_IDS: frozenset[str] = SHOP_RECORD_TASKS


def plan_fanout(plan: "RoutePlan | None", shop_ids: list[int]) -> list[list[tuple[str, list[int]]]]:
    """把「子意图」按依赖关系拓扑分批，返回 `[[(task, worker_shop_ids), ...], ...]`；纯函数（不读 State、不碰网络），可直接单测（见 `app/tests/test_plan_fanout.py`）。
    规则：引导意图（`BOOTSTRAP_INTENTS`：`nearby`）进第 1 批；消费店 id 的意图（`detail` / `voucher` / `rag`）需真实店 id 进第 2 批；多店 + 目标依赖结果 ⇒ 不预取 `voucher`（B4，`plans/0011`），`voucher` 的 `worker_shop_ids` 长度恒 ∈ {0,1}，只有"多店且 `target_depends_on_result=False`"才 N 家各取一次（每家都是目标）。
    分批语义：`shop_ids` 为空只发 `nearby` 批，没有可引导意图（`compare` / 只有消费类）则返回 `[]`、由调用方 `clarify`（或退回 `nearby`）；非空单店 `voucher` 直接发该批（跳过第 1 批），单店 `nearby` 等价于 `detail`；`compare` 展开为每店 `detail`（`specs/0011` 起只投影 `detail`、不再连带 `rag` —— 可比维度只在 `detail` 里，`rag` 只由入口模型的 `rag` 意图决定），已删除"多店自动补 `compare`"。"""
    intents = set(plan.sub_intents) if plan else set()
    ids = [int(i) for i in (shop_ids or []) if i is not None]

    producing = [i for i in BOOTSTRAP_INTENTS if i in intents]
    consuming = [i for i in ("detail", "voucher", "rag") if i in intents]

    # ---- 情形 A：手上还没有店 → 第 1 批只有引导意图，消费类必须等结果 ----
    if not ids:
        if not producing:
            return []
        return [[(task, []) for task in producing]]

    # ---- 情形 B：店已确定 → 一批发完，跳过拓扑等待 ----
    batch: list[tuple[str, list[int]]] = []
    multi = len(ids) > 1

    effective = set(consuming)
    # B1 `compare` 意图**展开**为每店的 detail（不再是一个独立的 worker 任务）。
    # `specs/0011`：**只补 detail、不连带 rag** —— 对比需要的是 detail 里的可比维度；
    #    `rag` 只由入口模型的 `rag` 意图决定（在此补 rag = 替用户做决定，且让慢路径变高频）。
    if "compare" in intents:
        effective.add("detail")
    # B2 单店的 nearby 意图等价于"看这家的详情"
    if not multi and "nearby" in intents:
        effective.add("detail")
    # ---- B4（`plans/0011`）：目标店未知时，券不预取 ----
    # 多店 + 目标依赖中间结果 ⇒ "最便宜那家"此刻还不知道是哪家（要等 detail 到手），预取券等于对候选全集各取一次、其中最多 1 家有语义；真正的代价是语义而非性能（`voucher/list` 只要 58ms，但多余店家的券会挤进汇总上下文、诱导模型都报一遍）。
    # 券不是必需信息（用户真想要，下一轮直接问那家店即可）；此处刻意不加任何兜底，模型自己调 `shop_vouchers` 属其自由裁量、不算兜底（见 `plans/0011` §1）。
    if multi and plan is not None and plan.target_depends_on_result:
        effective.discard("voucher")
    # B3 兜底：一个取数意图都没有时，至少取详情（对齐旧实现）。
    # 必须排在 B4 **之后**：`{voucher}` + 多店 + 目标依赖结果 这条路上 B4 会把唯一意图去掉，
    #    而"要确定是哪家"本身**就需要 detail** ⇒ 让它落到 B3 补 detail 是正确行为，不是副作用。
    if not effective:
        effective.add("detail")

    for task in ("detail", "voucher", "rag"):
        if task in effective:
            batch.append((task, list(ids)))

    return [batch] if batch else []


def group_by_shop(
    batch: list[tuple[str, list[int]]],
) -> "list[tuple[int | None, tuple[str, ...]]]":
    """把批内「每任务一个店铺列表」翻转为「每店一个任务元组」（`plans/0009.2` N5）。
    同一家店出现在多个任务里 ⇒ 合并成一项（`tasks` = 该店全部意图）⇒ 一个 Send；无店任务（引导意图 `nearby`）单独成一项（`id=None`）；顺序为无店项在前、其余按店 id 升序 ⇒ 输出确定、可复现、A/B 可对照（验收 17）。
    做成纯函数是因为这是本阶段唯一"把粒度从任务改成店"的地方，纯函数才能离线断言"批内每个 `(task, ids)` 恰好出现一次"（R2：翻转时静默丢任务）。"""
    per_shop: dict[int, list[str]] = {}
    shopless: list[str] = []
    for task, ids in batch or []:
        targets = [int(i) for i in (ids or []) if i is not None]
        if not targets:
            shopless.append(task)
            continue
        for sid in targets:
            per_shop.setdefault(sid, []).append(task)

    # 无店任务**不能**过 `_ordered_tasks`（它只认 worker 任务名，会把 `nearby` 整个滤掉）
    groups: list[tuple[int | None, tuple[str, ...]]] = [
        (None, (task,)) for task in dict.fromkeys(shopless)
    ]
    for sid in sorted(per_shop):
        groups.append((sid, _ordered_tasks(per_shop[sid])))
    return groups


def dispatch_workers(state: HmdpState) -> list[Send] | str:
    """扇出一个批次的取数任务：一次 Send = 一家店（该店的全部意图随 `worker_tasks` 一起带上）。
    只发 `plan_fanout(plan, shop_ids)` 的第一个批次：`shop_ids` 为空时该批只含引导意图（`BOOTSTRAP_INTENTS`：`nearby`），消费类（`voucher` / `detail` / `rag`）故意不发、要等引导批的真实结果；店已确定时引导批为空，直接发消费批。
    粒度（`plans/0009.2`）是先按店翻转（`group_by_shop`）再发：阶段一"一任务一 Send"会让同店详情与券落在两条记录、渲染成两个块，现在合并成一条记录 ⇒ 天然"一店一块"；这是形状/归属的收益，Send 数不会因此变少（任务数 ≤4 → 店数 ≤3）。"""
    plan = plan_of(state)
    shop_ids = [int(i) for i in (state.get("shop_ids") or []) if i is not None][:MAX_SHOPS]

    batches = plan_fanout(plan, shop_ids)
    if not batches:
        return SUMMARY_NODE

    # 批内**按店翻转**：`[(task, ids)]` → `[(sid, (tasks…))]`
    groups = group_by_shop(batches[0])
    query = (plan.rewritten_query if plan and plan.rewritten_query else state.get("query") or "")

    # 截断（验收 2）：口径 = **店铺数**（翻转之后一项 = 一个 Send），且**不得静默丢**
    kept, truncated = truncate_fanout(groups)
    if truncated:
        logger.warning(
            "扇出超过上限 %s，截断 %s 家店（原 %s 家）", MAX_FANOUT, truncated, len(groups)
        )

    sends: list[Send] = []
    for sid, tasks in kept:
        if sid is None:
            # 无店任务（引导意图）的目标是 `nearby_worker`、不是 `shop_worker`（后者拿不到店 id 时直接返回空，发出去等于空转）。
            # 当前两个调用方都只在"店已确定"时进来（`if state.get("shop_ids")`），这条分支是把情形 A 接回可用引导节点的接线。
            sends.append(Send("nearby_worker", _fanout_args(state, worker_query=query)))
        else:
            sends.append(Send("shop_worker", _fanout_args(
                state,
                worker_tasks=list(tasks),
                worker_shop_ids=[sid],
                worker_query=query,
            )))

    if not sends:
        return SUMMARY_NODE
    logger.info(
        "扇出 %s 个 worker(第1批，按店): %s", len(sends), [t.arg.get("worker_shop_ids") for t in sends]
    )
    return sends


def truncate_fanout(
    groups: "list[tuple[int | None, tuple[str, ...]]]", limit: int | None = None
) -> "tuple[list[tuple[int | None, tuple[str, ...]]], int]":
    """对按店翻转后的扇出计划做上限截断，上限 `limit`（默认 `MAX_FANOUT`）作用在店铺数上。
    `plans/0009.2` 把 Send 粒度从"任务"改成"店铺"后每项恰好一个 Send，项数即店铺数（+ 先定位时的那个引导任务）；沿用旧口径（同一家店在每个任务里各数一次）会把上限放大成几倍 ⇒ 该截的没截、静默多发把 DB/ES 打满。
    返回 `(保留项, 被截断的数量)`：未超限原样返回 `truncated=0`（正常请求不触发）；超限按顺序保留到上限（顺序由 `group_by_shop` 定为按店 id 升序）。不得静默丢，调用方必须把 `truncated` 记进日志与返回结构。"""
    cap = MAX_FANOUT if limit is None else limit
    if len(groups) <= cap:
        return list(groups), 0
    return list(groups[:cap]), len(groups) - cap


def _breaker_degraded_output(tasks: "tuple[str, ...]", shop_ids: list[int]) -> dict:
    """熔断短路时的降级返回：不发起任何实际调用，每条记录只写各块的降级说明。
    返回结构与正常 worker 输出同形（有 `items` / `error` / `degraded`），因此 `summarize` 与 `render_context` 不需要为降级路径写特例。
    记录由 `_blocked_record` 产出，与"部分任务熔断"时逐块降级的形状完全一致，渲染层看不出两条路径的差别。"""
    joined = "+".join(tasks)
    logger.warning(
        "shop_worker(%s) 已被熔断短路（连续失败达 %s），未发起实际调用", joined, FANOUT_FAIL_THRESHOLD
    )
    return {
        "worker_outputs": [
            {
                "worker": joined,
                "shop_ids": list(shop_ids),
                "items": [_blocked_record(sid, tasks) for sid in shop_ids],
                "degraded": True,
                "error": _degraded_block_error(tasks[0]) if tasks else "",
            }
        ]
    }


#: 降级文案用的中文标签（给人看的中文，缺失时回退到 task 原名），**定义在 `degrade.py`**（单点真源），此处只剩引用（见文件顶部 `_TASK_LABEL`）。
#: 只登记可执行的取数任务（外加引导意图 `nearby`）：`compare` 自 `plans/0009.2` 起已不是 worker 任务（意图保留、展开为每店 `detail`；`specs/0011` 收窄为只投影 `detail`），放进此表会让人误以为还有 compare 任务在跑。


# ---- ReAct 补跳：接在固定编排之后，只读、有界、可单测（specs/0005 §5.5）----
#: 补跳允许调用的只读工具白名单：从 `hmdp_tools.REACT_TOOLS` 派生（不再手工维护），使「提示词清单 / 白名单 / 绑定列表 / 日志口径」四者同一，消灭"手写清单"这个漂移源。
#: 这里绝不允许出现任何写操作（下单 / 支付 / 取消 / 退款）—— 写操作不进 `react_loop` 是规格 §3 non-goal 8 的硬边界，见 `test_react_loop_constraints`。
REACT_READONLY_TOOLS: frozenset[str] = frozenset(t.name for t in REACT_TOOLS)

#: `finish` 已退场：收敛改由「模型一个工具都不调」表达（`_react_decide` 返回 `[]`）。
#: 旧 `finish` 是伪工具（只为给模型一个"说完了"的出口），多一个反而逼它在"调 finish"与"什么都不调"之间二选一、成为空转的温床；`_react_exec_tool` 里的 `finish` 分支也已删除，收敛只走"空动作列表"。

#: `depends_on` 四个取值对应的字段与方向（只丢裸 token `price` 给模型，它只能猜"评分最高"还是"数字最大"）。
#: 方向必须与 `_narrow_by_depends_on` 的 `key_map` 保持一致，否则提示词说的"最便宜"与规则实际选中的店会对不上。
_DEPENDS_ON_HINTS: dict[str, str] = {
    "price": "人均价格最低（字段 avgPrice，越小越优）",
    "score": "评分最高（字段 score，越大越优）",
    "distance": "距离最近（字段 distance，越小越优）",
    "sold": "销量最高（字段 sold，越大越优）",
    # ---- 反向 ----
    "price_desc": "人均价格最高（最贵；字段 avgPrice，越大越优）",
    "score_asc": "评分最低（字段 score，越小越优）",
    "distance_desc": "距离最远（字段 distance，越大越优）",
    "sold_asc": "销量最低（字段 sold，越小越优）",
}


class ReactAction(BaseModel):
    """补跳的下一步动作。由便宜模型产出，规则侧只负责执行与限步。
    模型只回答"下一步调哪个工具、参数是什么"（语义），不回答"要不要进这个环"—— 那由 `entry_router` 的 `target_depends_on_result` 决定。"""

    # `finish` 已退场（2026-09-28 迁移到 `bind_tools`）：收敛改由"工具通道里一个工具都不调"
    #    (`tool_calls == []`) 表达，不再有 `finish` 这个伪工具。本结构只承载"已解析出的动作"，
    #    `tool` 由运行时从 `REACT_TOOLS` 白名单校验后得到，不来自模型手填的字段。
    tool: str = Field(default="", description="要调用的只读工具名（由工具通道解析，白名单见 REACT_READONLY_TOOLS；模型一个都不调即表示收敛）")
    args: dict[str, Any] = Field(default_factory=dict, description="工具参数")


def _grounding_notice_of(state: HmdpState) -> dict | None:
    """取出 `grounding` 挂上的"点名店铺没解析出来"说明（没有则 `None`）。"""
    for item in state.get("worker_outputs") or []:
        if item.get("worker") == "grounding" and item.get("notice"):
            return item
    return None


def _route_unresolved_named_shop(state: HmdpState, plan: RoutePlan) -> list[Send] | str:
    """用户点名了店、但 `grounding` 没解析出 id 时的去向（`specs/0007` §8.1 选项 B）。
    - 有坐标 → 退回 `nearby_worker` 找同类候选（一次往返给到用户）；否则 → `clarify`，由它用 `grounding` 的说明作答（"没找到「海底捞」这家店"）。
    服务报错时不去附近兜底（店铺服务都不可用，附近查询必然也失败，只会把"暂时查不到"冲淡成无用的"附近没找到"）；类型判定留给 `nearby_worker`（本函数是同步路由，在这里发网络请求会阻塞事件循环，服务不可达要卡满 5s 超时）。"""
    notice = _grounding_notice_of(state) or {}
    if (
        notice.get("outcome") != _LOOKUP_ERROR
        and state.get("x") is not None
        and state.get("y") is not None
    ):
        query = plan.rewritten_query or state.get("query") or ""
        return [Send("nearby_worker", _fanout_args(state, worker_query=query))]
    return "clarify"


def route_after_grounding(state: HmdpState) -> list[Send] | str:
    """Grounding 之后的去向：缺实体就先定位，否则直接扇出。"""
    # **必须第一句**：预约轮（`booking_pending`）要先落地店，才能谈别的。
    #    放后面会被 `if state.get("shop_ids")` 抢走 ⇒ 预约意图被当成"找店"去扇出取数，
    #    用户等来的是一堆卡片而不是预约确认卡。
    if state.get("booking_pending"):
        logger.info("Grounding 后转预约确认（本轮解析到 shop_ids=%s）", state.get("shop_ids"))
        return "booking_confirm"
    plan = plan_of(state)
    if state.get("shop_ids"):
        return dispatch_workers(state)
    if plan and plan.need_clarify:
        return "clarify"
    if plan and (plan.shop_type or "nearby" in plan.sub_intents):
        query = plan.rewritten_query or state.get("query") or ""
        return [Send("nearby_worker", _fanout_args(state, worker_query=query))]
    # 用户点名了店却没解析出来：必须单独接出来，且放在上述既有判据之后，否则会落到下面的 catch-all ⇒ `clarify` 反问"你想了解哪家店铺呢"，与用户输入直接矛盾（见 bug/BUG-20260928-grounding-named-shop-to-clarify.md）。
    if plan and plan.shop_names:
        return _route_unresolved_named_shop(state, plan)
    return "clarify"


def route_after_nearby(state: HmdpState) -> list[Send] | str:
    """定位之后：只有用户还问了详情/优惠/细节/对比，才继续扇出（只问"附近有什么火锅店"时候选列表已够，省掉一轮取数）。"""
    if not state.get("shop_ids"):
        return SUMMARY_NODE
    plan = plan_of(state)
    intents = set(plan.sub_intents) if plan else set()
    if not intents or intents <= {"nearby"}:
        return SUMMARY_NODE
    return dispatch_workers(state)


# ---------------------------------------------------------------------------
# ReAct 补跳（specs/0005 §5.5.4）——接在固定编排**之后**
# ---------------------------------------------------------------------------
def _narrowed_ids(state: Any) -> list[int]:
    """已收窄到的店 id 列表（`specs/00xx`）：优先新字段 `narrowed_shop_ids`，为空时回退旧 `narrowed_shop_id`。
    **必须两字段都认**：旧 checkpoint 里只有单值字段，只看新字段会把"已收窄"误判成"未收窄"
    ⇒ 白进一轮 ReAct（真机会多烧 ≤6 次模型调用）。"""
    ids = [i for i in (state.get("narrowed_shop_ids") or []) if isinstance(i, int)]
    if ids:
        return ids
    single = state.get("narrowed_shop_id")
    return [single] if isinstance(single, int) else []


def route_after_fanout(state: HmdpState) -> str:
    """固定编排（第 1 批）之后的去向：目标依赖中间结果（`plan.target_depends_on_result`）就去补跳，否则直接汇总。
    判据来自入口模型的语义，这里只做结构判断（布尔 → 走哪条边）；这是"要不要进环"的唯一入口，`react_loop` 内还有一道守卫双保险。
    收窄前置（`plans/0009.2` N7 / 决策 1-B）：已收窄时 ReAct 的唯一入口理由（"还不知道是哪家"）消失 ⇒ 直接汇总，省掉整轮（≤`REACT_MAX_STEPS`）模型调用与重复取数；收窄失败仍照旧进环兜底（R6）。"""
    plan = plan_of(state)
    if plan and plan.target_depends_on_result and not _narrowed_ids(state):
        return "react_loop"
    return SUMMARY_NODE


def _is_blank(value: Any) -> bool:
    """空值判定（`None` / `""` / `[]` / `{}`）—— 用于"只让非空覆盖空"的归并规则。"""
    return value is None or value == "" or value == [] or value == {}


def _react_candidates(outputs: list[dict]) -> list[dict]:
    """从已有 worker 结果里取出"待收敛的候选"（统一记录形状，只有一条读取路径）。
    过滤条件从"产出者是不是引导意图"改为「记录带 `id`」（`plans/0009.2` N8 / R1）：`compare` 退场后"点名多家"的生产者变成 `shop_worker`（detail/rag），旧判据会让候选恒空（`narrowed=[]` ⇒ 补跳空转）。
    同一家店的多条记录按 `id` 归并，只让非空字段覆盖空字段（整字典 `update` 会把候选的 `distance` 刷成 `None`）；取值一律经 `_split_list_or_error`（str 会被逐字迭代，在"候选来源"这一步静默污染）。"""
    merged: dict[int, dict] = {}
    for item in outputs or []:
        if not isinstance(item, dict):
            continue
        records, _err = _split_list_or_error(item.get("items") or [])
        for rec in records:
            sid = _shop_id_of(rec)
            if not isinstance(rec, dict) or sid is None:
                continue
            cand = dict(rec)
            # 顺手把主键**归一化到 `id`**：这样"候选一定有 `id`"成为下游可依赖的不变式，
            # 而不是"大多数时候有"（旧 checkpoint 里的记录只带 `shop_id`）。
            cand["id"] = sid
            prev = merged.get(sid)
            if prev is None:
                merged[sid] = cand
                continue
            for key, value in cand.items():
                if _is_blank(prev.get(key)) and not _is_blank(value):
                    prev[key] = value
            # `tasks` 取**并集**：同一家店"既有详情又有券"时，候选也要如实反映
            prev["tasks"] = list(
                _ordered_tasks(
                    {str(t) for t in (prev.get("tasks") or [])}
                    | {str(t) for t in (cand.get("tasks") or [])}
                )
            )
    return list(merged.values())


#: ReAct 决策 prompt 里**允许出现**的店维度键（`specs/0016` T1 / §8 Q1）：
#: 收窄维度必须在顶层（`specs/0009` 硬约束），故这里取顶层字段即可，不动记录形状。
_REACT_PROMPT_DIM_KEYS: tuple[str, ...] = ("score", "avgPrice", "distance", "sold")


def _react_prompt_candidates(records: list[dict]) -> list[dict]:
    """把候选记录**投影成精简行**喂给 ReAct 决策模型（`specs/0016` §10.1(b)）；`_react_candidates` 完整复制重字段（`vouchers`/`chunks`/`blogs`/`order_cards`），而 `_react_decide` 每步把候选塞进 prompt ⇒ 载荷随步数线性放大，故只输出 `{id, name, score, avgPrice, distance, sold, fetched}`。
    `fetched` = 该店**已取到**的任务标签（`detail`/`voucher`/`rag`，固定优先级序），模型只需知"券取了没"不须知券面（那是汇总模型的事）；空值不占位（`_is_blank` 字段省略）行更短；纯函数：同输入同输出，不读全局不改入参。"""
    out: list[dict] = []
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        sid = _shop_id_of(rec)
        if sid is None:
            continue
        row: dict[str, Any] = {"id": sid}
        name = str(rec.get("name") or "").strip()
        if name:
            row["name"] = name
        row["fetched"] = [t for t in _ordered_tasks(rec.get("tasks") or ()) if t in SHOP_RECORD_TASKS]
        for key in _REACT_PROMPT_DIM_KEYS:
            value = rec.get(key)
            if not _is_blank(value):
                row[key] = value
        out.append(row)
    return out


#: 维度 → (候选记录里的字段名, 是否"越大越优")。方向必须与 `_DEPENDS_ON_HINTS` 的文字描述一致。
#: 🔴 **维度自带方向**：`price_desc` / `score_asc` / `distance_desc` / `sold_asc` 是**反向**取值
#: （最贵 / 评分最低 / 最远 / 卖得最少），与基维度共用同一字段、翻转 `reverse` 即可 —— 这样
#: "最贵"不会再被收窄成最便宜，也不必新引入 `sort_direction` 全局方向字段（多维度复合偏好时
#: 单一方向表达不了，见 `specs/00xx` 问题 B）。
_DEPENDS_ON_FIELDS: dict[str, tuple[str, bool]] = {
    "price": ("avgPrice", False),
    "score": ("score", True),
    "distance": ("distance", False),
    "sold": ("sold", True),
    # ---- 反向（用户要的是另一头）----
    "price_desc": ("avgPrice", True),        # 最贵
    "score_asc": ("score", False),           # 评分最低
    "distance_desc": ("distance", True),     # 最远
    "sold_asc": ("sold", False),             # 卖得最少
}

#: 反向维度 → 基维度（反向只是基维度翻转方向，排序字段不变）。
#: 用于"同一维度出现正反两种说法 ⇒ 矛盾，视为无效"以及提示词人话渲染。
_REVERSE_DIMENSIONS: dict[str, str] = {
    "price_desc": "price",
    "score_asc": "score",
    "distance_desc": "distance",
    "sold_asc": "sold",
}


def _plan_ranking_dimensions(plan: "RoutePlan | None") -> list[str]:
    """本次收窄/排序用的**有序维度**（`specs/0024` / `plans/0043`）：优先软偏好 `preferences`，为空回退旧单值 `depends_on`。
    收窄前置 / `react_loop` / ReAct 提示词三处**共用这一份判据**，否则出现"规则按偏好收窄、提示词却说未收窄"的静默错位（`test_react_loop_constraints` 看住）。"""
    if plan is None:
        return []
    dims = _preference_dimensions(plan.preferences)
    if dims:
        return dims
    return _preference_dimensions([plan.depends_on])


def plan_top_k(plan: "RoutePlan | None") -> int:
    """本次收窄**要几家**（`specs/00xx` 问题 A）。判据**三级**，越靠前越优先：

    1. `top_n` —— 用户**明说的家数**（「评分最高的两家」⇒ 2）。与指代无关，
       「附近评分最高的两家」这类**没有指代**的问句正是靠它拿到 `top_k=2`（`plans/0050` 缺口 b）。
    2. `len(target_indices)` —— **指代本身带出的家数**（「最后两家」⇒ `[-2,-1]` ⇒ 2）。
       用于入口把家数写进了下标列表、没写 `top_n` 的旧行为 / 旧 checkpoint。
    3. 都为空 ⇒ 回退 `1`，**行为与改造前逐字节一致**（"入口抽取为主链、正则/单店为兜底"的落点：
       抽不出来就按老路子走）。

    三者都截断到 `MAX_TARGET_INDICES`，防"前 100 家"拉爆查询。

    ⚠️ **为什么 `top_n` 优先于 `target_indices` 长度**：「这三家里评分最高的两家」是
    指代（范围 3 家）与家数（取 2 家）**同时出现**的形态，此时列表长度 3 表达的是"范围"
    而不是"要几家"—— 拿长度当 `top_k` 会收窄到 3 家，与用户要的 2 家不符。
    """
    if plan is None:
        return 1
    n = int(getattr(plan, "top_n", 0) or 0)
    if n > 0:
        return min(n, MAX_TARGET_INDICES)
    n = len([i for i in (plan.target_indices or []) if isinstance(i, int)])
    if n <= 0:
        return 1
    return min(n, MAX_TARGET_INDICES)


def _sorted_by_dimensions(candidates: list[dict], dimensions: Any) -> list[dict] | None:
    """按**有序维度列表**做**字典序**排序（缺字段排最后）；不可算时返回 `None`（不臆造排序），是收窄 `_narrow_by_dimensions` 与多方案 `_build_schemes` **共用**口径（`plans/0044` R4）。
    `None` 条件：无有效维度，或**所有维度上全部候选都缺值**。"""
    dims = _preference_dimensions(dimensions)
    if not candidates or not dims:
        return None

    def _num(shop: dict, field: str) -> float | None:
        """取维度数值；缺字段 / 非数值一律 `None`（不冒充 0）。"""
        try:
            return float(shop.get(field))
        except (TypeError, ValueError):
            return None

    # 只要**有一个维度**在候选中存在取值就有排序依据；全部维度都缺值 ⇒ 不可算
    if all(
        all(_num(s, _DEPENDS_ON_FIELDS[d][0]) is None for s in candidates) for d in dims
    ):
        return None

    def _key(shop: dict) -> tuple:
        parts: list[tuple[bool, float]] = []
        for d in dims:
            field, reverse = _DEPENDS_ON_FIELDS[d]
            v = _num(shop, field)
            # 缺值位为 True ⇒ 排最后；有值按方向（"越小越优"不取负，"越大越优"取负）
            parts.append((v is None, -(v or 0.0) if reverse else (v or 0.0)))
        return tuple(parts)

    return sorted(candidates, key=_key)


def _narrow_by_dimensions(candidates: list[dict], dimensions: Any, top_k: int = 1) -> list[dict]:
    """按**有序维度列表**把候选收窄到 `top_k` 家（`specs/0024` §2 目标 2 / `plans/0043` / `specs/00xx` 问题 A）；**单维度**同旧 `_narrow_by_depends_on`（`price`→`avgPrice` 最小、`score` 最大、`distance` 最小、`sold` 最大），**多维度**按先后做**字典序**（刻意不用数值权重，量纲不同难解释难测）。
    **缺字段候选排最后**（绝不当 0 误胜）；维度非法/空/**所有维度全缺值** ⇒ **原样返回**。
    :param top_k: 要收窄到几家；`<=1` 等价旧行为（返回 `[best]`）。`top_k > 候选数` 时**返回全部候选**（候选池不足不报错）。
    :returns: 排序后**前 `min(top_k, len)` 条**；不可算返回**原样全量**（调用方按 `len(top_k)` 判定是否收窄成功）。
    """
    ordered = _sorted_by_dimensions(candidates, dimensions)
    if ordered is None:
        return list(candidates)
    return ordered[: max(1, int(top_k))]


#: 多方案（`specs/0027`）：维度 → 用户向**方案名**。与 `_PREFERENCE_LABELS` 同源集中定义（单一来源）。
_SCHEME_LABELS: dict[str, str] = {
    "price": "最便宜",
    "score": "评分最高",
    "distance": "离得最近",
    "sold": "最热门",
}


def _multi_scheme_enabled() -> bool:
    """多方案开关（默认开）。**调用时读 env**（导入期常量陷阱）。"""
    return _env_flag("AGENT_MULTI_SCHEME_ENABLED", True)


def _multi_scheme_max() -> int:
    """最多几套方案（默认 3）。**调用时读 env**。"""
    try:
        return max(1, int(os.getenv("AGENT_MULTI_SCHEME_MAX", "3")))
    except (TypeError, ValueError):
        return 3


def _multi_scheme_items() -> int:
    """每套方案最多几家（默认 5）。**调用时读 env**。"""
    try:
        return max(1, int(os.getenv("AGENT_MULTI_SCHEME_ITEMS", "5")))
    except (TypeError, ValueError):
        return 5


def _build_schemes(preferences: Any, records: list[dict], cards: list[dict]) -> list[dict]:
    """按偏好维度**各出一套方案**（`specs/0027`），**纯函数**（不调模型、不读全局、不改入参）；仅 `preferences` 含 **≥2 有效维度**才产出（否则 `[]`，行为同 `plans/0043`），**最多 `AGENT_MULTI_SCHEME_MAX`（默认 3）套**。
    每套 `{dimension, label, items[]}`，`items` 为排序后的**店铺卡片**（**与 `shops` 同形状**，前端复用，**每套 ≤ `AGENT_MULTI_SCHEME_ITEMS`（默认 5）家**）；**方案间不互斥**（同店可现多套），排序复用 `_sorted_by_dimensions`。"""
    dims = _preference_dimensions(preferences)
    if len(dims) < 2:
        return []
    card_by_id = {
        c.get("shop_id"): c
        for c in cards
        if isinstance(c, dict) and c.get("shop_id") is not None
    }
    items_per = _multi_scheme_items()
    schemes: list[dict] = []
    for dim in dims[: _multi_scheme_max()]:
        ordered = _sorted_by_dimensions(records, [dim]) or list(records)
        items: list[dict] = []
        for rec in ordered:
            card = card_by_id.get(_shop_id_of(rec))
            if card is not None:
                items.append(card)
            if len(items) >= items_per:
                break
        schemes.append(
            {"dimension": dim, "label": _SCHEME_LABELS.get(dim, dim), "items": items}
        )
    return schemes


def _narrow_by_depends_on(candidates: list[dict], depends_on: str | None) -> list[dict]:
    """单维度收窄（**保留旧入口**，行为与改造前完全一致；多维度请用 `_narrow_by_dimensions`）。"""
    return _narrow_by_dimensions(candidates, [depends_on] if depends_on else [])


def _react_shop_list(tool: str, shops: list, shops_error: str) -> dict:
    """`shops_by_name` / `shops_nearby` 的产出：候选记录列表（`tasks=()`），无 id 的元素丢弃。
    - 记录走统一形状 ⇒ 渲染与候选提取只需一条路径（旧稿把它们放在 worker 级 `shops` 键上，渲染层没有 `react` 分支 ⇒ 一个块都不产生，见 specs/0009 §1.1 ③）；
    - 搜不到时的说明放 worker 级 `error`（旧稿是 `shops_error`，那个键不在 §2.3 的 worker 级键表里）。"""
    items = []
    for shop in shops:
        sid = _shop_id_of(shop)
        if sid is None:
            continue
        items.append(_new_shop_record(sid, shop.get("name"), tasks=(), **_shop_dims(shop)))
    out: dict[str, Any] = {"worker": "react", "tool": tool, "items": items, "error": ""}
    if shops_error:
        out["error"] = shops_error
    return out


async def _react_exec_tool(action: ReactAction, state: HmdpState, narrowed: list[dict]) -> dict:
    """执行一步只读工具，返回一条 worker_outputs 形状的记录（`items` 里是统一记录）。
    约束 1：白名单（`REACT_READONLY_TOOLS`）之外的任何 `tool` 一律拒绝执行 —— 这是"连续下单不会落到模型手里"在代码路径上的保证，不依赖模型自觉。
    约束 3：任何失败都正常返回（带 `failed: True` / `error`），绝不向图抛异常（环内一次取数失败不能把整轮问答打挂）。"""
    tool = (action.tool or "").strip()
    args = action.args or {}
    user_id = state.get("user_id") or ""

    if tool not in REACT_READONLY_TOOLS:
        logger.warning("react_loop 拒绝非白名单工具: %s", tool)
        return {
            "worker": "react", "tool": tool, "items": [], "failed": True,
            "error": f"工具 {tool} 不在只读白名单内，已拒绝",
        }

    def _fail(msg: str) -> dict:
        return {"worker": "react", "tool": tool, "items": [], "failed": True, "error": msg}

    def _resolve_sid() -> int | None:
        """本次要点名的店 id：`args` 优先，其次回退到已收窄的候选。
        不写 `narrowed[0]["id"] if narrowed else 0`（候选为空会静默去查 sid=0 这个不存在的店，正是"点名多家 + 按维度定位"路径的退化成因，specs/0009 §1.4 / 验收 4）；这里显式返回 `None`，由调用方 `_fail` 报缺参。"""
        raw = args.get("shop_id")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None
        if narrowed:
            return _shop_id_of(narrowed[0])
        return None

    try:
        if tool in ("shop_detail", "shop_vouchers", "shop_rag"):
            sid = _resolve_sid()
            if sid is None:
                return _fail("缺少店铺参数，无法取数")

        if tool == "shop_detail":
            res, ok = await _run_with_timeout(asyncio.to_thread(fetch_shop_detail, sid, user_id), tool)
            if not ok:
                return _fail("取数失败")
            # 详情不存在/失败时上游返回**字符串** ⇒ 只填 error，不编造维度字段
            record = (
                _new_shop_record(sid, res.get("name"), tasks=("detail",), **_shop_dims(res))
                if isinstance(res, dict)
                else _new_shop_record(sid, tasks=("detail",), error=res)
            )
            return {"worker": "react", "tool": tool, "items": [record], "error": ""}
        if tool == "shop_vouchers":
            res, ok = await _run_with_timeout(asyncio.to_thread(fetch_shop_vouchers, sid, user_id), tool)
            if not ok:
                return _fail("取数失败")
            # 归一化：该店无券时 fetch_shop_vouchers 返回字符串
            vouchers, vouchers_error = _split_list_or_error(res or [])
            return {"worker": "react", "tool": tool, "items": [
                _new_shop_record(
                    sid, tasks=("voucher",), vouchers=vouchers, vouchers_error=vouchers_error,
                    # 0006.1：与 `shop_worker` 的券分支同口径地附上下单卡片（纯函数、零出站）
                    order_cards=[prepare_order_card(v, sid) for v in vouchers],
                )
            ], "error": ""}
        if tool == "shops_by_name":
            name = str(args.get("name") or state.get("query") or "")
            res, ok = await _run_with_timeout(asyncio.to_thread(fetch_shops_by_name, name, 1, user_id), tool)
            if not ok:
                return _fail("取数失败")
            # 归一化：搜不到时 fetch_shops_by_name 返回字符串
            shops, shops_error = _split_list_or_error(res or [])
            return _react_shop_list(tool, shops, shops_error)
        if tool == "shops_nearby":
            type_name = str(args.get("type_name") or "")
            res, ok = await _run_with_timeout(
                asyncio.to_thread(fetch_shops_nearby, type_name, state.get("x"), state.get("y"), 1, user_id), tool
            )
            if not ok:
                return _fail("取数失败")
            # 归一化：同上
            shops, shops_error = _split_list_or_error(res or [])
            return _react_shop_list(tool, shops, shops_error)
        if tool == "shop_rag":
            query = str(args.get("query") or state.get("query") or "")
            res, ok = await _run_with_timeout(
                asyncio.to_thread(fetch_rag_chunks, sid, query, RAG_TOP_K), tool
            )
            if not ok:
                return _fail("取数失败")
            # 归一化：检索不到时 fetch_rag_chunks 返回字符串
            chunks, chunks_error = _split_list_or_error(res or [])
            return {"worker": "react", "tool": tool, "items": [
                _new_shop_record(sid, tasks=("rag",), chunks=chunks, chunks_error=chunks_error)
            ], "error": ""}
    except Exception as exc:  # noqa: BLE001  约束 3：环内失败只降级，不外抛
        logger.error("react_loop 执行 %s 失败: %s", tool, exc)
        return _fail(str(exc))

    return _fail(f"未知工具 {tool}")


async def _react_decide(
    state: HmdpState, candidates: list[dict], history: list[dict], step: int = 0
) -> list[ReactAction]:
    """问模型：下一步调哪些工具。返回空列表 = 收敛（模型认为数据够了）。
    走 `bind_tools(REACT_TOOLS)`：键名由框架生成，不会像旧 `with_structured_output` 那样漂移到 `{"name":…,"arguments":…}` 再被 Pydantic 洗成"合法空动作"⇒ 空转；三层契约 = 枚举（绑定列表 + `REACT_READONLY_TOOLS` 运行时检查）/ `extra="forbid"`（`@tool.args_schema` + 这里 `model_validate` 真校验，LangChain 转 OpenAI schema 会剥掉 `additionalProperties`，只有运行时拦得住）/ 空 `tool_calls` ⇒ 收敛。
    模型只做语义判断（还差什么信息），不做结构判断（要不要进环），也不负责排序/收窄（`_narrow_by_depends_on` 已算完）；提示词补了 `depends_on` 字段方向、「什么算够了」、「排序不由模型做」三项。"""
    plan = plan_of(state)
    # 与规则侧**共用同一份维度判据**（`specs/0024` / `plans/0043`）：偏好优先、回退旧 depends_on。
    #    否则会出现"规则按偏好收窄了、提示词却说未收窄"的静默错位。
    dims = _plan_ranking_dimensions(plan)

    if dims:
        dep_text = " > ".join(f"{d}（{_DEPENDS_ON_HINTS[d]}）" for d in dims)
        candidate_note = "（已按该偏好顺序收窄，排序无需你做）"
    else:
        # 非法/缺失维度**不臆造**：`_narrow_by_dimensions` 此时原样返回候选，
        #    提示词也必须如实说"未被收窄"，否则模型会以为已经定到某一家而直接作答。
        dep_text = "未指定（候选未被规则收窄）"
        candidate_note = "（未按维度收窄，需要你自己判断该查哪一家）"

    prompt = (
        "你在帮用户补齐一条「先比较、再定位到某一家」的查询。\n"
        "分工：**排序与收窄已经由代码按规则算完了，你只负责判断「下一步还缺哪份数据」**；\n"
        "      不要再比较或重排候选，也不要重复调用「已完成的操作」里出现过的工具。\n"
        f"用户的原始问题：{state.get('query') or ''}\n"
        f"用户想要的收敛维度：{dep_text}\n"
        f"已有候选{candidate_note}（{len(candidates)} 家）：{_react_prompt_candidates(candidates[:5])}\n"
        f"已完成的操作：{[h.get('tool') for h in history]}\n\n"
        # 工具清单**不再手写** —— 由 `bind_tools` 把 schema 随请求发出，
        # 模型看到的是框架生成的标准定义，不存在"清单与白名单对不上"的可能。
        "**什么都不调**即表示你认为数据已经够了（无需再取数）：\n"
        "- 用户问的那份数据已经取回（已出现在「已完成的操作」里）；或\n"
        "- 本来就没有可补的数据（候选为空，或本轮不需要额外取数）。\n"
        "拿不准就**什么都不调** —— 多取一次就多花一次调用，且汇总模型已经能基于现有数据作答。\n"
        "若确实缺数据，可以**一次调用多个**互不依赖的工具（它们会被并行执行）。"
    )
    try:
        # 提醒走 **user 侧**（`specs/0014` T2）：system 随步数变会击穿 prompt cache。
        messages = [SystemMessage(content=prompt), *_react_step_reminder(step)]
        msg = await get_light_model().bind_tools(list(REACT_TOOLS)).ainvoke(messages)
    except Exception as exc:  # noqa: BLE001
        logger.warning("react_loop 决策失败，提前收敛: %s", exc)
        return []

    return _react_actions_from_tool_calls(getattr(msg, "tool_calls", None) or [])


def _react_actions_from_tool_calls(calls: list[Any]) -> list[ReactAction]:
    """把标准 `tool_calls` 解析成内部动作列表，在运行时完成枚举与参数校验。
    必须再校验：① `bind_tools` 只保证发出的工具是这 5 个，模型仍可能回不在 `REACT_READONLY_TOOLS` 里的名字 ⇒ 白名单检查留在运行时（"写操作不进环"的硬边界）；
    ② LangChain 转 OpenAI schema 会剥掉 `additionalProperties: false`，只有这里用 `args_schema.model_validate()` 才拦得住。校验失败不抛异常：单个工具非法就跳过，一个都不剩则返回 `[]`（等价收敛）。"""
    actions: list[ReactAction] = []
    schema_by_name = {t.name: getattr(t, "args_schema", None) for t in REACT_TOOLS}

    for call in calls:
        name = str(call.get("name") or "").strip() if isinstance(call, dict) else str(
            getattr(call, "name", "") or ""
        ).strip()
        raw_args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
        args = dict(raw_args) if isinstance(raw_args, dict) else {}

        if not name:
            # 空工具名在新通道里不应出现（键名由框架生成）；真出现了就当收敛信号。
            logger.warning("react_loop 收到空工具名，视为收敛信号（新通道下不应发生）")
            continue
        if name not in REACT_READONLY_TOOLS:
            logger.warning("react_loop 拒绝非白名单工具: %s", name)
            continue

        schema = schema_by_name.get(name)
        if schema is not None:
            try:
                validated = schema.model_validate(args)
                args = validated.model_dump()
            except Exception as exc:  # noqa: BLE001
                logger.warning("react_loop 工具参数校验失败，跳过该工具: %s / %s", name, exc)
                continue

        actions.append(ReactAction(tool=name, args=args))

    return actions


async def _midrun_water_level(state: HmdpState, step: int) -> dict | None:
    """环内**每步**的 token 水位检查（`specs/0014` T3）；**默认只监测、不压缩**（`AGENT_REACT_MIDRUN_COMPRESS` 默认 false）：先把水位**可见**（每步超预算记 warning），压缩作开关后行为，等实测环内步数变多再开。
    压缩若开启，**必须沿用与 `trim_history` 完全相同的原子决策**：摘要生成失败 ⇒ **一条原文都不删**（否则记忆永久丢失）。"""
    messages = state.get("messages") or []
    budget = token_budget()
    if not should_compress(messages, budget):
        return None

    over = estimate_messages_tokens(messages)
    if not _env_flag("AGENT_REACT_MIDRUN_COMPRESS", False):
        logger.warning(
            "react_loop step=%s 水位超预算（估算 %s > 预算 %s）；环内压缩未开启，只记录",
            step, over, budget,
        )
        return None

    cut = split_index(messages, budget)
    outdated = [m for m in messages[:cut] if getattr(m, "id", None)]
    if not outdated:
        return None
    new_summary = await _summarize_history(state.get("history_summary") or "", outdated, state)
    if not new_summary:
        # 原子决策：摘要没生成 ⇒ 保留原文（绝不"删了再说"）
        logger.warning("react_loop step=%s 环内压缩未产出摘要，保留原文", step)
        return None
    logger.info("react_loop step=%s 环内压缩：移除 %s 条早期消息", step, len(outdated))
    return {
        "messages": [RemoveMessage(id=m.id) for m in outdated],
        "history_summary": new_summary,
    }


async def react_loop(state: HmdpState) -> dict:
    """ReAct 补跳：只在目标依赖中间结果时才实际工作（固定编排之后的第二跳，specs/0005 §5.5.4）。
    守卫一：`target_depends_on_result=False` 时立即 `return {}`（即使上游 `route_after_fanout` 漏判也不产生副作用）。
    守卫二（`plans/0009.2` N7）：`narrowed_shop_id` 非空（规则已收窄到唯一一家）时同样立即返回，不白烧模型调用。"""
    plan = plan_of(state)
    if not plan or not plan.target_depends_on_result:
        # 守卫：无依赖 → 什么都不做（验收 5.3）
        return {}
    if _narrowed_ids(state):
        # 守卫：目标已被规则收窄确定（单店或多店都算确定）→ 补跳无事可做（决策 1-B / `specs/00xx`）
        return {}

    record_node("react_loop")

    outputs = list(state.get("worker_outputs") or [])
    candidates = _react_candidates(outputs)

    # 规则侧先做一次结构性收窄：按偏好（或回退 depends_on）的有序维度取 Top-`top_k`
    # （`specs/00xx`：`top_k` 来自 `plan_top_k` —— 用户明说的 `top_n` 优先、其次指代下标长度，
    #   都缺省 1 ⇒ 与改造前逐字节一致）
    narrowed = _narrow_by_dimensions(candidates, _plan_ranking_dimensions(plan), plan_top_k(plan))

    produced: list[dict] = []
    history: list[dict] = []
    converged = False

    # 环内压缩若开启，删除指令与摘要在这里累积，最后随节点返回值一起写回 State
    midrun_messages: list[Any] = []
    midrun_summary = ""

    for _step in range(REACT_MAX_STEPS):
        # 每步 **LLM 调用之前**先查水位（环内消息会随取数结果膨胀）
        patch = await _midrun_water_level(state, _step + 1)
        if patch:
            midrun_messages.extend(patch.get("messages") or [])
            midrun_summary = patch.get("history_summary") or midrun_summary
        # 步数从 1 开始（给提醒文案用）；**只在本环内计数**，不写回 State（避免跨轮串味）
        actions = await _react_decide(state, narrowed, history, step=_step + 1)
        # 🆕 **收敛判据换了**：不再是"模型回了 `finish`"，而是"**一个工具都没调**"。
        # 旧通道里"空工具名"只是被拒、循环继续 ⇒ 白烧一轮；新通道下空列表就是明确的收敛信号。
        if not actions:
            converged = True
            logger.info("react_loop step=%s 收敛（模型未调用任何工具）", _step + 1)
            break

        # 多个互不依赖的只读工具 ⇒ **并行**执行（省一轮 LLM 往返 + 省一轮墙钟）。
        # 上限保护：`REACT_MAX_STEPS` 管的是"轮数"，这里再管"每轮的工具数"，
        #    避免模型一口气点 5 个工具把取数预算打满（尤其 `shop_rag` 会带重排，实测 2.3–6.6s/次）。
        if len(actions) > REACT_MAX_TOOLS_PER_STEP:
            logger.warning(
                "react_loop 单步工具数 %s 超过上限 %s，只取前 %s 个",
                len(actions), REACT_MAX_TOOLS_PER_STEP, REACT_MAX_TOOLS_PER_STEP,
            )
            actions = actions[: REACT_MAX_TOOLS_PER_STEP]

        results = await asyncio.gather(
            *(_react_exec_tool(a, state, narrowed) for a in actions),
            return_exceptions=True,
        )
        for action, result in zip(actions, results):
            history.append({"tool": action.tool})
            if isinstance(result, BaseException):
                # 约束 3：`_react_exec_tool` 自己已吞异常，这里是 gather 的兜底（绝不外抛）
                logger.error("react_loop 执行 %s 异常: %s", action.tool, result)
                continue
            if result.get("stop"):
                converged = True
                break
            produced.append(result)
            # 拿到新数据就更新收窄集合（例如 shop_detail 回来更全的维度字段）
            new_rec = (result.get("items") or [None])[0]
            new_sid = _shop_id_of(new_rec)
            if new_sid is not None:
                for cand in narrowed:
                    if _shop_id_of(cand) == new_sid:
                        # 只补**店维度**字段（avgPrice / score / distance / sold）—— 这才是收窄需要的；
                        # 不要把能力块（券/片段/攻略）或占位店名也刷进候选，那会让候选变得不自洽。
                        cand.update(_shop_dims(new_rec))
        logger.info("react_loop step=%s tool=%s", _step + 1, [a.tool for a in actions])
        if converged:
            break

    if not converged:
        # 约束 2：步数超限**强制收敛**（绝不无限循环）。日志字段固定为 `react_loop_truncated`
        # 便于事后 grep；落 `SUMMARY_NODE` 由 `workflow.py` 的 `add_edge("react_loop", SUMMARY_NODE)` 保证。
        logger.warning(
            "react_loop_truncated: 达到步数上限 %s 仍未收敛，强制汇总（已有 %s 项取数结果）",
            REACT_MAX_STEPS, len(produced),
        )

    if not produced:
        out: dict = {}
    else:
        out = {"worker_outputs": produced}
    if midrun_messages:
        out["messages"] = midrun_messages
        out["history_summary"] = midrun_summary
    return out


# ===========================================================================
# 第四层：汇总
# ===========================================================================
def _collect_sources(outputs: list[dict]) -> list[str]:
    """来源由代码从 worker 取回的真实数据里提取，不靠模型回忆，避免编造标题。
    统一遍历 `item["items"][].chunks | blogs` —— 记录形状统一后，ReAct 补跳取回的片段/攻略标题也能被看见（旧稿 ReAct 产出没有 `items` 键，一条来源都取不到）。"""
    titles: list[str] = []
    for item in outputs or []:
        if not isinstance(item, dict):
            continue
        for bundle in item.get("items") or []:
            if not isinstance(bundle, dict):   # str 会被逐字迭代出垃圾标题
                continue
            # 同样要归一化：chunks 是 str 时会逐字迭代出垃圾标题
            chunks, _chunks_err = _split_list_or_error(bundle.get("chunks") or [])
            for chunk in chunks:
                title = str(chunk.get("title") or "").strip()
                if title and title not in titles:
                    titles.append(title)
            blogs, _blogs_err = _split_list_or_error(bundle.get("blogs") or [])
            for blog in blogs[:CONTEXT_BLOGS]:
                title = str(blog.get("title") or "").strip()
                if title and title not in titles:
                    titles.append(title)
    return titles[:MAX_SOURCE]


def _collect_focus(outputs: list[dict], fallback: list[dict]) -> list[dict]:
    """把本轮实际用到的店铺沉淀为跨轮焦点（累积 reducer 负责去重追加）；无 nearby 产出时返回 fallback，去重后不会重复追加。
    候选列表的键已由 `shops` 统一为 `items`（specs/0009 §2.2），漏改会让多轮指代「这家店」静默失效（`_collect_focus` 是跨轮焦点的唯一沉淀点）。"""
    focus: list[dict] = []
    seen: set[int] = set()
    for item in outputs or []:
        if not isinstance(item, dict) or item.get("worker") != "nearby":
            continue
        for shop in item.get("items") or []:
                focus_item = _to_focus_item(shop)
                if focus_item is None or focus_item["id"] in seen:
                    continue
                seen.add(focus_item["id"])
                focus.append(focus_item)
    if focus:
        return focus[:NEARBY_TOP]
    return list(fallback or [])


def _render_shop_record(rec: dict) -> str:
    """把一条统一记录渲染成一个块；小节由 `rec["tasks"]` 显式决定。
    同一条记录多标签 ⇒ 同一块内多小节（验收 7：同时有详情与券时两者都进上下文）；这里不做 `isinstance(rec.get("shop"), dict)` 之类的形状嗅探（验收 8）。"""
    lines = [_shop_header(rec)]
    tasks = set(rec.get("tasks") or [])

    if not tasks:
        # 纯候选 / 纯定位结果（`nearby` 的 items、react 的 `shops_by_name` / `shops_nearby`）：
        # 没有能力块，只出一行店维度 —— 而它们正是"按维度定位"要看的数据。
        dims: list[str] = []
        if rec.get("distance") is not None:
            dims.append(f"距离{rec['distance']}m")
        if rec.get("avgPrice"):
            dims.append(f"人均{rec['avgPrice']}元")
        score = rec.get("score")
        if isinstance(score, (int, float)) and score:
            dims.append(f"评分{score / 10:.1f}分")
        elif score:
            dims.append(f"评分{score}")
        if rec.get("sold"):
            dims.append(f"销量{rec['sold']}")
        if rec.get("address"):
            dims.append(f"地址：{rec['address']}")
        if dims:
            lines.append("；".join(dims))
        if rec.get("error"):
            lines.append(f"查询失败：{rec['error']}")
        return "\n".join(lines)

    if "detail" in tasks:
        score = rec.get("score")
        score_text = f"{score / 10:.1f}分" if isinstance(score, (int, float)) and score else "暂无评分"
        distance = f"{rec.get('distance')}m" if rec.get("distance") is not None else "距离暂未获取"
        lines.extend([
            f"地址：{rec.get('address') or '暂未获取'}",
            f"人均：{rec.get('avgPrice') or '暂未获取'}元；评分：{score_text}；销量：{rec.get('sold') or 0}",
            f"营业时间：{rec.get('openHours') or '营业时间暂未获取'}；距离：{distance}",
        ])
        # 必须先用 _split_list_or_error：若 blogs 是 str，会被**逐字迭代**
        blogs, blogs_error = _split_list_or_error(rec.get("blogs") or [])
        if rec.get("blogs_error") or blogs_error:
            lines.append(f"攻略：{rec.get('blogs_error') or blogs_error}")
        for blog in blogs[:CONTEXT_BLOGS]:
            lines.append(f"攻略《{blog.get('title')}》：{(blog.get('content') or '')[:100]}")

    if "voucher" in tasks:
        # 同 blogs：若 vouchers 是 str，`for v in "店铺 9 暂时没有代金券"` 会**逐字迭代**出满屏单字
        vouchers, vouchers_error = _split_list_or_error(rec.get("vouchers") or [])
        if rec.get("vouchers_error") or vouchers_error:
            lines.append(f"代金券：{rec.get('vouchers_error') or vouchers_error}")
        for voucher in vouchers:
            # 金额单位是**分**（`tb_voucher.pay_value` / `actual_value`）⇒ 必须换算后再拼「元」，
            # 否则 47.5 元的券会被说成 4750 元。见 `_yuan` 与其 bug 台账。
            lines.append(
                f"代金券《{voucher.get('title')}》 售价{_yuan(voucher.get('payValue'))}元 "
                f"抵{_yuan(voucher.get('actualValue'))}元 库存{voucher.get('stock')} "
                f"规则：{voucher.get('rules') or '无'}"
            )

    if "rag" in tasks:
        chunks, chunks_error = _split_list_or_error(rec.get("chunks") or [])
        if rec.get("chunks_error") or chunks_error:
            lines.append(f"细节：{rec.get('chunks_error') or chunks_error}")
        for chunk in chunks:
            lines.append(f"细节《{chunk.get('title')}》：{(chunk.get('content') or '')[:200]}")

    if rec.get("error"):
        lines.append(f"查询失败：{rec['error']}")
    return "\n".join(lines)


def _record_counts(rec: dict) -> str:
    """同店块的**数量摘要**（`specs/0016` T2 / §8 Q2）：`含2张券；1条细节；1篇攻略`，券面/片段原文由前端从 `payload` 确定性渲染，这里只报**数量**。
    计数走 `_split_list_or_error`：`str` 会被**逐字迭代**，直接 `len()` 会得到"字数"这种垃圾。"""
    vouchers, _ = _split_list_or_error(rec.get("vouchers") or [])
    chunks, _ = _split_list_or_error(rec.get("chunks") or [])
    blogs, _ = _split_list_or_error(rec.get("blogs") or [])
    parts: list[str] = []
    if vouchers:
        parts.append(f"含{len(vouchers)}张券")
    if chunks:
        parts.append(f"{len(chunks)}条细节")
    if blogs:
        parts.append(f"{len(blogs)}篇攻略")
    return "；".join(parts)


def _render_shop_block(rec: dict) -> str:
    """渲染一条店记录，并在标题下补一行**数量摘要**（无能力块时不加）。"""
    block = _render_shop_record(rec)
    counts = _record_counts(rec)
    if not counts:
        return block
    head, _sep, rest = block.partition("\n")
    return f"{head}\n{counts}" + (f"\n{rest}" if rest else "")


def _merge_shop_records(records: list[dict]) -> list[dict]:
    """把**同一家店**的多条记录归并成一条（`specs/0016` T2 / §10.1(a)），走与 react / 抽卡同一条读取路径 `_react_candidates`（并 `tasks`、非空覆盖空）。
    返回**按店 id 升序**（确定性，并行完成顺序不影响结果）；无 `id` 的记录无法归并，原样放末尾（不丢）。"""
    by_id: dict[int, list[dict]] = {}
    orphans: list[dict] = []
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        sid = _shop_id_of(rec)
        if sid is None:
            orphans.append(rec)
            continue
        by_id.setdefault(sid, []).append(rec)
    merged: list[dict] = []
    for sid in sorted(by_id):
        merged.extend(_react_candidates([{"items": by_id[sid]}]))
    return merged + orphans


def _render_order(outputs: list[dict]) -> list[dict]:
    """把 worker 产出排成稳定的渲染顺序（`plans/0009.2` N9 / 验收 17）。
    `worker_outputs` 走 `add_and_clear` 追加，多个并行 Send 的写入顺序即完成顺序 ⇒ 块序会抖、答复措辞不稳定、A/B 无法逐字对照。
    规则：`nearby` / `grounding` 块恒定前置；其余按记录内最小店 id 升序；同键保持原相对顺序。只作用于块之间，`nearby` 自身候选顺序不得重排（那是后端返回顺序，承载"距离最近"的语义）。"""
    def _key(pair: tuple[int, dict]) -> tuple[int, int, int]:
        idx, item = pair
        if not isinstance(item, dict):
            return (2, 0, idx)
        if item.get("worker") in ("nearby", "grounding"):
            return (0, 0, idx)
        sids = [
            sid
            for sid in (_shop_id_of(rec) for rec in (item.get("items") or []) if isinstance(rec, dict))
            if sid is not None
        ]
        return (1, min(sids) if sids else 10 ** 9, idx)

    return [item for _idx, item in sorted(enumerate(outputs or []), key=_key)]


def render_context(outputs: list[dict], focus: list[dict]) -> str:
    """把 worker 的紧凑结果渲染成给汇总模型的文本，并在这一步完成裁剪。
    分派按显式标签（`record["tasks"]`），不再按"某字段存不存在 / 是不是 dict"（specs/0009 §2.4 / 验收 8）——
    旧稿以 `isinstance(bundle["shop"], dict)` 为唯一判据，导致"只有券"的产出一个块都渲染不出、同店有详情又有券时静默丢券；块序先经 `_render_order` 稳定化（验收 17）。"""
    if not outputs:
        return "（没有查到任何资料）"

    blocks: list[str] = []
    pending: list[dict] = []
    for item in _render_order(outputs):
        if not isinstance(item, dict):
            continue
        worker = item.get("worker")

        if worker == "nearby":
            if item.get("error"):
                blocks.append(f"【附近找店】{item['error']}")
                continue
            records = item.get("items") or []
            if not records:
                blocks.append(f"【附近找店】{item.get('empty') or '没有找到相关店铺'}")
                continue
            lines = [f"【附近找店 · {item.get('type_name') or ''}】"]
            for idx, rec in enumerate(records, 1):
                distance = f"{rec.get('distance')}m" if rec.get("distance") is not None else "距离暂未获取"
                score = rec.get("score")
                score_text = f"{score / 10:.1f}分" if isinstance(score, (int, float)) and score else "暂无评分"
                avg = f"人均{rec.get('avgPrice')}元" if rec.get("avgPrice") else "人均暂未获取"
                lines.append(
                    f"{idx}. {rec.get('name')}（id={_shop_id_of(rec)}） {distance} {score_text} {avg} "
                    f"销量{rec.get('sold') or 0} 地址：{rec.get('address') or '暂未获取'}"
                )
            # 「追问更多」时补一句"这是接续"（仅该场景有值）——否则汇总层可能把本批当成全部、
            # 或与前文已列的那几家混在一起重复表述。
            if item.get("listing_note"):
                lines.append(str(item["listing_note"]))
            blocks.append("\n".join(lines))
            continue

        if worker == "grounding":
            # 「点名店铺没解析出来」的说明必须进上下文：否则汇总模型只看到候选列表、不知道点名的店没找到，会回一句空洞推荐。
            # 只渲染 `notice`：原始错误（URL / 异常类型）不得进入上下文与答复。
            if item.get("notice"):
                blocks.append(f"【店铺解析】{item['notice']}")
            continue

        # 其它 worker（detail / voucher / rag / react_loop）：先收集，**同店收敛成一块**
        # （`specs/0016` T2 / §10.1(a)）：同一家店的多条记录（详情/券/片段）按 id 归并后**只渲染一次**，
        # 并附**数量摘要**（`含N张券`）—— 原始块不再并列多次；`nearby` / `grounding` 的专用块不变。
        for rec in item.get("items") or []:
            if isinstance(rec, dict):
                pending.append(rec)

    for rec in _merge_shop_records(pending):
        blocks.append(_render_shop_block(rec))

    if focus and not any("id=" in b for b in blocks):
        blocks.append(
            "【本轮涉及的店铺】" + "；".join(f"{s.get('name')}(id={s.get('id')})" for s in focus)
        )
    return "\n\n".join(blocks)


async def _summarize_stream(messages: list, sources: list) -> str:
    """流式生成最终回答；结构化失败时的兜底，也是产出自然语言的主路径。"""
    parts: list[str] = []
    record_model_call("summarize")
    await _stream_join(get_main_model().astream(messages), parts)
    raw = "".join(parts).strip()
    parsed_answer, parsed_sources = _parse_possible_json(raw)
    answer = parsed_answer or raw
    if parsed_sources:
        sources[:] = parsed_sources[:MAX_SOURCE]
    return answer


async def _summarize_structured_stream(messages: list, sources: list) -> str:
    """【已废弃，仅留作记录】曾试图用 `with_structured_output` 兼得「JSON 结构 + 逐 token」，实测此路不通。
    ① 结构化调用不做增量解析，整条响应只产出一个 `RagAnswer`；② 底层 HTTP 仍流式，LangGraph 透传裸 JSON token（实测 62 chunk 中 57 个为空，非空的还是 `{"source": [], ` 这类语法）。
    ⇒ 要真逐 token 必须走纯文本调用（见 `_summarize_plain_stream`）；本函数保留只为留下"为什么不用结构化流式"的证据。"""
    raise NotImplementedError(
        "结构化输出无法逐 token 流式（实测 62 chunk/57 空），请用 _summarize_plain_stream"
    )


async def _summarize_plain_stream(messages: list, sources: list) -> tuple[str, Any]:
    """逐 token 流式产出最终回答（纯文本，不带 JSON 约束），是 `summarize_answer` 的主路径；在 `summarize_answer` 内调用（该节点在 `STREAM_NODES` 白名单，token 转发前端），`SUMMARY_SYSTEM` 已要求不输出 JSON 仍保留 `_parse_possible_json` 防御。
    **返回 `(answer, 带 usage 的 chunk)`**（`specs/0013` §11.1）：第二个值是最近一次带 `usage_metadata` 的 chunk（无则退回聚合体），调用方拿它读 usage 建压缩锚点，最后一个 chunk 无 usage（见 `_stream_join` 实测）。
    调用点文本**必须**保持 `_summarize_plain_stream(messages, sources)` 形态（`test_sse_answer_streaming` 有源码级接线断言），只改返回值、不改调用。"""
    parts: list[str] = []
    record_model_call("summarize")
    response = await _stream_join(get_main_model().astream(messages), parts)
    raw = "".join(parts).strip()
    parsed_answer, parsed_sources = _parse_possible_json(raw)
    answer = parsed_answer or raw
    if parsed_sources:
        sources[:] = parsed_sources[:MAX_SOURCE]
    return answer, response


def _summarize_messages(state: HmdpState) -> tuple[list, list, str]:
    """构造 summarize 系节点共用的 messages / 初始 sources / context。
    抽出来是为了让「流式节点」与「回落节点」用完全相同的输入各调一次模型，否则回落时上下文不一致，等于换了个问题再问一遍。
    汇总**只有纯文本一版**（`prompts.summary_system()` 无参）：流式要逐 token，而结构化输出做不到（见 `_summarize_structured_stream` 的实测记录）。"""
    query = state.get("query") or ""
    outputs = list(state.get("worker_outputs") or [])
    focus = list(state.get("focus_shops") or [])
    sources = _collect_sources(outputs)
    context = render_context(outputs, focus)
    # 未覆盖活动**确定性标注**（`specs/0029` §2 目标 3）：多活动问句被丢掉一部分时，
    #    汇总模型此前会自由发挥成"暂时没有KTV/做发型"——把**没查**说成**没有**。
    #    这里由**代码**注入事实（哪些活动本轮没查），并要求汇总层如实转达（措辞由模型组织）。
    uncovered = _uncovered_activities(query, plan_of(state))
    if uncovered:
        logger.info("未覆盖活动标注：本轮尚未查询 %s", list(uncovered))
        context = f"{context}\n\n{UNCOVERED_ACTIVITIES_NOTE.format(terms='、'.join(uncovered))}"
    messages = [
        SystemMessage(summary_system()),
        *_summary_block(state),
        *_history_before_current(state.get("messages"), token_budget()),
        HumanMessage(f"用户问题：{query}\n\n已查到的资料：\n{context}"),
    ]
    return messages, sources, context


def _terminal_cleanup(state: HmdpState, recommended: list[dict] | None = None) -> dict[str, Any]:
    """**终态收尾**：清掉本轮中间结果，绝不能被 checkpoint 带进下一轮（脏上下文）；漏清 `worker_outputs` 会把候选变成下一轮"已知事实"，渲染无关店。
    `summarize_answer` 与 `suggest` 共用本函数，终态字段须逐字段一致；**记录还在的这一刻**做摘要投影（`specs/0016` §10.3）：`worker_outputs` 马上清空，只有这里能读到 ⇒ 把 `[已推荐]`/`[死路]` 投影进 State 供下一轮消费（`recommended` 传本轮真正展示的店，plan 轨不传退回按记录推）。"""
    outputs = list(state.get("worker_outputs") or [])
    focus = list(state.get("focus_shops") or [])
    return {
        "focus_shops": _collect_focus(outputs, focus),
        # 🆕 摘要的确定性养料（`specs/0016` T4）：跨轮累积（`state.merge_digest`），带上限。
        "recommended_digest": project_recommended_digest(outputs, recommended),
        "excluded_digest": project_excluded_digest(outputs),
        # 用完即清：中间结果绝不能被 checkpoint 带进下一轮
        "worker_outputs": CLEAR,
        "shop_ids": [],
        "task": "",
        "worker_tasks": [],
        "worker_shop_ids": [],
        "worker_query": "",
        "narrowed_shop_id": None,
        "narrowed_shop_ids": [],
    }


async def summarize_answer(state: HmdpState) -> dict:
    """终态权威节点：逐 token 流式产出最终回答（在 `hmdp_agent.STREAM_NODES` 白名单内，`astream` 会被转成 `messages` 事件）。
    独立成节点是因为 `summarize` 的 `with_structured_output` 被 LangGraph 跳过（0 个 messages 事件），且结构化输出无法增量解析（实测 62 chunk/57 空）。
    `sources` 改由 `_collect_sources` 从真实取数结果确定性提取；流式拿不到文本时兜底一次纯文本 `ainvoke`。"""
    record_node("summarize")
    messages, sources, context = _summarize_messages(state)

    answer = ""
    response = None
    try:
        answer, response = await _summarize_plain_stream(messages, sources)
    except Exception as exc:
        logger.error("流式汇总失败，转一次性兜底: %s", exc)

    if not answer:
        # 流式完全没拿到文本（模型异常 / 被截断）：同一份输入再要一次完整回答
        try:
            response = await get_main_model().ainvoke(messages)
            answer = as_text(getattr(response, "content", "")).strip()
            parsed_answer, parsed_sources = _parse_possible_json(answer)
            answer = parsed_answer or answer
            if parsed_sources:
                sources = parsed_sources[:MAX_SOURCE]
        except Exception as exc:
            logger.error("一次性兜底也失败: %s", exc)

    if not answer:
        answer = "暂未查到相关推荐"

    # usage 锚点必须在**主汇总调用之后**取（`specs/0013` T5 / §11.1）：下面给卡片写推荐理由还会再调一次小上下文模型，拿它当锚点会低估整包 token，使 `trim_history` 压缩触发过晚。
    # 用主汇总那次的 `response` 自查 usage，**不经回调**：同步回调在异步链路上写 ContextVar 是断的（见 `anchor_patch` 与 `bug/BUG-20261009-*`）。
    anchor = anchor_patch(response, state.get("messages") or [])

    # 推荐卡片（#117 / #120 / #121）：从 worker_outputs 抽店铺记录给每店生成 LLM 推荐理由；判据是「有没有取回店铺记录」而非 `mode_hint`（2026-10-08 真机：对话模式拿不到卡片、行程轨正常）。
    # 与行程轨对齐：真取回店铺就发卡片；`_recommend_shops` 无记录返回 `[]` 放宽也不空烧模型。抽卡失败不影响正文（降级为无卡片）。
    recommend_shops: list[dict] = []
    try:
        recommend_shops = await _recommend_shops(state)
    except Exception as exc:  # noqa: BLE001
        logger.warning("推荐卡片抽取失败（不影响正文）: %s", exc)

    # 多方案编排（`specs/0027` / `plans/0044`）：`preferences` 含 ≥2 维度时按维度各出一套方案。
    # **只加字段、不改 `recommend_shops`**（旧前端不受影响）；失败/关闭/维度 <2 ⇒ 空 ⇒ 回单方案。
    schemes: list[dict] = []
    try:
        route_plan = plan_of(state)
        if recommend_shops and _multi_scheme_enabled():
            schemes = _build_schemes(
                route_plan.preferences if route_plan else [],
                _react_candidates(list(state.get("worker_outputs") or [])),
                recommend_shops,
            )
    except Exception as exc:  # noqa: BLE001 - 多方案失败降级为单方案，绝不打断正文
        logger.warning("多方案编排失败（降级为单方案）: %s", exc)
    if schemes:
        logger.info("多方案：%s 套（维度=%s）", len(schemes), [s["dimension"] for s in schemes])

    logger.info("汇总完成（流式节点），answer 长度 %s，来源 %s 条，推荐卡片 %s 张",
                len(answer), len(sources), len(recommend_shops))
    # 卡片摘要埋点（`specs/0017` T2）：卡片**已抽完**、`recommend_shops` 就在手上，
    #    在这里把必填字段落账本，供之后冻结行为回归集。旁路：失败只 warning。
    try:
        ledger.record_card_digest(
            ledger.get_run_id(), card_digest=card_digest_of(recommend_shops), answer_digest=answer
        )
    except Exception as exc:  # noqa: BLE001 - 埋点绝不打断主链路
        logger.warning("卡片摘要埋点失败（不影响本轮）: %s", exc)
    return {
        "answer": answer,
        "sources": sources,
        # recommend 轨的结构化卡片（#117）：前端据此渲染店铺卡片 + 推荐理由 + 地图，
        #    与 `answer` 正文互补（正文是引导语，卡片是结构化结果）。daily 轨此字段为空列表。
        "recommend_shops": recommend_shops,
        # recommend 轨的多方案（`specs/0027`）：与 `recommend_shops` **并存** —— 前端"有 schemes 用
        #    schemes、否则读 shops"；单方案 / 关闭 / 失败时为空列表（`shops` 始终完整保留）。
        "schemes": schemes,
        # `messages` 必须返回：否则 checkpoint 无 AIMessage，下一轮看不到本轮回答（重复事件由消费侧去重）。
        # 写回的是摘要（`_digest_for_memory(..., AGENT_RECOMMEND_DIGEST_MAX_CHARS)`）：answer 全文只用于 SSE/DB，
        # 进 message list 的摘要避免多轮会话累积长正文撑爆预算、触发频繁压缩。
        "messages": [AIMessage(content=_digest_for_memory(answer, recommend_digest_max_chars()))],
        # 投影用**本轮真正展示**的卡片（`recommend_shops`）作为 [已推荐] 的真源（`specs/0016` T4）。
        **_terminal_cleanup(state, recommend_shops),
        **anchor,
    }


def _parse_possible_json(text: str) -> tuple[str, list[str]]:
    """模型偶尔仍会吐 JSON，这里兜底解析，解析不了就当纯文本用。"""
    import json

    cleaned = (text or "").strip()
    if not cleaned.startswith("{"):
        return "", []
    try:
        body = cleaned.strip("`")
        if body.startswith("json"):
            body = body[4:].strip()
        data = json.loads(body)
        if isinstance(data, dict) and data.get("answer"):
            return str(data["answer"]).strip(), [str(s) for s in (data.get("source") or []) if s]
    except Exception:
        pass
    return "", []


# ===========================================================================
# 兜底：需要用户补充信息
# ===========================================================================
def clarify(state: HmdpState) -> dict:
    """实体解析不出来时，直接追问，不再消耗后续 token。"""
    record_node("clarify")
    answer = ""
    # 优先用 grounding 给出的说明作答复：用户**已经点名了店**却没解析出来时，
    #    再反问"你想了解哪家店铺呢"是答非所问
    #    （见 bug/BUG-20260928-grounding-named-shop-to-clarify.md）。
    notice = _grounding_notice_of(state)
    if notice:
        answer = str(notice.get("notice") or "").strip()
        hint = str(notice.get("hint") or "").strip()
        if answer and hint:
            answer = f"{answer}，{hint}"
    plan = plan_of(state)
    if not answer and plan and plan.clarify_question:
        answer = plan.clarify_question.strip()
    if not answer:
        answer = "你想了解哪家店铺呢？可以告诉我店名，或者说说想找什么类型的店（比如火锅、咖啡）。"
    return {
        "answer": answer,
        "sources": [],
        "messages": [AIMessage(content=answer)],
        "worker_outputs": CLEAR,
        "shop_ids": [],
    }
