"""LangGraph 节点实现。

节点按职责分为两类：
- LLM 节点：entry_router、daily_chat、summarize
- Data 节点：grounding、nearby_worker、shop_worker（只取数，不生成自然语言，省 token 且稳定）
"""

import asyncio
import os
import re
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.types import Command, Send

from app.agents.graph.llm import get_light_model, get_main_model
from app.agents.graph.prompts import (
    ENTRY_SYSTEM,
    SUMMARY_MODE,
    daily_system,
    history_summary_system,
    summary_system,
)
from app.agents.graph.state import CLEAR, HmdpState, RoutePlan, plan_of
from app.agents.hmdp_metrics import record_model_call, record_node, record_usage_from_message
from app.agents.hmdp_tools import (
    fetch_rag_chunks,
    fetch_shop_blogs,
    fetch_shop_detail,
    fetch_shop_types,
    fetch_shop_vouchers,
    fetch_shops_by_name,
    fetch_shops_nearby,
)
from app.common.logger import logger
from app.models.schemas import RagAnswer

load_dotenv()

# ---- 可调参数 -------------------------------------------------------------
def _history_keep() -> int:
    """保留的对话轮数（运行时可经 AGENT_HISTORY_KEEP 调整，便于评测切换阈值）。"""
    try:
        return int(os.getenv("AGENT_HISTORY_KEEP", "8"))
    except ValueError:
        return 8

MAX_SHOPS = int(os.getenv("AGENT_MAX_SHOPS", "3"))                # 单次最多处理的店铺数
NEARBY_TOP = int(os.getenv("AGENT_NEARBY_TOP", "5"))              # 附近找店返回条数
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
CONTEXT_BLOGS = int(os.getenv("AGENT_CONTEXT_BLOGS", "3"))        # 每家店进上下文的博客数
MAX_SOURCE = int(os.getenv("AGENT_MAX_SOURCE", "3"))

# ---- 规则快通道 -----------------------------------------------------------
_GREETING = re.compile(
    r"^[\s\W_]*(你好|您好|哈喽|嗨|hi|hello|hey|谢谢|感谢|多谢|辛苦了|再见|拜拜|在吗|"
    r"你是谁|你叫什么|早上好|中午好|下午好|晚上好|嗯|哦|好的|ok|okay)[\s\W_]*$",
    re.IGNORECASE,
)

_LOCAL_KEYWORDS = (
    "店", "附近", "周边", "优惠", "团购", "折扣", "券", "好吃", "推荐", "探店",
    "人均", "营业", "几点关门", "停车", "包间", "包厢", "菜单", "排队", "等位",
    "外卖", "地址", "在哪", "怎么走", "评分", "口碑", "攻略", "排行榜",
)

_ORDINAL = {"一": 0, "二": 1, "两": 1, "三": 2, "四": 3, "五": 4, "1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
_ANAPHORA = re.compile(r"这家|那家|此店|它|它们|这两家|这几家|第二家|第[一二两三四五1-5]家|其中|哪个更好|哪家更")
# 复数指代：这类说法指向多家店，不能只取焦点里的第一家
_PLURAL_ANAPHORA = re.compile(r"两[家个]|这几家|这些|它们|都|分别|各自|对比|比较|哪个更好|哪家更")


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


def _chat_messages(messages: list[Any], limit: int):
    """取出最近若干条人类/AI 消息，过滤掉系统消息和空消息。"""
    picked = [
        m
        for m in (messages or [])
        if isinstance(m, (HumanMessage, AIMessage)) and as_text(getattr(m, "content", ""))
    ]
    return picked[-limit:]


def _history_before_current(messages: list[Any], turns: int = 2):
    """当前这条用户消息已经入 State，取它之前的若干条作为上下文。"""
    picked = _chat_messages(messages, turns * 2 + 1)
    if picked and isinstance(picked[-1], HumanMessage):
        picked = picked[:-1]
    return picked[-(turns * 2):]


# ===========================================================================
# 第一层：历史裁剪 + 入口意图识别
# ===========================================================================
async def trim_history(state: HmdpState) -> dict:
    """控制 messages 长度：超出 HISTORY_KEEP 轮后，把最早的消息用便宜模型压缩成
    一段 running summary 存入 State，再 RemoveMessage 删掉原文。既避免 token 无限膨胀，
    又尽量不丢上文语义。"""
    record_node("trim_history")
    messages = state.get("messages") or []
    keep = _history_keep() * 2
    if len(messages) <= keep:
        return {}
    outdated = [m for m in messages[:-keep] if getattr(m, "id", None)]
    if not outdated:
        return {}
    logger.info("历史压缩：拟移除 %s 条早期消息并生成摘要", len(outdated))

    prev_summary = (state.get("history_summary") or "").strip()
    new_summary = await _summarize_history(prev_summary, outdated)

    update: dict = {"messages": [RemoveMessage(id=m.id) for m in outdated]}
    if new_summary:
        update["history_summary"] = new_summary
    return update


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


async def _summarize_history(prev_summary: str, outdated: list[Any]) -> str | None:
    """用便宜模型把旧消息（连同已有摘要）压缩成新的 running summary。

    失败时沿用旧摘要，绝不因为压缩失败而丢失已有记忆。
    """
    try:
        record_model_call("history_summary")
        text = _render_for_summary(outdated)
        human = (
            f"已有的对话摘要：\n{prev_summary}\n\n"
            f"需要并入的新一轮对话：\n{text}\n\n"
            f"请输出合并后的最新摘要："
        )
        resp = await get_light_model().ainvoke(
            [SystemMessage(history_summary_system()), HumanMessage(human)]
        )
        summary = as_text(getattr(resp, "content", "")).strip()
        return summary or prev_summary or None
    except Exception as exc:
        logger.error("历史摘要生成失败，沿用旧摘要: %s", exc)
        return prev_summary or None


def _summary_block(state: HmdpState) -> list[SystemMessage]:
    """把压缩摘要作为一条系统消息注入下游 prompt，弥补被删除的早期消息。"""
    s = (state.get("history_summary") or "").strip()
    if not s:
        return []
    return [
        SystemMessage(
            "以下是更早若干轮对话的压缩摘要（原文已移除，关键信息保留在此）：\n" + s
        )
    ]


def _fast_plan(query: str, mode_hint: str) -> RoutePlan | None:
    """规则快通道：命中就完全不调模型。"""
    if not query:
        return RoutePlan(scene="daily", rewritten_query="")
    text = query.strip()
    if _GREETING.match(text):
        return RoutePlan(scene="daily", rewritten_query=text)
    # 日常入口且没有任何本地生活关键词：直接走闲聊，省掉一次模型调用
    if mode_hint == "daily" and not any(k in text for k in _LOCAL_KEYWORDS):
        return RoutePlan(scene="daily", rewritten_query=text)
    return None


async def _llm_plan(state: HmdpState, query: str, mode_hint: str) -> RoutePlan:
    """便宜模型做意图识别 + 实体抽取，失败重试后回落到 mode_hint。"""
    router = get_light_model().with_structured_output(RoutePlan)
    messages = [
        SystemMessage(ENTRY_SYSTEM),
        *_summary_block(state),
        *_history_before_current(state.get("messages"), turns=2),
        HumanMessage(f"入口模式：{mode_hint}\n用户问题：{query}\n请只返回 json 对象。"),
    ]
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            record_model_call("entry_router")
            result = await router.ainvoke(messages)
            plan = result if isinstance(result, RoutePlan) else RoutePlan.model_validate(result)
            return plan
        except Exception as exc:  # 分类失败不能让整个请求挂掉
            last_error = exc
            logger.warning("意图识别失败（第 %s 次）: %s", attempt, exc)
            await asyncio.sleep(min(1.5 ** attempt, 5))

    logger.error("意图识别连续失败，按入口模式兜底: %s", last_error)
    fallback_scene: Literal["daily", "recommend"] = "daily" if mode_hint == "daily" else "recommend"
    return RoutePlan(
        scene=fallback_scene,
        sub_intents=["nearby"] if fallback_scene == "recommend" else [],
        rewritten_query=query,
    )


async def entry_router(
    state: HmdpState,
) -> Command[Literal["daily_chat", "grounding"]]:
    """第一层入口：先做意图判断再转发，不管前端传的是哪种模式。"""
    record_node("entry_router")
    query = (state.get("query") or "").strip()
    mode_hint = state.get("mode_hint") or "recommend"

    plan = _fast_plan(query, mode_hint)
    if plan is None:
        plan = await _llm_plan(state, query, mode_hint)
    if plan.scene == "recommend" and not plan.rewritten_query:
        plan.rewritten_query = query

    logger.info(
        "入口路由: scene=%s intents=%s names=%s type=%s",
        plan.scene, plan.sub_intents, plan.shop_names, plan.shop_type,
    )
    goto: Literal["daily_chat", "grounding"] = "daily_chat" if plan.scene == "daily" else "grounding"
    return Command(update={"plan": plan.model_dump()}, goto=goto)


# ===========================================================================
# 第二层：日常闲聊
# ===========================================================================
async def daily_chat(state: HmdpState) -> dict:
    """日常闲聊：轻量模型 + 最近历史，流式直出后直接 END。"""
    record_node("daily_chat")
    messages = [
        SystemMessage(daily_system()),
        *_summary_block(state),
        *_chat_messages(state.get("messages"), _history_keep() * 2),
    ]
    parts: list[str] = []
    last = None
    try:
        record_model_call("daily_chat")
        async for chunk in get_light_model().astream(messages):
            last = chunk
            parts.append(as_text(getattr(chunk, "content", "")))
    except Exception as exc:
        logger.error("日常闲聊生成失败: %s", exc)
        answer = "抱歉，我这边出了点小状况，能再说一次吗？"
        return {"answer": answer, "sources": [], "messages": [AIMessage(content=answer)]}

    record_usage_from_message(last)
    answer = "".join(parts).strip() or "嗯，我在这儿，有什么想聊的？"
    return {"answer": answer, "sources": [], "messages": [AIMessage(content=answer)]}


# ===========================================================================
# 第二层：Grounding——把用户的表述解析成 shop_ids
# ===========================================================================
def _ordinal_index(query: str) -> int | None:
    match = re.search(r"第([一二两三四五1-5])家", query)
    if match:
        return _ORDINAL.get(match.group(1))
    if "最后一家" in query or "最后那家" in query:
        return -1
    return None


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

    # 1) 指代消解：这家 / 第二家 / 这两家 —— 依赖跨轮焦点，这是 focus_shops 必须入 State 的原因
    if not names and _ANAPHORA.search(query) and focus:
        if _PLURAL_ANAPHORA.search(query):
            limit = 2 if re.search(r"两[家个]", query) else MAX_SHOPS
            ids.extend(int(s["id"]) for s in focus[:limit] if s.get("id") is not None)
        else:
            idx = _ordinal_index(query)
            if idx is None:
                pick = focus[0]
            elif idx == -1:
                pick = focus[-1]
            else:
                pick = focus[idx] if idx < len(focus) else focus[0]
            ids.append(int(pick["id"]))

    # 2) query 里直接出现了焦点店铺名
    if not ids and focus:
        for shop in focus:
            name = str(shop.get("name") or "")
            if name and name in query:
                ids.append(int(shop["id"]))
                break

    # 3) 按店名检索（支持一次提到多家店）
    if not ids and names:
        for name in names[:MAX_SHOPS]:
            item = await asyncio.to_thread(fetch_shops_by_name, name, 1, user_id)
            if isinstance(item, list) and item:
                ids.append(int(item[0]["id"]))
                continue
            # 整串匹配失败时，去掉「餐厅/火锅/店」等通用品类后缀重试。
            # 后端 /shop/of/name 按店名子串匹配，而「新白鹿餐厅」的店名实为「新白鹿(...)」。
            norm = _normalize_shop_name(name)
            if norm and norm != name:
                item2 = await asyncio.to_thread(fetch_shops_by_name, norm, 1, user_id)
                if isinstance(item2, list) and item2:
                    logger.info("店名规范化重试命中: %r -> %r (id=%s)", name, norm, item2[0].get("id"))
                    ids.append(int(item2[0]["id"]))

    # 去重保序
    ordered: list[int] = []
    seen: set[int] = set()
    for sid in ids:
        if sid not in seen:
            seen.add(sid)
            ordered.append(sid)

    logger.info("Grounding 解析到 shop_ids=%s（焦点 %s 家）", ordered, len(focus))
    return {"shop_ids": ordered}


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
    """去掉店名里的通用品类/后缀词，便于后端按子串匹配。

    例：「新白鹿餐厅」→「新白鹿」、「海底捞火锅」→「海底捞」、「星巴克咖啡」→「星巴克」。
    全部剥掉后为空则原样返回（让调用方仍可尝试原串）。
    """
    s = (name or "").strip()
    if not s:
        return s
    for w in _SHOP_NAME_NOISE:
        if w and w in s:
            s = s.replace(w, "")
    s = s.strip()
    return s if s else (name or "").strip()


# ===========================================================================
# 第三层：Worker（定位 + 扇出取数）
# ===========================================================================
# 后端 /shop/of/type 对部分类型（如"美食"）返回空，而 /shop/of/name 按名称能命中。
# 这里把"规范类型名"映射到用户口语化的名称搜关键字，作为回退候选。
_TYPE_NAME_KEYWORDS: dict[str, list[str]] = {
    "美食": ["餐厅", "美食", "饭店", "餐馆"],
    "火锅": ["火锅"],
    "KTV": ["KTV"],
    "咖啡": ["咖啡"],
    "酒吧": ["酒吧"],
    "奶茶": ["奶茶"],
}


async def nearby_worker(state: HmdpState) -> dict:
    """没有具体店铺时，先按类型定位，产出候选 shop_ids 交给后续扇出。"""
    record_node("nearby_worker")
    plan = plan_of(state)
    query = state.get("query") or ""
    user_id = state.get("user_id") or ""
    x, y = state.get("x"), state.get("y")

    type_name = (plan.shop_type if plan else "") or ""
    if not type_name:
        type_name = await asyncio.to_thread(_guess_shop_type, query, user_id)
    if not type_name:
        return {
            "shop_ids": [],
            "worker_outputs": [
                {
                    "worker": "nearby",
                    "error": "没能判断你想找什么类型的店铺，能补充一下吗？比如火锅、咖啡、KTV。",
                }
            ],
        }

    shops = await asyncio.to_thread(fetch_shops_nearby, type_name, x, y, 1, user_id)
    fallback_kw = None
    if isinstance(shops, str) or not shops:
        # 按类型搜不到（类型名不存在，如"火锅"；或该类型在后端无数据）时，
        # 退化为按名称关键字搜——实测 /shop/of/name 对"火锅/KTV/餐厅"等都能命中，
        # 而 /shop/of/type 对部分类型（美食/KTV）在后端返回空。该回退不触碰 Java 代码。
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
        elif isinstance(shops, str):
            return {"shop_ids": [], "worker_outputs": [{"worker": "nearby", "error": shops}]}
        else:
            return {
                "shop_ids": [],
                "worker_outputs": [
                    {"worker": "nearby", "type_name": type_name, "shops": [], "empty": f"附近暂时没有「{type_name}」相关的店铺"}
                ],
            }

    picked = shops[:NEARBY_TOP]
    focus = [
        {
            "id": int(s.get("id")),
            "name": s.get("name"),
            "distance": s.get("distance"),
            "avg_price": s.get("avgPrice"),
            "score": s.get("score"),
        }
        for s in picked
        if s.get("id") is not None
    ]
    logger.info("附近找店：类型=%s 命中 %s 家", type_name, len(picked))
    return {
        "shop_ids": [int(s["id"]) for s in picked if s.get("id") is not None],
        "focus_shops": focus,
        "worker_outputs": [
            {
                "worker": "nearby",
                "type_name": type_name,
                "shops": [
                    {
                        "id": s.get("id"),
                        "name": s.get("name"),
                        "area": s.get("area"),
                        "address": s.get("address"),
                        "distance": s.get("distance"),
                        "avgPrice": s.get("avgPrice"),
                        "score": s.get("score"),
                        "sold": s.get("sold"),
                    }
                    for s in picked
                ],
            }
        ],
    }


async def _one_shop_bundle(shop_id: int, query: str, user_id: str, with_rag: bool) -> dict:
    """单店的详情 + 攻略（+ 可选细节检索），用于详情和对比。"""
    detail, blogs = await asyncio.gather(
        asyncio.to_thread(fetch_shop_detail, shop_id, user_id),
        asyncio.to_thread(fetch_shop_blogs, shop_id, 1, user_id),
        return_exceptions=True,
    )
    bundle: dict[str, Any] = {"shop_id": shop_id}
    bundle["shop"] = detail if not isinstance(detail, BaseException) else f"店铺 {shop_id} 详情暂时不可用"
    blog_list = [] if isinstance(blogs, BaseException) else blogs
    bundle["blogs"] = blog_list if not isinstance(blog_list, str) else []
    bundle["blogs_error"] = blog_list if isinstance(blog_list, str) else ""
    if with_rag and query:
        chunks = await asyncio.to_thread(fetch_rag_chunks, shop_id, query, RAG_TOP_K)
        bundle["chunks"] = [] if isinstance(chunks, str) else chunks
        bundle["chunks_error"] = chunks if isinstance(chunks, str) else ""
    return bundle


async def shop_worker(state: HmdpState) -> dict:
    """扇出 worker：按 task 只做一类取数，返回紧凑 JSON，不生成自然语言。

    Send 的粒度是「任务 × 店铺」，同一节点多实例并行，全部完成后汇总节点只执行一次。
    """
    task = state.get("task") or "detail"
    shop_ids = [int(i) for i in (state.get("worker_shop_ids") or []) if i is not None]
    query = state.get("worker_query") or state.get("query") or ""
    user_id = state.get("user_id") or ""

    if not shop_ids:
        return {}

    record_node(f"shop_worker:{task}")
    output: dict[str, Any] = {"worker": task, "shop_ids": shop_ids, "query": query}

    try:
        if task == "detail":
            output["items"] = [await _one_shop_bundle(shop_ids[0], "", user_id, with_rag=False)]

        elif task == "voucher":
            vouchers = await asyncio.gather(
                *[asyncio.to_thread(fetch_shop_vouchers, sid, user_id) for sid in shop_ids],
                return_exceptions=True,
            )
            output["items"] = [
                {
                    "shop_id": sid,
                    "vouchers": [] if isinstance(res, (BaseException, str)) else res,
                    "error": res if isinstance(res, str) else "",
                }
                for sid, res in zip(shop_ids, vouchers)
            ]

        elif task == "rag":
            chunks = await asyncio.gather(
                *[asyncio.to_thread(fetch_rag_chunks, sid, query, RAG_TOP_K) for sid in shop_ids],
                return_exceptions=True,
            )
            output["items"] = [
                {
                    "shop_id": sid,
                    "chunks": [] if isinstance(res, (BaseException, str)) else res,
                    "error": res if isinstance(res, str) else "",
                }
                for sid, res in zip(shop_ids, chunks)
            ]

        elif task == "compare":
            output["items"] = list(
                await asyncio.gather(
                    *[_one_shop_bundle(sid, query, user_id, with_rag=True) for sid in shop_ids]
                )
            )

        else:
            logger.warning("未知 worker 任务: %s", task)
            return {}
    except Exception as exc:  # 单个 worker 失败不能拖垮整条链路
        logger.error("shop_worker(%s) 执行失败: %s", task, exc)
        output["items"] = []
        output["error"] = str(exc)

    return {"worker_outputs": [output]}


# ===========================================================================
# 第三层：路由——条件边做主骨架，Send 做局部优化
# ===========================================================================
def _fanout_args(state: HmdpState, **extra: Any) -> dict[str, Any]:
    """构造 Send 的 payload。

    LangGraph 的 Send 只把 arg 作为 worker 的输入 State，父 State 的字段不会透传，
    所以 worker 需要的一切（用户、坐标、原始问题、路由结果）都必须显式带上。
    """
    args: dict[str, Any] = {
        "query": state.get("query") or "",
        "user_id": state.get("user_id") or "",
        "x": state.get("x"),
        "y": state.get("y"),
        "plan": state.get("plan"),
    }
    args.update(extra)
    return args


def dispatch_workers(state: HmdpState) -> list[Send] | str:
    """按「任务 × 店铺」扇出。单店/多店由 len(shop_ids) 决定，不需要额外一次模型判断。"""
    plan = plan_of(state)
    shop_ids = [int(i) for i in (state.get("shop_ids") or []) if i is not None][:MAX_SHOPS]
    if not shop_ids:
        return "summarize"

    intents = set(plan.sub_intents) if plan else set()
    query = (plan.rewritten_query if plan and plan.rewritten_query else state.get("query") or "")

    tasks: list[Send] = []
    if len(shop_ids) > 1:
        # 多店：必然走对比
        tasks.append(
            Send("shop_worker", _fanout_args(state, task="compare", worker_shop_ids=shop_ids, worker_query=query))
        )
        if "voucher" in intents:
            tasks.append(
                Send("shop_worker", _fanout_args(state, task="voucher", worker_shop_ids=shop_ids, worker_query=query))
            )
    else:
        if not intents:
            intents = {"detail"}
        # 单店：按子意图按需取数，只问优惠就只查优惠
        if "detail" in intents or "nearby" in intents:
            tasks.append(
                Send("shop_worker", _fanout_args(state, task="detail", worker_shop_ids=shop_ids, worker_query=query))
            )
        if "voucher" in intents:
            tasks.append(
                Send("shop_worker", _fanout_args(state, task="voucher", worker_shop_ids=shop_ids, worker_query=query))
            )
        if "rag" in intents:
            tasks.append(
                Send("shop_worker", _fanout_args(state, task="rag", worker_shop_ids=shop_ids, worker_query=query))
            )

    if not tasks:
        return "summarize"
    logger.info("扇出 %s 个 worker: %s", len(tasks), [t.arg.get("task") for t in tasks])
    return tasks


def route_after_grounding(state: HmdpState) -> list[Send] | str:
    """Grounding 之后的去向：缺实体就先定位，否则直接扇出。"""
    plan = plan_of(state)
    if state.get("shop_ids"):
        return dispatch_workers(state)
    if plan and plan.need_clarify:
        return "clarify"
    if plan and (plan.shop_type or "nearby" in plan.sub_intents):
        query = plan.rewritten_query or state.get("query") or ""
        return [Send("nearby_worker", _fanout_args(state, worker_query=query))]
    return "clarify"


def route_after_nearby(state: HmdpState) -> list[Send] | str:
    """定位之后：只有用户还问了详情/优惠/细节/对比，才继续扇出。

    用户只是问"附近有什么火锅店"时，候选列表已经足够，不再逐家查详情，省掉一轮取数。
    """
    if not state.get("shop_ids"):
        return "summarize"
    plan = plan_of(state)
    intents = set(plan.sub_intents) if plan else set()
    if not intents or intents <= {"nearby"}:
        return "summarize"
    return dispatch_workers(state)


# ===========================================================================
# 第四层：汇总
# ===========================================================================
def _collect_sources(outputs: list[dict]) -> list[str]:
    """来源由代码从 worker 取回的真实数据里提取，不靠模型回忆，避免编造标题。"""
    titles: list[str] = []
    for item in outputs or []:
        for bundle in item.get("items") or []:
            for chunk in bundle.get("chunks") or []:
                title = str(chunk.get("title") or "").strip()
                if title and title not in titles:
                    titles.append(title)
            for blog in (bundle.get("blogs") or [])[:CONTEXT_BLOGS]:
                title = str(blog.get("title") or "").strip()
                if title and title not in titles:
                    titles.append(title)
    return titles[:MAX_SOURCE]


def _collect_focus(outputs: list[dict], fallback: list[dict]) -> list[dict]:
    """把本轮实际用到的店铺沉淀为下一轮的焦点，供"这家/第二家"解析。"""
    focus: list[dict] = []
    seen: set[int] = set()
    for item in outputs or []:
        if item.get("worker") == "nearby":
            for shop in item.get("shops") or []:
                sid = shop.get("id")
                if sid is not None and int(sid) not in seen:
                    seen.add(int(sid))
                    focus.append(
                        {
                            "id": int(sid),
                            "name": shop.get("name"),
                            "distance": shop.get("distance"),
                            "avg_price": shop.get("avgPrice"),
                            "score": shop.get("score"),
                        }
                    )
    if focus:
        return focus[:NEARBY_TOP]
    return list(fallback or [])


def render_context(outputs: list[dict], focus: list[dict]) -> str:
    """把 worker 的紧凑结果渲染成给汇总模型的文本，并在这一步完成裁剪。"""
    if not outputs:
        return "（没有查到任何资料）"

    blocks: list[str] = []
    for item in outputs or []:
        worker = item.get("worker")

        if worker == "nearby":
            if item.get("error"):
                blocks.append(f"【附近找店】{item['error']}")
                continue
            shops = item.get("shops") or []
            if not shops:
                blocks.append(f"【附近找店】{item.get('empty') or '没有找到相关店铺'}")
                continue
            lines = [f"【附近找店 · {item.get('type_name') or ''}】"]
            for idx, shop in enumerate(shops, 1):
                distance = f"{shop.get('distance')}m" if shop.get("distance") is not None else "距离暂未获取"
                score = shop.get("score")
                score_text = f"{score / 10:.1f}分" if isinstance(score, (int, float)) and score else "暂无评分"
                avg = f"人均{shop.get('avgPrice')}元" if shop.get("avgPrice") else "人均暂未获取"
                lines.append(
                    f"{idx}. {shop.get('name')}（id={shop.get('id')}） {distance} {score_text} {avg} "
                    f"销量{shop.get('sold') or 0} 地址：{shop.get('address') or '暂未获取'}"
                )
            blocks.append("\n".join(lines))
            continue

        for bundle in item.get("items") or []:
            shop_id = bundle.get("shop_id")
            name = ""
            if isinstance(bundle.get("shop"), dict):
                shop = bundle["shop"]
                name = shop.get("name") or f"店铺{shop_id}"
                open_hours = shop.get("openHours") or "营业时间暂未获取"
                distance = f"{shop.get('distance')}m" if shop.get("distance") is not None else "距离暂未获取"
                score = shop.get("score")
                score_text = f"{score / 10:.1f}分" if isinstance(score, (int, float)) and score else "暂无评分"
                lines = [
                    f"【店铺{shop_id} · {name}】",
                    f"地址：{shop.get('address') or '暂未获取'}",
                    f"人均：{shop.get('avgPrice') or '暂未获取'}元；评分：{score_text}；销量：{shop.get('sold') or 0}",
                    f"营业时间：{open_hours}；距离：{distance}",
                ]
                if bundle.get("blogs_error"):
                    lines.append(f"攻略：{bundle['blogs_error']}")
                for blog in (bundle.get("blogs") or [])[:CONTEXT_BLOGS]:
                    lines.append(f"攻略《{blog.get('title')}》：{(blog.get('content') or '')[:100]}")
                if bundle.get("chunks_error"):
                    lines.append(f"细节：{bundle['chunks_error']}")
                for chunk in bundle.get("chunks") or []:
                    lines.append(f"细节《{chunk.get('title')}》：{(chunk.get('content') or '')[:200]}")
                blocks.append("\n".join(lines))
            else:
                lines = [f"【店铺{shop_id}】{name or ''}".rstrip()]
                if bundle.get("error"):
                    lines.append(f"查询失败：{bundle['error']}")
                for voucher in bundle.get("vouchers") or []:
                    lines.append(
                        f"优惠券《{voucher.get('title')}》 售价{voucher.get('payValue')}元 "
                        f"抵{voucher.get('actualValue')}元 库存{voucher.get('stock')} "
                        f"规则：{voucher.get('rules') or '无'}"
                    )
                if bundle.get("chunks_error"):
                    lines.append(f"细节：{bundle['chunks_error']}")
                for chunk in bundle.get("chunks") or []:
                    lines.append(f"细节《{chunk.get('title')}》：{(chunk.get('content') or '')[:200]}")
                blocks.append("\n".join(lines))

    if focus and not any("id=" in b for b in blocks):
        blocks.append(
            "【本轮涉及的店铺】" + "；".join(f"{s.get('name')}(id={s.get('id')})" for s in focus)
        )
    return "\n\n".join(blocks)


async def _summarize_stream(messages: list, sources: list) -> str:
    """流式生成最终回答；结构化失败时的兜底，也是产出自然语言的主路径。"""
    parts: list[str] = []
    last = None
    record_model_call("summarize")
    async for chunk in get_main_model().astream(messages):
        last = chunk
        parts.append(as_text(getattr(chunk, "content", "")))
    record_usage_from_message(last)
    raw = "".join(parts).strip()
    parsed_answer, parsed_sources = _parse_possible_json(raw)
    answer = parsed_answer or raw
    if parsed_sources:
        sources[:] = parsed_sources[:MAX_SOURCE]
    return answer


async def summarize(state: HmdpState) -> dict:
    """汇总：整条链路里唯一一次主模型调用。"""
    record_node("summarize")
    query = state.get("query") or ""
    outputs = list(state.get("worker_outputs") or [])
    focus = list(state.get("focus_shops") or [])
    sources = _collect_sources(outputs)
    context = render_context(outputs, focus)

    messages = [
        SystemMessage(summary_system()),
        *_summary_block(state),
        *_history_before_current(state.get("messages"), turns=1),
        HumanMessage(f"用户问题：{query}\n\n已查到的资料：\n{context}"),
    ]

    answer = ""
    try:
        if SUMMARY_MODE == "structured":
            # 强约束模式：用 with_structured_output 直接拿到 answer + source
            record_model_call("summarize")
            result = await get_main_model().with_structured_output(RagAnswer).ainvoke(messages)
            answer = (getattr(result, "answer", "") or "").strip()
            if getattr(result, "source", None):
                sources = [str(s) for s in result.source if s][:MAX_SOURCE]
            # 结构化偶发返回空（模型吐空 / 流式超时截断）：context 有资料时回退流式再试，
            # 避免"明明查到了却回答没查到"的静默失败。
            if not answer and context and context != "（没有查到任何资料）":
                answer = await _summarize_stream(messages, sources)
        else:
            answer = await _summarize_stream(messages, sources)
    except Exception as exc:
        logger.error("汇总生成失败: %s", exc)
        answer = "抱歉，整理回答时出了点问题，请稍后再试。"

    if not answer:
        answer = "根据现有博客暂未查到"

    logger.info("汇总完成，answer 长度 %s，来源 %s 条", len(answer), len(sources))
    return {
        "answer": answer,
        "sources": sources,
        "messages": [AIMessage(content=answer)],
        "focus_shops": _collect_focus(outputs, focus),
        # 用完即清：中间结果绝不能被 checkpoint 带进下一轮
        "worker_outputs": CLEAR,
        "shop_ids": [],
        "task": "",
        "worker_shop_ids": [],
        "worker_query": "",
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
    plan = plan_of(state)
    answer = ""
    if plan and plan.clarify_question:
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
