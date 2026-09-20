"""LangGraph 编排的全局最小状态集。

设计原则：
- 下游节点要用来做决策的，才入 State
- 能从其它字段推导出来的，不存（例如"单店/多店"由 len(shop_ids) 推导）
- 用完即弃的中间结果，必须能从 State 里清掉，否则会被 checkpoint 带进下一轮
"""

import os
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Scene = Literal["daily", "recommend"]
SubIntent = Literal["nearby", "detail", "voucher", "rag", "compare"]
WorkerTask = Literal["detail", "voucher", "rag", "compare"]

_VALID_SUBINTENTS = {"nearby", "detail", "voucher", "rag", "compare"}


class RoutePlan(BaseModel):
    """入口节点结构化输出：意图识别 + 实体抽取一次调用全部完成。

    注意：大模型在 JSON 模式下对“无值”字段常返回 null，
    这里在 model_validator / field_validator 里统一兜底为空值，
    避免单个字段为 null 导致整条校验失败、意图识别被迫走兜底分支。
    """

    model_config = ConfigDict(extra="ignore")

    scene: Scene = Field(
        description="daily=日常闲聊/问候/与找店无关的话题；recommend=找店、店铺攻略、优惠、店铺细节、多店对比"
    )
    sub_intents: list[SubIntent] = Field(
        default_factory=list,
        description="仅在 scene=recommend 时填写，可多选：nearby=找某类型的店；detail=店铺详情/攻略；voucher=优惠券/团购；rag=价格/停车/包间/服务等细节；compare=两家及以上对比",
    )
    shop_names: list[str] = Field(
        default_factory=list, description="用户明确提到的店铺名称，没有则留空"
    )
    shop_type: str = Field(
        default="", description="店铺类型关键词，如火锅、咖啡、KTV；没有则留空"
    )
    need_clarify: bool = Field(
        default=False, description="用户用了这家/那家/第二家等指代且无法判断具体店铺时为 true"
    )
    clarify_question: str = Field(
        default="", description="need_clarify 为 true 时，给用户的追问内容"
    )
    rewritten_query: str = Field(
        default="", description="去掉口语化表达后、用于检索的 query"
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_nulls(cls, data: Any) -> Any:
        """模型常把“无值”字段输出为 null，这里兜底成空值，避免整条校验失败。"""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for k in ("sub_intents", "shop_names"):
            if out.get(k) is None:
                out[k] = []
        for k in ("shop_type", "clarify_question", "rewritten_query"):
            if out.get(k) is None:
                out[k] = ""
        return out

    @field_validator("sub_intents", mode="before")
    @classmethod
    def _clean_sub_intents(cls, v: Any) -> list:
        """只保留合法意图，丢弃模型可能拼错的变体（如 near_by）。"""
        if not isinstance(v, list):
            return []
        return [x for x in v if x in _VALID_SUBINTENTS]


# 清空信号。必须是可序列化的字符串：checkpoint 会把每次写入的值落库（msgpack），
# 自定义对象会直接抛 TypeError，所以这里不能用 sentinel object。
CLEAR = "__CLEAR__"

# 跨轮焦点的最大保留家数。焦点是累积的（见 merge_focus），必须设上限，
# 否则长会话（几十轮）会把焦点列表撑成几十家，既拖慢指代解析、也让兜底渲染变长。
FOCUS_MAX = int(os.getenv("AGENT_FOCUS_MAX", "10"))


def add_and_clear(left: Any, right: Any) -> list[Any]:
    """累积 reducer，遇到 CLEAR 信号时清空。

    worker 结果如果不清掉，会被 checkpoint 带进下一轮，几轮之后就变成巨大的上下文。
    """
    if isinstance(right, str) and right == CLEAR:
        return []
    if not right:
        return list(left) if isinstance(left, list) else []
    base = list(left) if isinstance(left, list) else []
    return base + list(right)


def merge_focus(left: Any, right: Any) -> list[dict]:
    """focus_shops 的累积 reducer：按 id 去重后追加，保持出现顺序，超出上限丢最早的。

    为什么必须累积而不能覆盖（原实现是直接覆盖）：
    用户先分别问「A 店怎么样 / B 店怎么样 / C 店怎么样」，再问「这三家对比如何」时，
    如果焦点每轮整体覆盖，后一轮会把前几轮的店冲掉；而 A/B/C 分别查询时又走不到
    nearby 分支（原来只有 nearby 结果才沉淀焦点），于是最后一句指代解析拿不到任何实体，
    表现为答非所问地反问「你想了解哪家店铺呢」。

    累积后焦点列表的顺序 = 店铺被提及的顺序，因此「第一/第二/第三家」的索引语义
    反而更符合直觉（依次问 A、B、C 后说「第三家」＝ C）。
    """
    base = list(left) if isinstance(left, list) else []
    if not right or (isinstance(right, str) and right == CLEAR):
        return base
    merged: list[dict] = list(base)
    seen: set[int] = {
        int(s["id"]) for s in merged if isinstance(s, dict) and s.get("id") is not None
    }
    for shop in right:
        if not isinstance(shop, dict):
            continue
        sid = shop.get("id")
        if sid is None or int(sid) in seen:
            continue
        seen.add(int(sid))
        merged.append(shop)
    return merged[-FOCUS_MAX:]


class HmdpState(TypedDict):
    # ---- 每轮输入（覆盖写入）----
    query: str
    mode_hint: str
    user_id: str
    x: float | None
    y: float | None

    # ---- 第一层：路由结果 ----
    plan: dict | None

    # ---- 第二层：跨轮焦点，用于解析"这家/第二家/这三家" ----
    # 累积写入（见 merge_focus）：分别问过 A/B/C 后问"这三家对比"，焦点里三家都在。
    focus_shops: Annotated[list[dict], merge_focus]
    shop_ids: list[int]

    # ---- 第三层：worker 扇出参数（由 Send 注入）----
    task: str
    worker_shop_ids: list[int]
    worker_query: str

    # ---- 第三层：worker 产出（短生命周期，汇总后清空）----
    worker_outputs: Annotated[list[dict], add_and_clear]

    # ---- 跨轮压缩摘要：被移除的早期消息由便宜模型总结后存这里 ----
    history_summary: str

    # ---- 第四层：最终产物 ----
    messages: Annotated[list[AnyMessage], add_messages]
    answer: str
    sources: list[str]


def plan_of(state: Any) -> RoutePlan | None:
    """把 State 里的 plan 还原成 RoutePlan，解析失败返回 None。"""
    raw = state.get("plan") if isinstance(state, dict) else None
    if not raw:
        return None
    if isinstance(raw, RoutePlan):
        return raw
    try:
        return RoutePlan.model_validate(raw)
    except Exception:
        return None
