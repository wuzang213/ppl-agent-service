"""LangGraph 编排的全局最小状态集：只有下游要用来决策的才入 State，可推导的不存
（如"单店/多店"由 len(shop_ids) 推导）；用完即弃的中间结果必须能清掉，否则会被 checkpoint 带进下一轮。"""

import os
import re
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: `capability`=产品能力咨询，由 `entry_router` 确定性正则短路（不调模型）从 daily 摘出（见 bug/BUG-20261003）。
#: `plan`=多活动/多地点且需排先后的行程规划，是独立轨（specs/0012）非更复杂的推荐；判据在 prompts.ENTRY_SYSTEM，代码侧 nodes._apply_plan_guard 二次校验（类别<2 回落 recommend），由 AGENT_PLAN_ENABLED（默认关）控制。
Scene = Literal["daily", "recommend", "capability", "plan"]
SubIntent = Literal["nearby", "detail", "voucher", "rag", "compare"]
# 这里不该再有 `WorkerTask`：可执行取数任务只有 `detail` / `voucher` / `rag`（权威定义见
#    `nodes.SHOP_RECORD_TASKS`，引导意图见 `nodes.BOOTSTRAP_INTENTS`）；`compare` 自 `plans/0009.2`
#    起已不是 worker 任务（仍是 `SubIntent`）。需要任务集合请直接引用 `nodes` 的常量。
DependsOn = Literal["price", "score", "distance", "sold"]

_VALID_SUBINTENTS = {"nearby", "detail", "voucher", "rag", "compare"}
_VALID_DEPENDS_ON = {"price", "score", "distance", "sold"}
#: 软偏好（排序）维度的合法取值。与 `depends_on` **共用同一组维度**，但语义不同：
#: `depends_on` 是"目标定位维度"（唯一、决定"是哪一家"），`preferences` 是"排序偏好"（有序、可多条）。
#: 二者**不强制一致**（`specs/0024` §8 Q2）。
_VALID_PREFERENCES = {"price", "score", "distance", "sold"}


class PreferenceItem(BaseModel):
    """软偏好（specs/0024）排序维度：优先级=在 preferences 列表里的先后（越前越重要）；刻意不用数值权重（量纲不同难解释难测），推荐轨按字典序逐维比较（plans/0043），plan 轨按序声明。"""

    model_config = ConfigDict(extra="ignore")

    dimension: str = Field(
        default="",
        description="排序维度：price=人均最低 / score=评分最高 / distance=距离最近 / sold=销量最高",
    )


class RoutePlan(BaseModel):
    """入口节点结构化输出：意图识别 + 实体抽取一次调用完成。模型在 JSON 模式下对“无值”字段
    常返回 null，故在 model_validator / field_validator 里统一兜底为空值，避免整条校验失败。"""

    model_config = ConfigDict(extra="ignore")

    scene: Scene = Field(
        description="daily=日常闲聊/问候/与找店无关的话题；"
        "recommend=找店、店铺攻略、优惠、店铺细节、多店对比；"
        "plan=一段时间内的多个活动/多个地点且需要排先后的行程规划"
        "（如「下午在西湖一带，先下午茶再晚餐」）。"
        "⚠️ 单品类找店、单店详情/单店券、两家比价**一律不算 plan**。"
    )
    sub_intents: list[SubIntent] = Field(
        default_factory=list,
        description="仅在 scene=recommend 时填写，可多选：nearby=找某类型的店；detail=店铺详情/攻略；voucher=代金券/团购（含用户口语的「优惠券」同义）；rag=价格/停车/包间/服务等细节；compare=两家及以上对比",
    )
    shop_names: list[str] = Field(
        default_factory=list, description="用户明确提到的店铺名称，没有则留空"
    )
    shop_type: str = Field(
        default="", description="店铺类型关键词，如火锅、咖啡、KTV；没有则留空"
    )
    shop_type_raw: str = Field(
        default="",
        description="用户原话里的**具体品类原词**（如 日料/火锅/手冲咖啡），**不归类、原样保留**；"
        "没有具体词就留空。"
        "⚠️ 与 `shop_type`（归类后的真实类型）**使命不同**："
        "原词只用于「细分品类筛选」与「博客反查检索词」，"
        "归类词仍作类型搜与既有错误/为空文案的载体（`specs/0028`）。",
    )
    shop_types: list[str] = Field(
        default_factory=list,
        description="**多活动/多类型**的归类结果（`specs/0030`）：把用户**并列提到的每个活动**各列一项"
        "（归类后的真实类型名），按原话顺序；**同一类型下的不同细分说法也算不同活动**"
        "（如「火锅和咖啡」⇒ `[\"美食\",\"美食\"]`，靠 `shop_terms_raw` 区分）。"
        "去重按 `(类型, 原词)` 组合在 `_resolve_type_targets` 里做；只有一个活动时也只放一项。"
        "`shop_type` 恒等于它的**首项**（列表为空时退回单值字段）。",
    )
    shop_terms_raw: list[str] = Field(
        default_factory=list,
        description="与 `shop_types` **按下标一一对应**的用户原词（缺位补空串，**保位置不压缩**）；"
        "`shop_type_raw` 恒等于它的**首项**。",
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
    target_depends_on_result: bool = Field(
        default=False,
        description="用户要的目标，是否必须先看完一批候选数据才能确定是谁。"
        "例：'最便宜那家的券'→true（先比价才知道是哪家）；'海底捞有什么券'→false（店已确定）。"
        "⚠️ 问的是目标本身是否依赖中间结果，不是问题复不复杂。",
    )
    depends_on: DependsOn | None = Field(
        default=None,
        description="target_depends_on_result=true 时，依赖的是哪个维度："
        "price=价格最低；score=评分最高；distance=距离最近；sold=销量最高。"
        "为 false 时留空。",
    )
    preferences: list[PreferenceItem] = Field(
        default_factory=list,
        description="用户的**软偏好**排序维度，按优先级从高到低排列（第一条最重要）。"
        "只在用户**显式表达**偏序时填写，例如「便宜点、近一点」→[price, distance]；"
        "没有偏好就留空数组。"
        "⚠️ 与 `depends_on` **独立、不强制一致**：`depends_on` 回答'是哪一家'，"
        "`preferences` 只是'排序倾向'；两者可同时出现，也可不同。",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_nulls(cls, data: Any) -> Any:
        """模型常把“无值”字段输出为 null，这里兜底成空值，避免整条校验失败。"""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for k in ("sub_intents", "shop_names", "shop_types", "shop_terms_raw"):
            if out.get(k) is None:
                out[k] = []
        for k in ("shop_type", "shop_type_raw", "clarify_question", "rewritten_query"):
            if out.get(k) is None:
                out[k] = ""
        # target_depends_on_result：模型可能回 null / 空串，统一兜底为 False。
        # 用 `not out.get(k)` 覆盖 null / "" / 0 三种"无值"形态；
        # 注意不能把 False 本身写成 True，所以先判"键存在但值假"再赋 False。
        if not out.get("target_depends_on_result"):
            out["target_depends_on_result"] = False
        # depends_on：无值（null / 空串）统一成 None。
        if not out.get("depends_on"):
            out["depends_on"] = None
        return out

    @field_validator("sub_intents", mode="before")
    @classmethod
    def _clean_sub_intents(cls, v: Any) -> list:
        """只保留合法意图，丢弃模型可能拼错的变体（如 near_by）。"""
        if not isinstance(v, list):
            return []
        return [x for x in v if x in _VALID_SUBINTENTS]

    @field_validator("depends_on", mode="before")
    @classmethod
    def _clean_depends_on(cls, v: Any) -> str | None:
        """只保留合法维度，丢弃模型可能拼错的变体（如 rating / 评论）；与 `_clean_sub_intents`
        同策略 —— 丢弃而非报错，一个脏字段不该让整条意图识别失败。"""
        if v is None:
            return None
        if not isinstance(v, str):
            return None
        v = v.strip().lower()
        return v if v in _VALID_DEPENDS_ON else None

    @field_validator("preferences", mode="before")
    @classmethod
    def _clean_preferences(cls, v: Any) -> list:
        """软偏好列表归一化 + 只留合法维度（去重保序），非法丢弃不报错；兼容裸字符串/对象数组/脏值，统一成 [{"dimension":dim}]（同 _clean_depends_on：一个脏字段不作废整条）。"""
        if not isinstance(v, list):
            return []
        out: list[dict] = []
        seen: set[str] = set()
        for item in v:
            raw = item.get("dimension") if isinstance(item, dict) else item
            dim = str(raw or "").strip().lower()
            if dim in _VALID_PREFERENCES and dim not in seen:
                seen.add(dim)
                out.append({"dimension": dim})
        return out

    @field_validator("shop_types", mode="before")
    @classmethod
    def _clean_shop_types(cls, v: Any) -> list:
        """多类型列表：非 list/None⇒[]；逐项 strip、丢非串/空串、保序。刻意不去重（同类型可重复表不同活动，靠 shop_terms_raw 区分，去重按(类型,原词)在 nodes._resolve_type_targets）。容错：null/裸串/带空串数组都不抛错（否则连 shop_type/sub_intents 一起丢）；裸串不解析，丢弃后退回单值路径。"""
        if not isinstance(v, list):
            return []
        return [item.strip() for item in v if isinstance(item, str) and item.strip()]

    @field_validator("shop_terms_raw", mode="before")
    @classmethod
    def _clean_shop_terms_raw(cls, v: Any) -> list:
        """与 shop_types 按下标对齐的原词列表：保位置（不去重、不压空位），否则下标错位；与 _clean_shop_types 的差别是刻意的——shop_terms_raw=["火锅",""] 须保留第二位，否则 KTV 错配成火锅。"""
        if not isinstance(v, list):
            return []
        return [item.strip() if isinstance(item, str) else "" for item in v]

    @field_validator("shop_type_raw", mode="before")
    @classmethod
    def _clean_shop_type_raw(cls, v: Any) -> str:
        """原词容错：非字符串/None⇒空串（丢弃而非报错，一个脏字段不作废整条）；模型可能回成数组/对象，抛 ValidationError 会连 shop_type/sub_intents 一起丢（同 _clean_sub_intents 口径）。"""
        if v is None:
            return ""
        if not isinstance(v, str):
            return ""
        return v.strip()

    @model_validator(mode="after")
    @classmethod
    def _drop_contradictory_depends_on(cls, obj: "RoutePlan") -> "RoutePlan":
        """语义一致性：`depends_on` 只在 `target_depends_on_result=true` 时有意义，
        否则清成 None，避免下游拿到"不依赖但有维度"的矛盾组合。"""
        if not obj.target_depends_on_result and obj.depends_on is not None:
            object.__setattr__(obj, "depends_on", None)
        return obj


# 清空信号。必须是可序列化的字符串：checkpoint 会把每次写入的值落库（msgpack），
# 自定义对象会直接抛 TypeError，所以这里不能用 sentinel object。
CLEAR = "__CLEAR__"

# 跨轮焦点的最大保留家数。焦点是累积的（见 merge_focus），必须设上限，
# 否则长会话（几十轮）会把焦点列表撑成几十家，既拖慢指代解析、也让兜底渲染变长。
FOCUS_MAX = int(os.getenv("AGENT_FOCUS_MAX", "10"))


def add_and_clear(left: Any, right: Any) -> list[Any]:
    """累积 reducer，遇到 CLEAR 信号时清空；worker 结果不清掉会被 checkpoint 带进下一轮、越滚越大。"""
    if isinstance(right, str) and right == CLEAR:
        return []
    if not right:
        return list(left) if isinstance(left, list) else []
    base = list(left) if isinstance(left, list) else []
    return base + list(right)


def merge_focus(left: Any, right: Any) -> list[dict]:
    """focus_shops 的累积 reducer：按 id 去重后追加，保持出现顺序，超出 `FOCUS_MAX` 丢最早的。
    必须累积而非覆盖：分别问过 A/B/C 再问「这三家对比」时，覆盖会把前几轮的店冲掉，
    指代解析拿不到实体；累积后列表顺序 = 提及顺序，「第三家」的索引语义才成立。"""
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


# ===========================================================================
# 摘要投影（`specs/0016` T4）：[已推荐] / [死路] 的跨轮载体
# ===========================================================================
def _digest_max() -> int:
    """投影列表的最大保留条数（调用时读 env，同 `FOCUS_MAX` 口径；导入期常量陷阱）。"""
    try:
        return max(1, int(os.getenv("AGENT_DIGEST_MAX", "10")))
    except (TypeError, ValueError):
        return 10


def _digest_key(item: dict) -> tuple | None:
    """去重键：优先 `id`（无 id 则退回 `name`）；两者都没有 → None（丢弃）。"""
    sid = item.get("id")
    if sid is not None:
        try:
            return ("id", int(sid))
        except (TypeError, ValueError):
            pass
    name = str(item.get("name") or "").strip()
    return ("name", name) if name else None


def merge_digest(left: Any, right: Any) -> list[dict]:
    """recommended_digest/excluded_digest 的累积 reducer：按 id（无则 name）去重追加、保序，超 _digest_max() 丢最早。必须累积——[已推荐]/[死路]是会话级记忆，覆盖会让上轮推过的店消失⇒多轮重复推荐。"""
    base = [x for x in left if isinstance(x, dict)] if isinstance(left, list) else []
    if not right or (isinstance(right, str) and right == CLEAR):
        return base[-_digest_max():]
    merged = list(base)
    seen = {_digest_key(x) for x in merged}
    for item in right:
        if not isinstance(item, dict):
            continue
        key = _digest_key(item)
        if key is None or key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged[-_digest_max():]


# 计划轨 task_draft 生命周期（specs/0012 §11.2/§11.3）。
#: 「显式结束」措辞⇒立即清理；只收确定结束的说法（误清=任务丢失不可逆），"可以了/没问题"这类可能是别的答话一律不收。
_PLAN_DONE_PATTERNS: tuple[str, ...] = (
    "就这个了", "就要这个", "就这个吧", "不用改了", "不用改",
    "就这样吧", "确定了", "ok就这个",
)
#: 「强制换题」措辞 ⇒ **强制清理**（用户显式开新话题 / 重置）。
_PLAN_RESET_PATTERNS: tuple[str, ...] = (
    "换个话题", "换一个话题", "换话题", "重新来", "重新开始",
    "开始新的", "新的问题", "不聊这个",
)


def _task_ttl_turns() -> int:
    """任务草稿 TTL：连续多少轮未提及行程即自动清理（默认 3）。调用时读 env、不做导入期常量（踩过：启动后改 os.environ 不生效也不报错）；解析失败回退默认，不让坏配置打断链路。"""
    try:
        return int(os.getenv("AGENT_TASK_TTL_TURNS", "3"))
    except (TypeError, ValueError):
        return 3


def _hits(query: str, patterns: tuple[str, ...]) -> bool:
    """query 是否包含给定措辞之一（大小写不敏感）。"""
    q = (query or "").strip().lower()
    return bool(q) and any(p in q for p in patterns)


def tick_idle(draft: dict | None, hit_plan: bool) -> dict:
    """推进"连续未提及"计数并返回新 dict（纯函数，不改入参）：命中 plan 轨归零，否则 +1。每轮必须恰好调一次，否则漏记会让超时清理永不触发（task_draft 永久驻留、旧行程污染新对话）。"""
    d = dict(draft or {})
    d["idle_turns"] = 0 if hit_plan else int(d.get("idle_turns") or 0) + 1
    return d


def should_clear(draft: dict | None, query: str, ttl_turns: int | None = None) -> bool:
    """是否清掉 task_draft（specs/0012 §11.3 三种清理）：立即（确认/明确结束）、强制（显式换题/重置）、超时（连续未提及 ≥ TTL，默认 3）。边界取等号 idle_turns>=ttl；拿不准的措辞一律偏向不清（误清不可逆，多留一轮下轮再清）。"""
    if not draft:
        return False
    if _hits(query, _PLAN_DONE_PATTERNS) or _hits(query, _PLAN_RESET_PATTERNS):
        return True
    ttl = _task_ttl_turns() if ttl_turns is None else int(ttl_turns)
    return int(draft.get("idle_turns") or 0) >= ttl


class HmdpState(TypedDict):
    # ---- 每轮输入（覆盖写入）----
    query: str
    mode_hint: str
    user_id: str
    x: float | None
    y: float | None

    # ---- 第一层：路由结果 ----
    plan: dict | None

    #: 本轮入口是否走了降级兜底（入口模型不可用⇒重试后按 mode_hint 硬判）。显式存的原因：降级后 plan.shop_type 为空，nearby_worker 会答"没能判断类型"——把服务故障说成用户没说明白（2026-10-05 真机复现）；有此标记才分流到"稍后再试"。
    #: 由 entry_router 每轮显式写（降级 True 否则 False，否则陈旧值随 checkpoint 带到下轮）；未声明的键会被 LangGraph 静默丢弃，必须在此声明。
    route_degraded: bool

    # 「追问更多」复用的品类（"还有吗"不含品类词），nearby_worker 每轮写入：第 2 轮只说"还有吗"时各途径都拿不到品类，没它只能答"没能判断类型"，正确应是"再列一批上轮品类"；未声明的键会被 LangGraph 静默丢弃，必须声明。
    last_nearby_type: str

    # 计划轨跨轮任务草稿（specs/0012 §11.2），只有 plan 轨写、其余置空；结构 {task_id, slots:{time,area,people,budget,scenes}, stops, clarify_retry, idle_turns, created_at}，是多轮行程的唯一跨轮载体（focus_shops 仅会话级焦点）。刻意不加 status/suspend_node：挂起由框架 pending interrupt 表示，自存即第二份真相（specs/0012 v3 定稿）。
    task_draft: dict | None

    # 计划轨产物（specs/0012 §12.1/§12.2）。必须在此声明：LangGraph 对未声明的键静默丢弃（连 updates 流都看不到，无警告）；漏声明⇒SSE 永远发不出 itinerary/suggestions，且排查无线索。
    itinerary: dict | None
    suggestions: list[dict]

    #: recommend 轨卡片（#117/#120）：summarize_answer 抽出的店铺卡片列表（每店带 LLM 推荐理由）。同样必须声明，否则 LangGraph 静默丢弃⇒SSE 发不出 shops 事件（plans/0020 真机：后端算出 5 张卡、日志也打了，前端却收不到，根因是漏声明）。
    recommend_shops: list[dict]

    #: recommend 多方案载荷（specs/0027/plans/0044）：preferences 含≥2 维度时每维度一套 [{dimension,label,items[]}]（最多 3 套、每套≤5 家、不互斥）；与 recommend_shops 并存（shops 保留，旧前端不受影响），单方案为空列表。同样必须声明，否则前端永远收不到 schemes。
    schemes: list[dict]

    #: 预约确认草稿（plans/0019），结构 {shop_id, shop_name, address, booking_time, people, remark}。必须放 State 而非局部变量：interrupt() 挂起后节点重放、局部变量必丢，而续跑要把 action 与当初参数配对，需跨轮可读真相；同样不存 status（挂起由框架 pending interrupt 表示）。
    booking_draft: dict | None

    #: 本轮请求携带的**结构化动作**（`plans/0019`）：`{type, args}`。
    #: 写操作的唯一入口是它 —— 前端点"帮我预约"按钮时回传 `{type:"booking", args:{...}}`，
    #:    点"确认"时回传 `{type:"confirm"}`。**刻意不接受模型产出的动作**（模型不参与写决策）。
    action: dict | None

    #: 预约执行结果：`{result: success|cancelled|failed, order_id}`（`plans/0019`）。
    #: 同样**必须声明** —— 未声明的键会被 LangGraph 静默丢弃（连 `updates` 流里都没有）。
    booking_result: dict | None

    #: 「本轮是预约轮、店名靠 grounding 解析」——entry_router 置位、同轮被 route_after_grounding 消费（转 booking_confirm）。每轮必须清零（trim_history 开头无条件写 False）：它会进 checkpoint，陈旧标记会把下一轮无关问句劫持进预约确认。需要它是因为预约意图在 entry_router 判，但本轮点名店要到 grounding 才解析，不传就只能拿到上一轮的店（真机 2026-10-08 复现）。
    booking_pending: bool

    #: 预约确认卡被用户修改过几次（plans/0021 T1/specs/0012 §16.3）。必须进 State 不能放局部变量：interrupt() 续跑节点从头重放、局部变量归零⇒上限永不触发⇒前端反复发 modify 时无限挂起、用户永远结束不了。每次 modify 成功 +1；确认/取消/交回入口不重置（否则"改5次→取消→再预约"又从0数，等于没上限）。
    booking_modify_count: int

    #: 上一次 modify 被拒的可读原因（plans/0021 §16.2），空串=没有；未声明的键会被 LangGraph 静默丢弃，必须声明。用途是重入时把原因带进新挂起的卡：校验失败不 goto END（那等于收卡、用户没机会改），而是节点从头重跑再 interrupt()，重跑须读到"上次为何被拒"，否则用户只看到一样的卡、不知错在哪。
    booking_modify_error: str

    # provider usage 锚点（specs/0013 T5）：上次模型调用返回的真实 prompt_tokens（含 cached，不再另加，否则 double count）；未声明的键会被 LangGraph 静默丢弃，必须声明。用途：trim_history 用「锚点 token + 锚点后新增消息的本地估算」与预算比较，锚点失效（指纹不匹配/消息变少）即回退纯本地估算。
    last_provider_context_tokens: int | None

    #: 上面的锚点是在"当时共多少条消息"的上下文里取到的（`anchor_message_count`）。
    #: 下一轮 `trim_history` 用 `messages[anchor_message_count:]` 算出"新增了多少"，
    #: 新增部分没有真实 usage，只能本地估算 ⇒ **锚点 + 增量**，而不是拿锚点当总数。
    anchor_message_count: int | None

    #: 锚点时"上下文头部形状"指纹（前若干条非系统消息的哈希）。只看头部：压缩删的正是最老的几条，头部一变即锚点失效；取尾部每轮都变⇒锚点永远用不上。trim_history 须能独立复算同一值，故口径与 ledger_callback._fingerprint 一致。
    prompt_fingerprint: str

    #: 预约「补到店时间」的澄清是否已问过一轮（plans/0022）。必须进 State：interrupt() 续跑节点从头重放、局部变量归零⇒"解析不出时间就收尾"永不成立⇒用户答句听不懂的就被无限挂起。拿到答复后置 True；确认/取消/交回入口/新预约时清 False。
    booking_fill_tried: bool

    # ---- 第二层：跨轮焦点，用于解析"这家/第二家/这三家" ----
    # 累积写入（见 merge_focus）：分别问过 A/B/C 后问"这三家对比"，焦点里三家都在。
    focus_shops: Annotated[list[dict], merge_focus]
    shop_ids: list[int]

    #: 最近一次用户点名并成功解析的店铺 id（grounding 写）。单独存在的原因（2026-10-08 真机）：focus_shops 是累积列表、顺序只反映首次出现，没有"最近聊过哪家"；"帮我预约下"要的正是后者，取 focus_shops[0] 会约到首批里最近的美食店、与刚聊的店无关。写侧只在点名解析成功时更新（附近列表类问句不改）。
    last_grounded_shop_id: int | None

    # ---- 第三层：worker 扇出参数（由 Send 注入）----
    #: **旧字段，不要删也不要改类型**：checkpoint 里已落库的会话 worker 入参只有 `task: str`，
    #: 改成 `list` 会让旧会话反序列化后行为错乱。新代码写 `worker_tasks`，读侧两者都认。
    task: str
    #: 本次 Send 要做的**该店全部意图**（`plans/0009.2`：一次 Send = 一家店）。
    #: 空/缺失时回退到 `task`（单值）。
    worker_tasks: list[str]
    worker_shop_ids: list[int]
    worker_query: str

    # ---- 第三层：收窄前置的结果（`nearby_worker` 写出，`route_after_fanout` 读）----
    #: 规则收窄成功后**唯一确定**的目标店 id。非空 ⇒ 目标已明确 ⇒ 跳过 `react_loop`
    #: （它的唯一入口理由是"还不知道是哪家"）。收窄失败时保持 `None`，照旧进环兜底。
    narrowed_shop_id: int | None

    # ---- 第三层：worker 产出（短生命周期，汇总后清空）----
    worker_outputs: Annotated[list[dict], add_and_clear]

    #: 摘要投影（specs/0016 T4）：会话里真正展示过的店（[已推荐] 的确定性养料），结构 [{id,name,score,avgPrice,has_voucher}]；未声明的键被 LangGraph 静默丢弃，必须声明；跨轮累积（merge_digest 按 id 去重+上限），[已推荐] 是会话级记忆。
    recommended_digest: Annotated[list[dict], merge_digest]

    #: 摘要投影（`specs/0016` T4）：**真去取过但失败 / 点名却没找到**的店（`[死路]`）。
    #: 结构：`[{id?, name, reason}]`。同样**必须声明** + 跨轮累积。
    excluded_digest: Annotated[list[dict], merge_digest]

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
