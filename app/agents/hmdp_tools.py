"""本地生活 Agent 工具：通过 HTTP 调用 Java 微服务接口。"""

import os
from contextvars import ContextVar
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from dotenv import load_dotenv
from langchain.tools import tool

from app.common.logger import logger
from app.service_discovery import resolve as resolve_service_url
from app.agents.hmdp_metrics import record_tool_call
from app.rag.hmdp_rag import retrieve_shop_rag_chunks

load_dotenv()

SHOP_SERVICE_URL = os.getenv("SHOP_SERVICE_URL", "").rstrip("/")
BLOG_SERVICE_URL = os.getenv("BLOG_SERVICE_URL", "").rstrip("/")
VOUCHER_SERVICE_URL = os.getenv("VOUCHER_SERVICE_URL", "").rstrip("/")
#: 站内搜索服务地址（specs/0026）：细分词用博客全文反查店铺时调用，走 search-service（GET /search/blog）而非直连 ES（护栏都在那边）。
#: 留空则该工具返回 SERVICE_UNAVAILABLE，主链路降级回原行为（不影响原有召回）。
SEARCH_SERVICE_URL = os.getenv("SEARCH_SERVICE_URL", "").rstrip("/")
# 兜底坐标必须与演示数据同城（店铺全在杭州，默认杭州坐标）；否则客户端未传坐标时 `/shop/of/type` 距离过滤会筛空、回退按名搜、答复出现"距离暂未获取"。
# `.env` 的 DEFAULT_LONGITUDE/DEFAULT_LATITUDE 须与 `.env.example` 一起改，避免配置漂移退回北京。
DEFAULT_X = float(os.getenv("DEFAULT_LONGITUDE", "120.149192"))
DEFAULT_Y = float(os.getenv("DEFAULT_LATITUDE", "30.316078"))

_current_user_id: ContextVar[str] = ContextVar("hmdp_user_id", default="")
_current_x: ContextVar[float | None] = ContextVar("hmdp_x", default=None)
_current_y: ContextVar[float | None] = ContextVar("hmdp_y", default=None)

# ---- 「业务为空」的**用户向**文案模板（唯一真相源）--------------------------
#: `fetch_*` 的统一契约是 **`list | str`**：取数失败与业务为空都返回字符串。调用方若想区分"没这家店 / 没这类店"与"服务出错"，必须按**等值比较**分类（`startswith` 嗅探会因文案改字而静默失效）。
#: 措辞必须是用户能看懂的话（这些串会经 `render_context` 被模型原样转达），**分页参数、字段名等内部实现细节一律不许出现**。
SHOP_NAME_NOT_FOUND = "没有找到名称包含「{name}」的店铺"
SHOP_TYPE_NOT_FOUND = "暂时没有「{type_name}」这一类店铺"
SHOPS_NEARBY_EMPTY = "附近暂时没有「{type_name}」相关的店铺"

#: 下游 HTTP 调用**技术性失败**（连不上 / 非 2xx / 坏 JSON / 地址没配）时的文案。
#: 同样进 `list | str` → `render_context` → 被模型**原样转达给用户**，故不许出现 URL、异常类名、
#: `WinError` / `Connection refused` 等实现细节；原始错误（`{base_url}{path}` + 异常）**只进日志**。
SERVICE_UNAVAILABLE = "信息暂时查不到，请稍后再试。"


def set_user_context(user_id: str, x: float | None, y: float | None) -> list[Any]:
    """写入一次请求的用户上下文，返回用于恢复的 token。"""
    return [
        _current_user_id.set(user_id or ""),
        _current_x.set(x),
        _current_y.set(y),
    ]


def reset_user_context(tokens: list[Any]) -> None:
    for var, token in zip((_current_user_id, _current_x, _current_y), tokens):
        var.reset(token)


class ApiResponse(BaseModel):
    success: bool
    errorMsg: str | None = None
    data: Any = None


def _request_json(
    base_url: str,
    path: str,
    params: dict[str, Any] | None = None,
    user_id: str | None = None,
) -> ApiResponse:
    if not base_url:
        logger.error("Java 服务地址未配置（检查 .env 的 *_SERVICE_URL / NACOS_*）")
        return ApiResponse(success=False, errorMsg=SERVICE_UNAVAILABLE)

    headers = {"Accept": "application/json"}
    uid = user_id if user_id is not None else _current_user_id.get()
    if uid:
        headers["user-info"] = str(uid)

    try:
        with httpx.Client(base_url=base_url, timeout=5.0, headers=headers) as client:
            resp = client.get(
                path,
                params={k: v for k, v in (params or {}).items() if v is not None},
            )
            resp.raise_for_status()
            return ApiResponse.model_validate(resp.json())
    except Exception as exc:
        # 原始错误（含 URL 与异常类型）**只进日志**；返回给调用方的是用户向文案
        logger.error("调用 Java 服务失败: %s%s, error: %s", base_url, path, exc)
        return ApiResponse(success=False, errorMsg=SERVICE_UNAVAILABLE)


def _unwrap(payload: ApiResponse | dict[str, Any]) -> Any:
    if isinstance(payload, ApiResponse):
        if not payload.success:
            return payload.errorMsg or "接口调用失败"
        return payload.data
    if isinstance(payload, dict):
        return payload.get("data")
    return payload


def _post_json(
    base_url: str,
    path: str,
    json_body: dict[str, Any] | None = None,
    user_id: str | None = None,
) -> ApiResponse:
    """POST 请求（写路径专用，与 `_request_json` 分开）。写操作必须只有一个出口，便于限流/审计/回滚；
    错误处理与只读侧一致（原始错误只进日志、返回用户向文案），预约失败不抛异常进图。"""
    if not base_url:
        logger.error("Java 服务地址未配置（检查 .env 的 *_SERVICE_URL / NACOS_*）")
        return ApiResponse(success=False, errorMsg=SERVICE_UNAVAILABLE)

    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    uid = user_id if user_id is not None else _current_user_id.get()
    if uid:
        # `user-info` 必须是**数字**：Java 侧 common 拦截器直接 `Long.parseLong`，
        #    非数字会在进入业务前抛 NumberFormatException ⇒ 下游全 500。
        headers["user-info"] = str(uid)

    try:
        with httpx.Client(base_url=base_url, timeout=10.0, headers=headers) as client:
            resp = client.post(path, json=json_body or {})
            resp.raise_for_status()
            return ApiResponse.model_validate(resp.json())
    except Exception as exc:
        logger.error("调用 Java 服务失败(POST): %s%s, error: %s", base_url, path, exc)
        return ApiResponse(success=False, errorMsg=SERVICE_UNAVAILABLE)


#: 预约失败的用户向文案（**不编造原因**：具体失败原因只进日志）。
BOOKING_FAILED = "预约没有成功，请稍后再试。"
#: 未登录时的文案（写操作前置条件不满足，**不挂起、不下单**）。
BOOKING_NEEDS_LOGIN = "预约需要先登录，登录后我再帮你提交。"

#: 撤单失败的用户向文案（`plans/0026`）。同样**不编造原因**：具体失败原因只进日志。
BOOKING_CANCEL_FAILED = "取消预约没有成功，请稍后再试。"


def create_booking(
    shop_id: int,
    booking_time: str,
    people: int | None = None,
    remark: str = "",
    user_id: str = "",
) -> dict[str, Any]:
    """提交门店预约（唯一写操作）。幂等由 Java 侧唯一键 uk_user_shop_time 兜底（plans/0018），同店同时间重复提交返回既有单号。
    :param booking_time: yyyy-MM-dd HH:mm:ss（Java 侧 @JsonFormat 口径）。
    :returns: {"ok","order_id","created","message"}，不抛异常；`created` 缺省按 True（宁可少说一次，别误报"没新建"）。"""
    record_tool_call("create_booking")
    if not shop_id or not booking_time:
        return {"ok": False, "order_id": None, "created": False, "message": BOOKING_FAILED}

    body: dict[str, Any] = {"shopId": int(shop_id), "bookingTime": booking_time}
    if people is not None:
        body["people"] = int(people)
    if remark:
        body["remark"] = str(remark)[:255]

    payload = _unwrap(
        _post_json(resolve_service_url("shop"), "/shop/booking", body, user_id=user_id)
    )
    if isinstance(payload, str):
        # 服务地址没配 / 网络失败 / 非 2xx
        return {"ok": False, "order_id": None, "created": False, "message": payload}
    if isinstance(payload, dict):
        # Java 侧返回 {"orderId": 123, "created": true|false}；取不到单号视为失败（不能谎报成功）
        order_id = payload.get("orderId")
        if order_id is not None:
            return {
                "ok": True,
                "order_id": int(order_id),
                "created": bool(payload.get("created", True)),
                "message": "",
            }
    return {"ok": False, "order_id": None, "created": False, "message": BOOKING_FAILED}


def cancel_booking(order_id: int, user_id: str = "") -> dict[str, Any]:
    """撤销门店预约（plans/0026，agent 第二条写路径）。同步 httpx，调用方须 asyncio.to_thread。
    `order_id` 只接受来自 State（booking_result.order_id），绝不收前端 id（否则可撤他人单）；Java 侧幂等，无需先查再撤。
    :returns: {"ok","cancelled","message"}，不抛异常。"""
    record_tool_call("cancel_booking")
    if not order_id:
        return {"ok": False, "cancelled": False, "message": BOOKING_CANCEL_FAILED}

    payload = _unwrap(
        _post_json(
            resolve_service_url("shop"), "/shop/booking/cancel", {"orderId": int(order_id)}, user_id=user_id
        )
    )
    if isinstance(payload, str):
        return {"ok": False, "cancelled": False, "message": payload}
    # Java 侧 `Result.ok(updated)` ⇒ data 是布尔；**取不到布尔一律当失败**（不能谎报"已取消"）
    if isinstance(payload, bool):
        return {"ok": True, "cancelled": payload, "message": ""}
    return {"ok": False, "cancelled": False, "message": BOOKING_CANCEL_FAILED}


def _load_shop_types(user_id: str = "") -> Any:
    return _unwrap(_request_json(resolve_service_url("shop"), "/shop-type/list", user_id=user_id))


def _compact_shops(shops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """候选列表的投影。`x`/`y` 必须带上（`specs/0012`）：行程/地图按坐标画点、算站间直线距离。"""
    return [
        {
            "id": s.get("id"),
            "name": s.get("name"),
            "address": s.get("address"),
            "area": s.get("area"),
            "distance": s.get("distance"),
            "avgPrice": s.get("avgPrice"),
            "score": s.get("score"),
            "sold": s.get("sold"),
            "x": s.get("x"),
            "y": s.get("y"),
        }
        for s in shops or []
    ]


def _compact_blogs(blogs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": b.get("id"),
            "shopId": b.get("shopId"),
            "title": b.get("title"),
            "content": (b.get("content") or "")[:120],
            "liked": b.get("liked"),
            "comments": b.get("comments"),
            "author": b.get("name"),
            "createTime": b.get("createTime"),
        }
        for b in blogs or []
    ]


#: ReAct 补跳环专用工具（`REACT_TOOLS`）—— **单一真相源**：不复用下面的 `fetch_*`（那组是动词前缀、含写语义与"多店批量"，环内**只允许只读、单店**）。
# 这些函数的函数体**不会被执行**：`bind_tools(REACT_TOOLS)` 只为把 schema 发给模型，`tool_calls` 由 `nodes._react_exec_tool` 接管（超时预算 / `list | str` 归一化 / 下单卡片 / 降级不外抛）。
# ⇒ 唯一重要的是名字 + `args_schema` + docstring；三处必须同名（`test_react_tool_channel.py` 看住）：`REACT_TOOLS[*].name` == `nodes.REACT_READONLY_TOOLS` == 日志 `react_loop step=N tool=...`。

class _ReactToolArgs(BaseModel):
    """ReAct 环工具参数的公共基类：`extra="forbid"` ⇒ 模型多给参数会被**显式拒绝**。
    `with_structured_output` 走 `json_schema`，即便声明 `strict:true` 服务端也不执行
    （实测违反 `required` 与 `additionalProperties` 仍零报错）⇒ 只能靠这里。"""

    model_config = ConfigDict(extra="forbid")


class ShopsNearbyArgs(_ReactToolArgs):
    type_name: str = Field(description="店铺类型名称，例如「火锅」「咖啡」")


class ShopsByNameArgs(_ReactToolArgs):
    name: str = Field(description="店铺名称关键字，例如「海底捞」")


class ShopDetailArgs(_ReactToolArgs):
    shop_id: int = Field(description="店铺 id（整数）", ge=1)


class ShopVouchersArgs(_ReactToolArgs):
    shop_id: int = Field(description="店铺 id（整数）", ge=1)


class ShopRagArgs(_ReactToolArgs):
    shop_id: int = Field(description="店铺 id（整数）", ge=1)
    query: str = Field(description="要检索的关键词，例如「停车」「包间」")


_REACT_TOOL_NOT_EXECUTED = "（ReAct 工具只用于生成 schema，实际执行由 _react_exec_tool 接管）"


@tool("shops_nearby", args_schema=ShopsNearbyArgs)
def react_shops_nearby(type_name: str) -> str:
    """按类型查询附近的店铺，返回候选列表（含 id、人均、评分、距离）。
    只在「手上还没有候选店铺」时才需要调。"""
    return _REACT_TOOL_NOT_EXECUTED


@tool("shops_by_name", args_schema=ShopsByNameArgs)
def react_shops_by_name(name: str) -> str:
    """按店名关键字查店铺，返回候选列表（含 id）。用于把用户点名的店解析成 id。"""
    return _REACT_TOOL_NOT_EXECUTED


@tool("shop_detail", args_schema=ShopDetailArgs)
def react_shop_detail(shop_id: int) -> str:
    """查询某家店的人均价格、评分、距离、销量、营业时间。比价/定位前需要它。"""
    return _REACT_TOOL_NOT_EXECUTED


@tool("shop_vouchers", args_schema=ShopVouchersArgs)
def react_shop_vouchers(shop_id: int) -> str:
    """查询某家店当前可用的代金券。用户问「有什么券/代金券」时需要它。"""
    return _REACT_TOOL_NOT_EXECUTED


@tool("shop_rag", args_schema=ShopRagArgs)
def react_shop_rag(shop_id: int, query: str) -> str:
    """检索某家店博客里的细节片段（停车、包间、环境、服务）。问「细节/体验」时需要它。"""
    return _REACT_TOOL_NOT_EXECUTED


#: ReAct 环**唯一**的工具清单 —— 绑定列表 == 白名单 == 日志口径。
#: 没有 `finish`：收敛由「模型**一个工具都不调**」表达（见 `nodes._react_decide`）。
REACT_TOOLS: tuple[Any, ...] = (
    react_shops_nearby,
    react_shops_by_name,
    react_shop_detail,
    react_shop_vouchers,
    react_shop_rag,
)

# ---- 数据服务层：供 LangGraph 各 Worker 节点直接调用 -------------------------
# 参数显式传入，不依赖 ContextVar（并行 worker 下 ContextVar 不可靠）；
# 返回值约定：成功返回 list / dict，失败或查不到返回 str（错误或空结果描述）。


def fetch_shop_types(user_id: str = "") -> list[dict[str, Any]] | str:
    """获取全部店铺类型。"""
    record_tool_call("fetch_shop_types")
    data = _load_shop_types(user_id)
    if isinstance(data, str):
        return data
    if not data:
        return "暂无可用的店铺类型"
    return [
        {"id": t.get("id"), "name": t.get("name"), "icon": t.get("icon")}
        for t in data
    ]


def fetch_shops_nearby(
    type_name: str,
    x: float | None,
    y: float | None,
    page: int = 1,
    user_id: str = "",
) -> list[dict[str, Any]] | str:
    """按类型查询附近店铺。"""
    record_tool_call("fetch_shops_nearby")
    px = x if x is not None else DEFAULT_X
    py = y if y is not None else DEFAULT_Y

    type_data = _load_shop_types(user_id)
    if isinstance(type_data, str):
        return type_data
    matched = next(
        (t for t in type_data or [] if type_name and type_name in str(t.get("name") or "")),
        None,
    )
    if matched is None:
        return SHOP_TYPE_NOT_FOUND.format(type_name=type_name)

    dis = int(os.getenv("DEFAULT_SEARCH_RADIUS", "50000"))
    payload = _request_json(
        SHOP_SERVICE_URL,
        "/shop/of/type",
        {"typeId": matched.get("id"), "current": page, "x": px, "y": py, "distance": dis},
        user_id=user_id,
    )
    shops = _unwrap(payload)
    if isinstance(shops, str):
        return shops
    if not shops:
        # 同上：这是**用户向**文案，不许出现页码
        return SHOPS_NEARBY_EMPTY.format(type_name=type_name)
    return _compact_shops(shops)


def fetch_shops_by_name(
    name: str, page: int = 1, user_id: str = ""
) -> list[dict[str, Any]] | str:
    """按店铺名称关键字搜索。"""
    record_tool_call("fetch_shops_by_name")
    payload = _request_json(
        SHOP_SERVICE_URL, "/shop/of/name", {"name": name, "current": page}, user_id=user_id
    )
    shops = _unwrap(payload)
    if isinstance(shops, str):
        return shops
    if not shops:
        return SHOP_NAME_NOT_FOUND.format(name=name)
    return _compact_shops(shops)


def fetch_shop_detail(shop_id: int, user_id: str = "") -> dict[str, Any] | str:
    """查询单个店铺详情。含 `x`/`y` 坐标（`specs/0012`：行程节点要按坐标画图）。"""
    record_tool_call("fetch_shop_detail")
    data = _unwrap(
        _request_json(resolve_service_url("shop"), f"/shop/{shop_id}", user_id=user_id)
    )
    if isinstance(data, str):
        return data
    if not data:
        return f"店铺 {shop_id} 不存在"
    return {
        "id": data.get("id"),
        "name": data.get("name"),
        "address": data.get("address"),
        "area": data.get("area"),
        "avgPrice": data.get("avgPrice"),
        "score": data.get("score"),
        "sold": data.get("sold"),
        "openHours": data.get("openHours"),
        "distance": data.get("distance"),
        "x": data.get("x"),
        "y": data.get("y"),
    }


def fetch_shop_blogs(
    shop_id: int, page: int = 1, user_id: str = ""
) -> list[dict[str, Any]] | str:
    """查询某店铺下的探店博客。"""
    record_tool_call("fetch_shop_blogs")
    payload = _request_json(
        BLOG_SERVICE_URL, f"/blog/of/shop/{shop_id}", {"current": page}, user_id=user_id
    )
    blogs = _unwrap(payload)
    if isinstance(blogs, str):
        return blogs
    if not blogs:
        return f"店铺 {shop_id} 暂时没有博客攻略"
    return _compact_blogs(blogs)


def fetch_shop_vouchers(shop_id: int, user_id: str = "") -> list[dict[str, Any]] | str:
    """查询店铺可用代金券。"""
    record_tool_call("fetch_shop_vouchers")
    payload = _request_json(
        VOUCHER_SERVICE_URL, f"/voucher/list/{shop_id}", user_id=user_id
    )
    vouchers = _unwrap(payload)
    if isinstance(vouchers, str):
        return vouchers
    if not vouchers:
        return f"店铺 {shop_id} 暂时没有代金券"
    return [
        {
            "id": v.get("id"),
            "title": v.get("title"),
            "subTitle": v.get("subTitle"),
            "rules": v.get("rules"),
            "payValue": v.get("payValue"),
            "actualValue": v.get("actualValue"),
            "type": v.get("type"),
            "stock": v.get("stock"),
            "beginTime": v.get("beginTime"),
            "endTime": v.get("endTime"),
        }
        for v in vouchers
    ]


def fetch_rag_chunks(
    shop_id: int, query: str, top_k: int = 3, deadline: float | None = None
) -> list[dict[str, Any]] | str:
    """检索某店铺博客中与 query 相关的片段。
    `deadline`（`time.monotonic()` 绝对截止时刻）**只透传给可选重排**（见 `hmdp_rerank`）；不传 = 无预算约束。"""
    record_tool_call("fetch_rag_chunks")
    top_k = min(max(top_k, 1), 5)
    try:
        chunks = retrieve_shop_rag_chunks(int(shop_id), query, top_k, deadline)
    except Exception as exc:  # RAG 索引未就绪时不能让整个流程崩掉
        logger.error(f"RAG 检索失败 shop_id={shop_id}: {exc}")
        return f"店铺 {shop_id} 的细节检索暂时不可用"
    if not chunks:
        return f"店铺 {shop_id}：暂未查到与「{query}」相关的内容"
    result = []
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        content = (chunk.get("content") or "").strip()
        if not content:
            continue
        result.append(
            {
                # `id` 取索引 metadata 的 blog_id（hmdp_rag 建 chunk 时写死），仅用于前端"数据来源可回溯"跳转（plans/0025），非模型生成、不进模型上下文。
                "id": meta.get("blog_id"),
                "title": meta.get("title") or "未知标题",
                "author": meta.get("author") or "",
                "content": content[:300],
            }
        )
    if not result:
        return f"店铺 {shop_id}：暂未查到与「{query}」相关的内容"
    return result


def search_blogs_by_key(key: str, size: int = 5) -> list[int] | str:
    """经 search-service 在博客全文索引检索 `key`，返回命中博客所属的店铺 id 列表（specs/0026 细分词补充召回通道）。
    只读、走 search-service（不直连 ES）；沿用 list|str 契约，不可用时返回用户向文案、调用方降级回原链路。
    :param key: 检索关键词（如"牛油果"）。:param size: 命中博客条数上限（search-service 侧钳制 ≤50，先夹一次避免无意义请求）。:returns: 去重、保序的 shopId 列表；无命中返回空列表，技术性失败返回错误文案（str）。"""
    record_tool_call("search_blogs_by_key")
    text = str(key or "").strip()
    if not text:
        return []
    try:
        size = min(max(int(size), 1), 50)
    except (TypeError, ValueError):
        size = 5

    payload = _unwrap(
        _request_json(
            SEARCH_SERVICE_URL,
            "/search/blog",
            {"key": text, "size": size, "sort": "relevance"},
        )
    )
    # 服务地址没配 / 网络失败 / 非 2xx ⇒ 返回用户向文案（调用方降级）
    if isinstance(payload, str):
        return payload
    hits = payload.get("hits") if isinstance(payload, dict) else None
    if not hits:
        return []

    ids: list[int] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        # `shopId` 为 JSON number（Java 侧 Long）⇒ 只认 int，避免 null / 字符串混入
        sid = hit.get("shopId")
        if isinstance(sid, int) and sid not in ids:
            ids.append(sid)
    return ids


# ---- 下单卡片：纯数据组装（**零出站**，`specs/0006-agent-order-card-boundary.md`）----
# agent 不下单、不读订单状态，只把券信息组装成前端可渲染 / 可跳转的卡片，URL 由前端自行拼接。
# 本函数**绝对不能**发任何网络请求（调 `_request_json` / `seckill` 就退回了被否定的"agent 写下单"）。
def prepare_order_card(voucher: dict[str, Any], shop_id: int) -> dict[str, Any]:
    """把券信息组装成前端可渲染 / 可跳转的卡片。**纯函数，零出站请求。**
    普通券（`type != 1`）→ ``action="confirm"``（仅携带数据，提交 URL 由前端拼）；
    秒杀券（`type == 1`，hm-dianping 约定）→ ``action="goto_activity"``，不含可提交的确认载荷。"""
    #: hm-dianping 约定：`Voucher.type` = 1 表示秒杀券（见 `Voucher` 实体的 `type` 字段，
    #: 与 `VoucherServiceImpl.addSeckillVoucher` 把 DTO 的 type 原样写入 `tb_voucher.type` 的口径；
    #: 本仓库未定义枚举常量，按约定取 1）。
    is_seckill = voucher.get("type") == 1
    card = {
        "action": "goto_activity" if is_seckill else "confirm",
        "voucher_id": voucher.get("id"),
        "shop_id": shop_id,
        "title": voucher.get("title"),
        "sub_title": voucher.get("subTitle"),
        "pay_value": voucher.get("payValue"),
        "actual_value": voucher.get("actualValue"),
        "stock": voucher.get("stock"),
        "begin_time": voucher.get("beginTime"),
        "end_time": voucher.get("endTime"),
        "is_seckill": is_seckill,
    }
    if is_seckill:
        # 秒杀券：只给跳转参数，绝不带可提交的确认载荷
        # （确认载荷 = 供前端直接拼出下单请求的完整数据；秒杀走活动页，由前端自行拉取）
        card.pop("pay_value", None)
        card.pop("actual_value", None)
        card.pop("stock", None)
    return card


__all__ = [
    "set_user_context",
    "reset_user_context",
    "fetch_shop_types",
    "fetch_shops_nearby",
    "fetch_shops_by_name",
    "fetch_shop_detail",
    "fetch_shop_blogs",
    "fetch_shop_vouchers",
    "fetch_rag_chunks",
    "search_blogs_by_key",
    "prepare_order_card",
    # ---- 写路径（唯一一个，别再新增第二个出口）----
    "create_booking",
]
