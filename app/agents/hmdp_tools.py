"""本地生活 Agent 工具：通过 HTTP 调用 Java 微服务接口。"""

import json
import os
from contextvars import ContextVar
from typing import Any

import httpx
from pydantic import BaseModel

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
DEFAULT_X = float(os.getenv("DEFAULT_LONGITUDE", "116.397128"))
DEFAULT_Y = float(os.getenv("DEFAULT_LATITUDE", "39.916527"))

_current_user_id: ContextVar[str] = ContextVar("hmdp_user_id", default="")
_current_x: ContextVar[float | None] = ContextVar("hmdp_x", default=None)
_current_y: ContextVar[float | None] = ContextVar("hmdp_y", default=None)


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
        return ApiResponse(success=False, errorMsg="Java 服务地址未配置，请检查 .env")

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
        logger.error(f"调用 Java 服务失败: {base_url}{path}, error: {exc}")
        return ApiResponse(success=False, errorMsg=f"调用 {path} 失败: {exc}")


def _unwrap(payload: ApiResponse | dict[str, Any]) -> Any:
    if isinstance(payload, ApiResponse):
        if not payload.success:
            return payload.errorMsg or "接口调用失败"
        return payload.data
    if isinstance(payload, dict):
        return payload.get("data")
    return payload


def _load_shop_types(user_id: str = "") -> Any:
    return _unwrap(_request_json(resolve_service_url("shop"), "/shop-type/list", user_id=user_id))


def _compact_shops(shops: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


@tool
def get_shop_types() -> str:
    """获取平台全部店铺类型，如美食、火锅、咖啡、KTV，返回类型 id 和名称。"""
    record_tool_call("get_shop_types")
    data = _load_shop_types()
    if isinstance(data, str):
        return data
    if not data:
        return "暂无可用的店铺类型"
    return json.dumps(
        [{"id": t.get("id"), "name": t.get("name"), "icon": t.get("icon")} for t in data],
        ensure_ascii=False,
    )


@tool
def get_shops_nearby(type_name: str, page: int = 1) -> str:
    """按类型名称和当前位置查询附近店铺，例如 type_name=火锅。坐标自动取请求上下文。"""
    record_tool_call("get_shops_nearby")
    x = _current_x.get() if _current_x.get() is not None else DEFAULT_X
    y = _current_y.get() if _current_y.get() is not None else DEFAULT_Y

    type_data = _load_shop_types()
    if isinstance(type_data, str):
        return type_data
    matched = next(
        (t for t in type_data or [] if type_name in str(t.get("name") or "")),
        None,
    )
    if matched is None:
        return f"没有找到类型「{type_name}」"

    # 增加 dis 参数，默认 5000 米
    dis = int(os.getenv("DEFAULT_SEARCH_RADIUS", "50000"))

    payload = _request_json(
        SHOP_SERVICE_URL,
        "/shop/of/type",
        {"typeId": matched.get("id"), "current": page, "x": x, "y": y, "distance": dis},
    )
    shops = _unwrap(payload)
    if isinstance(shops, str):
        return shops
    if not shops:
        return f"第 {page} 页没有「{type_name}」类型的店铺"
    return json.dumps(_compact_shops(shops), ensure_ascii=False)


@tool
def get_shops_by_name(name: str, page: int = 1) -> str:
    """按店铺名称关键字搜索店铺，例如 name=海底捞。"""
    record_tool_call("get_shops_by_name")
    payload = _request_json(
        SHOP_SERVICE_URL,
        "/shop/of/name",
        {"name": name, "current": page},
    )
    shops = _unwrap(payload)
    if isinstance(shops, str):
        return shops
    if not shops:
        return f"没有找到名称包含「{name}」的店铺"
    return json.dumps(_compact_shops(shops), ensure_ascii=False)


@tool
def get_shop_detail(shop_id: int) -> str:
    """查询单个店铺的详细信息，shop_id 为店铺 id。"""
    record_tool_call("get_shop_detail")
    data = _unwrap(_request_json(resolve_service_url("shop"), f"/shop/{shop_id}"))
    if isinstance(data, str):
        return data
    if not data:
        return f"店铺 {shop_id} 不存在"
    return json.dumps(
        {
            "id": data.get("id"),
            "name": data.get("name"),
            "address": data.get("address"),
            "area": data.get("area"),
            "avgPrice": data.get("avgPrice"),
            "score": data.get("score"),
            "sold": data.get("sold"),
            "openHours": data.get("openHours"),
            "distance": data.get("distance"),
        },
        ensure_ascii=False,
    )


@tool
def get_shop_blogs(shop_id: int, page: int = 1) -> str:
    """查询某个店铺下的探店博客和攻略，shop_id 为店铺 id。"""
    record_tool_call("get_shop_blogs")
    payload = _request_json(
        BLOG_SERVICE_URL,
        f"/blog/of/shop/{shop_id}",
        {"current": page},
    )
    blogs = _unwrap(payload)
    if isinstance(blogs, str):
        return blogs
    if not blogs:
        return f"店铺 {shop_id} 暂时没有博客攻略"
    return json.dumps(_compact_blogs(blogs), ensure_ascii=False)


@tool
def get_blog_detail(blog_id: int) -> str:
    """查询单篇博客攻略的完整内容，blog_id 为博客 id。"""
    record_tool_call("get_blog_detail")
    data = _unwrap(_request_json(resolve_service_url("blog"), f"/blog/{blog_id}"))
    if isinstance(data, str):
        return data
    if not data:
        return f"博客 {blog_id} 不存在"
    return json.dumps(
        {
            "id": data.get("id"),
            "shopId": data.get("shopId"),
            "title": data.get("title"),
            "content": data.get("content"),
            "images": data.get("images"),
            "liked": data.get("liked"),
            "comments": data.get("comments"),
            "author": data.get("name"),
            "createTime": data.get("createTime"),
        },
        ensure_ascii=False,
    )


@tool
def get_hot_blogs(page: int = 1) -> str:
    """获取当前热门探店博客列表。"""
    record_tool_call("get_hot_blogs")
    payload = _request_json(resolve_service_url("blog"), "/blog/hot", {"current": page})
    blogs = _unwrap(payload)
    if isinstance(blogs, str):
        return blogs
    if not blogs:
        return "暂时没有热门博客"
    return json.dumps(_compact_blogs(blogs), ensure_ascii=False)


@tool
def get_shop_vouchers(shop_id: int) -> str:
    """查询店铺当前可用的优惠券，shop_id 为店铺 id。"""
    record_tool_call("get_shop_vouchers")
    payload = _request_json(resolve_service_url("voucher"), f"/voucher/list/{shop_id}")
    vouchers = _unwrap(payload)
    if isinstance(vouchers, str):
        return vouchers
    if not vouchers:
        return f"店铺 {shop_id} 暂时没有优惠券"
    return json.dumps(
        [
            {
                "id": v.get("id"),
                "title": v.get("title"),
                "subTitle": v.get("subTitle"),
                "rules": v.get("rules"),
                "payValue": v.get("payValue"),
                "actualValue": v.get("actualValue"),
                "type": v.get("type"),
                "status": v.get("status"),
                "stock": v.get("stock"),
                "beginTime": v.get("beginTime"),
                "endTime": v.get("endTime"),
            }
            for v in vouchers
        ],
        ensure_ascii=False,
    )



def _format_rag_context(shop_id: int, query: str, top_k: int) -> str:
    """把检索切片压缩成精简结构，只保留标题与正文要点。"""
    chunks = retrieve_shop_rag_chunks(shop_id, query, top_k)
    lines = []
    for chunk in chunks or []:
        meta = chunk.get("metadata") or {}
        title = meta.get("title") or "未知标题"
        content = (chunk.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"《{title}》{content}")
    if not lines:
        return f"店铺{shop_id}：根据现有博客暂未查到与「{query}」相关的内容"
    return f"店铺{shop_id}：\n" + "\n".join(lines)


@tool
def get_shop_rag_context(shop_id: int, query: str, top_k: int = 3) -> str:
    """检索某家店铺博客中与 query 相关的文本片段，用于回答价格、服务、体验等细节问题。"""
    record_tool_call("get_shop_rag_context")
    top_k = min(max(top_k, 1), 5)
    return _format_rag_context(int(shop_id), query, top_k)


@tool
def get_shops_rag_context(shop_ids: list[int], query: str, top_k: int = 3) -> str:
    """一次检索多个店铺的博客片段，用于对比多家店铺的停车、价格、环境、优惠等体验。每家店最多返回 3 条。"""
    record_tool_call("get_shops_rag_context")
    top_k = min(max(top_k, 1), 3)
    if not shop_ids:
        return "请提供至少一个店铺ID"
    seen = set()
    parts = []
    for shop_id in shop_ids:
        sid = int(shop_id)
        if sid in seen:
            continue
        seen.add(sid)
        parts.append(_format_rag_context(sid, query, top_k))
    return "\n\n".join(parts)
# ---------------------------------------------------------------------------
# 数据服务层：供 LangGraph 各 Worker 节点直接调用
# 与上面的 @tool 版本的区别是参数显式传入，不依赖 ContextVar（并行 worker 下 ContextVar 不可靠）
# 返回值约定：成功返回 list / dict，失败或查不到返回 str（错误或空结果描述）
# ---------------------------------------------------------------------------


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
        return f"没有找到类型「{type_name}」"

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
        return f"第 {page} 页没有「{type_name}」类型的店铺"
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
        return f"没有找到名称包含「{name}」的店铺"
    return _compact_shops(shops)


def fetch_shop_detail(shop_id: int, user_id: str = "") -> dict[str, Any] | str:
    """查询单个店铺详情。"""
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
    """查询店铺可用优惠券。"""
    record_tool_call("fetch_shop_vouchers")
    payload = _request_json(
        VOUCHER_SERVICE_URL, f"/voucher/list/{shop_id}", user_id=user_id
    )
    vouchers = _unwrap(payload)
    if isinstance(vouchers, str):
        return vouchers
    if not vouchers:
        return f"店铺 {shop_id} 暂时没有优惠券"
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
    shop_id: int, query: str, top_k: int = 3
) -> list[dict[str, Any]] | str:
    """检索某店铺博客中与 query 相关的片段。"""
    record_tool_call("fetch_rag_chunks")
    top_k = min(max(top_k, 1), 5)
    try:
        chunks = retrieve_shop_rag_chunks(int(shop_id), query, top_k)
    except Exception as exc:  # RAG 索引未就绪时不能让整个流程崩掉
        logger.error(f"RAG 检索失败 shop_id={shop_id}: {exc}")
        return f"店铺 {shop_id} 的细节检索暂时不可用"
    if not chunks:
        return f"店铺 {shop_id}：根据现有博客暂未查到与「{query}」相关的内容"
    result = []
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        content = (chunk.get("content") or "").strip()
        if not content:
            continue
        result.append(
            {
                "title": meta.get("title") or "未知标题",
                "author": meta.get("author") or "",
                "content": content[:300],
            }
        )
    if not result:
        return f"店铺 {shop_id}：根据现有博客暂未查到与「{query}」相关的内容"
    return result


TOOLS = [
    get_shop_types,
    get_shops_nearby,
    get_shops_by_name,
    get_shop_detail,
    get_shop_blogs,
    get_blog_detail,
    get_hot_blogs,
    get_shop_vouchers,
    get_shop_rag_context,
    get_shops_rag_context,
]

__all__ = [
    "TOOLS",
    "set_user_context",
    "reset_user_context",
    "fetch_shop_types",
    "fetch_shops_nearby",
    "fetch_shops_by_name",
    "fetch_shop_detail",
    "fetch_shop_blogs",
    "fetch_shop_vouchers",
    "fetch_rag_chunks",
]