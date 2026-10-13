"""博客 RAG 索引与检索（混合检索：向量 Chroma + 稀疏 ES + RRF）。
稀疏侧原本地 bm25s+jieba 已迁 ES 后整体删除，降级路径不得再加回（见 test_rag_es_sparse.py::NoBm25FallbackTest，改坏必红）。"""

import json
import os
import threading
from pathlib import Path

import httpx
from typing import Any

from dotenv import load_dotenv

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.common.config import require_env
from app.common.logger import logger
from app.rag.hmdp_rerank import rerank_chunks

load_dotenv()

BLOG_SERVICE_URL = os.getenv("BLOG_SERVICE_URL", "").rstrip("/")
RAG_DB_DIR = Path("db/rag")
CHROMA_DIR = str(RAG_DB_DIR / "chroma")
CHUNKS_JSON = str(RAG_DB_DIR / "chunks.json")
COLLECTION_NAME = "hmdp_blog"
# 嵌入模型是"用哪个模型"的标识符，必须来自 env，缺失即 fail-fast —— 不再静默回退
#    "text-embedding-v4"（漏配会悄无声息用错/未授权模型）。.env 已配置 RAG_EMBEDDING_MODEL。
RAG_EMBEDDING_MODEL = require_env("RAG_EMBEDDING_MODEL")
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "400"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
BLOG_PAGE_SIZE = int(os.getenv("BLOG_PAGE_SIZE", "100"))
RAG_HYBRID = os.getenv("RAG_HYBRID", "true").lower() == "true"
RAG_EXPAND_QUERY = os.getenv("RAG_EXPAND_QUERY", "true").lower() == "true"

# ---- 稀疏侧迁 ES（plans/0005 T7 / specs/0005 §5.3.2）------------------------
# 稀疏检索不再自建 BM25，直接查 ES 的 `hmdp_blog` 别名。降级语义：连不上 / 超时 / 字段缺失
# → **只用向量侧**（RRF 单列表），不抛、不阻塞主链路；**不降级回 BM25**（降级路径不该变成未被测试的隐性主路径）。
ES_HOST = os.getenv("ES_HOST", "")
ES_PORT = int(os.getenv("ES_PORT", "9200"))
ES_BLOG_ALIAS = os.getenv("ES_BLOG_ALIAS", "hmdp_blog")
ES_SPARSE_TIMEOUT = float(os.getenv("ES_SPARSE_TIMEOUT", "3"))

_embeddings: DashScopeEmbeddings | None = None
_vectorstore: Chroma | None = None
_chunks_map: dict[str, dict[str, Any]] | None = None
_index_lock = threading.Lock()
#: 向量库懒加载**专用**锁 —— **不得**与 `_index_lock` 复用（见 `_get_vectorstore`）：
#: `sync_blog_by_id` 会在**持有 `_index_lock` 期间**调用 `_get_vectorstore`，
#: 而 `threading.Lock` **不可重入** ⇒ 复用会**同线程死锁**（已由单测看住）。
_vectorstore_lock = threading.Lock()

QUERY_EXPANSION = {
    "酒水": ["酒水", "饮料", "饮品", "啤酒", "白酒", "红酒"],
    "停车": ["停车", "车位", "停车场"],
    "包间": ["包间", "包厢", "包房"],
    "收费": ["收费", "价格", "多少钱", "费用"],
    "味道": ["味道", "口味", "好吃", "口感"],
    "服务": ["服务", "服务员", "态度"],
    "环境": ["环境", "装修", "氛围"],
    "排队": ["排队", "等位", "等待"],
}


def _get_embeddings() -> DashScopeEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = DashScopeEmbeddings(
            model=RAG_EMBEDDING_MODEL,
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
    return _embeddings


def _get_vectorstore() -> Chroma:
    """懒加载向量库单例（**并发安全**，double-checked：先无锁快路径，未命中才进锁并在锁内再判一次）。
    必须有锁：多个 `shop_worker` 并行经 `asyncio.to_thread` 首访时会各构造一个 `Chroma`，chromadb 1.x 的 rust binding 在并发初始化下崩溃（`'RustBindingsAPI' object has no attribute 'bindings'`，症状「首访必挂、次访恢复」）。
    用独立的 `_vectorstore_lock`，**不能**复用 `_index_lock`：`sync_blog_by_id` 持锁期间会调本函数，`threading.Lock` 不可重入 ⇒ 同线程死锁。"""
    global _vectorstore
    if _vectorstore is not None:  # 快路径：已构造，不入锁
        return _vectorstore
    with _vectorstore_lock:
        if _vectorstore is None:  # 迟到者重判：只让**第一个**进来的构造
            RAG_DB_DIR.mkdir(parents=True, exist_ok=True)
            _vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=_get_embeddings(),
                persist_directory=CHROMA_DIR,
            )
    return _vectorstore


def _fetch_json(url: str) -> dict[str, Any]:
    with httpx.Client(timeout=10.0, headers={"Accept": "application/json"}) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.json()


def _fetch_all_blogs() -> list[dict[str, Any]]:
    if not BLOG_SERVICE_URL:
        raise RuntimeError("BLOG_SERVICE_URL 未配置，无法同步博客索引")
    all_blogs = []
    current = 1
    while True:
        payload = _fetch_json(
            f"{BLOG_SERVICE_URL}/blog/page?current={current}&size={BLOG_PAGE_SIZE}"
        )
        if not payload.get("success"):
            raise RuntimeError(payload.get("errorMsg") or "博客分页接口调用失败")
        records = payload.get("data") or []
        if not records:
            break
        all_blogs.extend(records)
        if len(records) < BLOG_PAGE_SIZE:
            break
        current += 1
    return all_blogs


def _build_documents(blogs: list[dict[str, Any]]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " "],
    )
    docs = []
    for blog in blogs:
        content = blog.get("content") or ""
        title = blog.get("title") or ""
        if not content.strip() and not title.strip():
            continue
        text = f"标题：{title}\n正文：{content}"
        chunks = splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "shop_id": blog.get("shopId"),
                        "blog_id": blog.get("id"),
                        "title": title,
                        "author": blog.get("name") or "",
                        "create_time": str(blog.get("createTime") or ""),
                        "liked": blog.get("liked") or 0,
                        "comments": blog.get("comments") or 0,
                        "chunk_index": idx,
                    },
                    id=f"{blog.get('id')}_{idx}",
                )
            )
    return docs


def _build_chunks_map(docs: list[Document]) -> dict[str, dict[str, Any]]:
    return {
        doc.id: {
            "id": doc.id,
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    }


def _save_chunks_map(chunks_map: dict[str, dict[str, Any]]) -> None:
    RAG_DB_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks_map, f, ensure_ascii=False)


def _load_chunks_map() -> dict[str, dict[str, Any]]:
    global _chunks_map
    if _chunks_map is None:
        if Path(CHUNKS_JSON).exists():
            with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
                _chunks_map = json.load(f)
        else:
            _chunks_map = {}
    return _chunks_map


def _build_es_query(query: str, shop_id: int, k: int) -> dict[str, Any]:
    """构造 ES 稀疏检索查询体（**纯函数**，便于单测断言结构）。
    `bool.filter = [term.shopId]` 先按店铺收窄（`filter` 不参与打分、可缓存）；`bool.must = [multi_match(query, [title^2, content])]` 标题权重更高。
    必须用 `term`（精确值、大小写敏感）而非 `match`：`shopId` 若是 `keyword` 映射，`match` 在某些分析器下会分词而漏掉。"""
    return {
        "query": {
            "bool": {
                "filter": [{"term": {"shopId": shop_id}}],
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^2", "content"],
                        }
                    }
                ],
            }
        },
        "size": max(int(k), 1),
    }


def _es_search(query: str, shop_id: int, k: int) -> list[tuple[str, float]]:
    """查 ES 的 blog 别名做稀疏检索。**绝不抛异常**：任何失败（未配置 / 连不上 / 超时 /
    字段缺失 / 返回体不符预期）都只 `logger.warning` 并返回 `[]`，让调用方退化成"只用向量侧"。"""
    if not ES_HOST:
        logger.warning("ES_HOST 未配置，稀疏检索降级为「仅向量」")
        return []
    try:
        url = f"http://{ES_HOST}:{ES_PORT}/{ES_BLOG_ALIAS}/_search"
        body = _build_es_query(query, shop_id, k)
        resp = httpx.post(url, json=body, timeout=ES_SPARSE_TIMEOUT)
        if resp.status_code >= 400:
            logger.warning("ES 稀疏检索返回 %s，降级为「仅向量」", resp.status_code)
            return []
        hits = ((resp.json() or {}).get("hits") or {}).get("hits") or []
        items: list[tuple[str, float]] = []
        for hit in hits:
            src = hit.get("_source") or {}
            doc_id = src.get("id") or src.get("blogId") or hit.get("_id")
            if doc_id is None:
                continue
            items.append((str(doc_id), float(hit.get("_score") or 0.0)))
        return items[:k]
    except Exception as exc:  # noqa: BLE001  兜底捕 Exception（不是 ConnectionError —— 超时/解析错不会命中）
        logger.warning("ES 稀疏检索失败，降级为「仅向量」: %s", exc)
        return []


def _sparse_search(query: str, shop_id: int, k: int) -> list[tuple[str, float]]:
    """稀疏检索入口：**优先 ES**；ES 不可用时返回空列表（由调用方做单列表融合）。"""
    return _es_search(query, shop_id, k)


def _expand_query(query: str) -> str:
    if not RAG_EXPAND_QUERY:
        return query
    parts = [query]
    for keyword, words in QUERY_EXPANSION.items():
        if keyword in query:
            parts.extend(words)
    return " ".join(dict.fromkeys(parts))




def _same_shop(meta_shop_id: Any, shop_id: Any) -> bool:
    """兼容知识库中 shop_id 可能为字符串或数字的情况。"""
    return str(meta_shop_id) == str(shop_id)

def _dense_search(shop_id: int, query: str, k: int) -> list[tuple[str, float]]:
    vectorstore = _get_vectorstore()
    try:
        results = vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter={"shop_id": str(shop_id)},
        )
    except Exception:
        results = vectorstore.similarity_search_with_score(query, k=k)

    matched = []
    for doc, score in results:
        if not _same_shop(doc.metadata.get("shop_id"), shop_id):
            continue
        doc_id = doc.id or f"{doc.metadata.get('blog_id')}_{doc.metadata.get('chunk_index')}"
        matched.append((doc_id, float(score)))
    return matched[:k]


def _rrf(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            if doc_id is None:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def rebuild_blog_index() -> dict[str, Any]:
    """全量重建索引（稠密侧 Chroma；稀疏侧由 broker 侧 Canal 持续同步进 ES，不在此重建）。"""
    global _chunks_map
    blogs = _fetch_all_blogs()
    docs = _build_documents(blogs)
    chunks_map = _build_chunks_map(docs)
    _save_chunks_map(chunks_map)
    _chunks_map = chunks_map

    vectorstore = _get_vectorstore()
    vectorstore.reset_collection()
    if docs:
        vectorstore.add_documents(docs)

    logger.info("RAG 稠密索引重建完成，博客数: %s，chunk 数: %s", len(blogs), len(docs))
    return {
        "total_blogs": len(blogs),
        "total_chunks": len(docs),
    }


def _count_chunks() -> int:
    try:
        return _get_vectorstore()._collection.count()
    except Exception:
        try:
            return len(_get_vectorstore().get()["ids"])
        except Exception:
            return 0


def get_index_stats() -> dict[str, Any]:
    return {
        "chunk_count": _count_chunks(),
        "embedding_model": RAG_EMBEDDING_MODEL,
        "chroma_dir": CHROMA_DIR,
        "hybrid": RAG_HYBRID,
        "es_sparse": bool(ES_HOST),
        "query_expansion": RAG_EXPAND_QUERY,
        "blog_page_size": BLOG_PAGE_SIZE,
    }
def retrieve_shop_rag_chunks(
    shop_id: int, query: str, top_k: int = RAG_TOP_K, deadline: float | None = None
) -> list[dict[str, Any]]:
    """返回店铺博客相关 chunk（向量 + 稀疏(ES) + RRF + 可选重排）。
    稀疏侧降级是**单列表融合**：ES 返回空列表时 `ranked_lists` 只剩向量一侧，`_rrf` 天然支持，
    **不会**退回去自己跑 BM25；`deadline`（`time.monotonic()` 绝对时刻）**只传给重排**，向量/稀疏是主路径。"""
    expanded_query = _expand_query(query)
    dense_results = _dense_search(shop_id, query, top_k * 3)
    ranked_lists = [[doc_id for doc_id, _ in dense_results]]
    if RAG_HYBRID:
        sparse_results = _sparse_search(expanded_query, shop_id, top_k * 3)
        if sparse_results:
            ranked_lists.append([doc_id for doc_id, _ in sparse_results])
        else:
            logger.warning("稀疏检索无结果（ES 不可用或确无命中），本轮只用向量侧融合")

    fused = _rrf(ranked_lists)
    chunks_map = _load_chunks_map()

    matched = []
    for doc_id, score in fused:
        chunk = chunks_map.get(doc_id)
        if chunk and _same_shop(chunk.get("metadata", {}).get("shop_id"), shop_id):
            item = dict(chunk)
            item["score"] = score
            matched.append(item)
        if len(matched) >= top_k * 3:
            break

    return rerank_chunks(query, matched, top_k, deadline)


def sync_blog_by_id(blog_id: int, blog_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """根据 Canal/MQ 消息增量同步单篇博客到 RAG 索引。"""
    global _chunks_map
    with _index_lock:
        if blog_data is None:
            payload = _fetch_json(f"{BLOG_SERVICE_URL}/blog/{blog_id}")
            blog_data = payload.get("data") if payload.get("success") else None

        chunks_map = _load_chunks_map()
        old_ids = [
            cid
            for cid, chunk in chunks_map.items()
            if chunk.get("metadata", {}).get("blog_id") == blog_id
        ]
        if old_ids:
            try:
                _get_vectorstore().delete(old_ids)
            except Exception as exc:
                logger.error("删除旧 chunk 失败: %s", exc)
            for cid in old_ids:
                chunks_map.pop(cid, None)

        added = 0
        if blog_data and (blog_data.get("content") or blog_data.get("title")):
            docs = _build_documents([blog_data])
            new_map = _build_chunks_map(docs)
            if new_map:
                _get_vectorstore().add_documents(docs)
                chunks_map.update(new_map)
                added = len(new_map)

        _chunks_map = chunks_map
        _save_chunks_map(chunks_map)

    return {
        "blog_id": blog_id,
        "deleted_chunks": len(old_ids),
        "added_chunks": added,
    }