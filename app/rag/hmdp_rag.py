"""博客 RAG 索引与检索（混合检索：向量 + BM25 + RRF）。"""

import json
import logging
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

from app.common.logger import logger
from app.rag.hmdp_rerank import rerank_chunks

load_dotenv()

BLOG_SERVICE_URL = os.getenv("BLOG_SERVICE_URL", "").rstrip("/")
RAG_DB_DIR = Path("db/rag")
CHROMA_DIR = str(RAG_DB_DIR / "chroma")
CHUNKS_JSON = str(RAG_DB_DIR / "chunks.json")
BM25_PATH = str(RAG_DB_DIR / "bm25_index.bm25")
COLLECTION_NAME = "hmdp_blog"
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-v4")
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "400"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
BLOG_PAGE_SIZE = int(os.getenv("BLOG_PAGE_SIZE", "100"))
RAG_HYBRID = os.getenv("RAG_HYBRID", "true").lower() == "true"
RAG_EXPAND_QUERY = os.getenv("RAG_EXPAND_QUERY", "true").lower() == "true"
RAG_BM25_REBUILD_DELAY = int(os.getenv("RAG_BM25_REBUILD_DELAY", "5"))

_embeddings: DashScopeEmbeddings | None = None
_vectorstore: Chroma | None = None
_bm25: Any | None = None
_chunks_map: dict[str, dict[str, Any]] | None = None
_index_lock = threading.Lock()
_bm25_timer: threading.Timer | None = None

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
    global _vectorstore
    if _vectorstore is None:
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


def _build_bm25_index(chunks: list[dict[str, Any]]) -> Any | None:
    if not RAG_HYBRID:
        return None
    try:
        import bm25s
        import jieba

        logging.getLogger("jieba").setLevel(logging.ERROR)
        corpus = [
            {"id": c["id"], "content": c["content"], "metadata": c["metadata"]}
            for c in chunks
        ]
        corpus_tokens = [list(jieba.cut(c["content"])) for c in corpus]
        retriever = bm25s.BM25(k1=1.5, b=0.75, corpus=corpus)
        retriever.index(corpus_tokens)
        retriever.save(BM25_PATH)
        return retriever
    except Exception as exc:
        logger.error("BM25 索引构建失败，降级为向量检索: %s", exc)
        return None


def _get_bm25() -> Any | None:
    global _bm25
    if _bm25 is None and Path(BM25_PATH).exists():
        try:
            import bm25s

            _bm25 = bm25s.BM25.load(BM25_PATH, load_corpus=True)
        except Exception as exc:
            logger.error("BM25 索引加载失败: %s", exc)
            _bm25 = None
    return _bm25


def _bm25_search(query: str, shop_id: int, k: int) -> list[tuple[str, float]]:
    retriever = _get_bm25()
    if retriever is None:
        return []
    try:
        import jieba

        tokens = [list(jieba.cut(query))]
        corpus_size = len(retriever.corpus) if getattr(retriever, "corpus", None) else 0
        request_k = min(k * 4, corpus_size) if corpus_size > 0 else k
        if request_k <= 0:
            return []
        results, scores = retriever.retrieve(tokens, k=request_k)
        items = []
        for i in range(results.shape[1]):
            doc = results[0, i]
            if doc is None:
                continue
            meta = doc.get("metadata") or {}
            if _same_shop(meta.get("shop_id"), shop_id):
                items.append((doc.get("id"), float(scores[0, i])))
        return items[:k]
    except Exception as exc:
        logger.error("BM25 检索失败: %s", exc)
        return []


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
    """全量重建混合检索索引。"""
    global _chunks_map, _bm25
    blogs = _fetch_all_blogs()
    docs = _build_documents(blogs)
    chunks_map = _build_chunks_map(docs)
    _save_chunks_map(chunks_map)
    _chunks_map = chunks_map

    vectorstore = _get_vectorstore()
    vectorstore.reset_collection()
    if docs:
        vectorstore.add_documents(docs)

    _bm25 = None
    bm25_ready = _build_bm25_index(list(chunks_map.values()))
    _bm25 = bm25_ready

    logger.info(
        "RAG 混合索引重建完成，博客数: %s，chunk 数: %s，hybrid: %s",
        len(blogs),
        len(docs),
        bm25_ready is not None,
    )
    return {
        "total_blogs": len(blogs),
        "total_chunks": len(docs),
        "hybrid": bm25_ready is not None,
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
        "bm25_path": BM25_PATH,
        "hybrid": RAG_HYBRID,
        "query_expansion": RAG_EXPAND_QUERY,
        "blog_page_size": BLOG_PAGE_SIZE,
    }
def retrieve_shop_rag_chunks(shop_id: int, query: str, top_k: int = RAG_TOP_K) -> list[dict[str, Any]]:
    """返回店铺博客相关 chunk（向量 + BM25 + RRF + 可选重排）。"""
    expanded_query = _expand_query(query)
    dense_results = _dense_search(shop_id, query, top_k * 3)
    ranked_lists = [[doc_id for doc_id, _ in dense_results]]
    if RAG_HYBRID:
        bm25_results = _bm25_search(expanded_query, shop_id, top_k * 3)
        ranked_lists.append([doc_id for doc_id, _ in bm25_results])

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

    return rerank_chunks(query, matched, top_k)


def retrieve_shop_rag_context(shop_id: int, query: str, top_k: int = RAG_TOP_K) -> str:
    """检索某店铺博客中与 query 相关的文本片段（向量 + BM25 + RRF + 可选重排）。"""
    if _count_chunks() == 0:
        return "RAG 知识库为空，请先调用重建接口"

    chunks = retrieve_shop_rag_chunks(shop_id, query, top_k)
    if not chunks:
        return f"店铺 {shop_id} 暂无与「{query}」相关的博客内容"

    lines = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source = (
            f"[博客：{meta.get('title') or '无标题'} | "
            f"作者：{meta.get('author') or '未知'} | "
            f"发布时间：{meta.get('create_time') or '未知'}]"
        )
        lines.append(f"{source}\n{chunk.get('content')}")
    return "\n\n".join(lines)

def _schedule_bm25_rebuild(delay: int = RAG_BM25_REBUILD_DELAY) -> None:
    """合并多次博客变更，延迟重建一次 BM25。"""
    global _bm25_timer
    if _bm25_timer is not None and _bm25_timer.is_alive():
        return
    _bm25_timer = threading.Timer(delay, _rebuild_bm25_now)
    _bm25_timer.daemon = True
    _bm25_timer.start()


def _rebuild_bm25_now() -> None:
    global _bm25
    with _index_lock:
        _bm25 = _build_bm25_index(list(_load_chunks_map().values()))


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

    _schedule_bm25_rebuild()
    return {
        "blog_id": blog_id,
        "deleted_chunks": len(old_ids),
        "added_chunks": added,
    }