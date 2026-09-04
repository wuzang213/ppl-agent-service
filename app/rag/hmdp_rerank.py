"""Cross-Encoder 重排（未安装 sentence-transformers 时自动回退 RRF 顺序）。"""

import os
from typing import Any

from app.common.logger import logger

RAG_RERANK = os.getenv("RAG_RERANK", "true").lower() == "true"
RAG_RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")

_model: Any | None = None
_model_checked = False


def _get_model() -> Any | None:
    global _model, _model_checked
    if not RAG_RERANK or _model_checked:
        return _model
    _model_checked = True
    try:
        import torch
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(
            RAG_RERANK_MODEL,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        logger.info("Reranker 模型加载完成: %s", RAG_RERANK_MODEL)
    except Exception as exc:
        logger.warning("Reranker 不可用，继续使用 RRF 排序: %s", exc)
        _model = None
    return _model


def rerank_chunks(query: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    model = _get_model()
    if model is None or not chunks:
        return chunks[:top_k]
    try:
        scores = model.predict([(query, chunk.get("content") or "") for chunk in chunks])
        ranked = sorted(
            zip(chunks, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        positive = [chunk for chunk, score in ranked if float(score) > 0]
        return (positive or [chunk for chunk, _ in ranked])[:top_k]
    except Exception as exc:
        logger.error("Rerank 执行失败，回退 RRF 排序: %s", exc)
        return chunks[:top_k]