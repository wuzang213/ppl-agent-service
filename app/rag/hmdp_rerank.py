"""Cross-Encoder 重排（未安装 sentence-transformers 时自动回退 RRF 顺序）。
本模块是"可选增强"，只允许影响排序顺序，**绝不允许**在请求路径上联网（见 `MODEL_LOAD_RESERVE_S` 与
`local_files_only=True`），也**绝不允许**把 `chunks` 变空（失败 / 超预算一律返回调用方给的候选顺序）。"""

import os
import time
from typing import Any

from dotenv import load_dotenv

from app.common.config import require_env
from app.common.logger import logger

load_dotenv()

RAG_RERANK = os.getenv("RAG_RERANK", "false").lower() == "true"
# 重排模型是"用哪个模型"的标识符，必须来自 env、缺失即 fail-fast（不再静默回退 "Qwen/Qwen3-Reranker-0.6B"，
#    漏配会悄无声息用错模型）；`.env` 已配置 `RAG_RERANK_MODEL`。
#    本模块显式调 `load_dotenv()` 保证自包含、且 `require_env` 在 .env 加载之后才执行（load_dotenv 幂等）。
RAG_RERANK_MODEL = require_env("RAG_RERANK_MODEL")

#: 开始加载模型至少要留出的预算（秒），即 `MODEL_LOAD_RESERVE_S`。
#: 实测：已导入 torch / sentence-transformers 时加载只要 1.0s；未导入时导入本身要 ≈14s（torch 2.6s + s-t 11.7s）。
#: 懒加载一旦开始就在 `asyncio.to_thread` 里**不可被 `wait_for` 取消** ⇒ 预算不够时宁可用 RRF，也不要开始一个跑不完的加载。
MODEL_LOAD_RESERVE_S = 1.5

_model: Any | None = None
_model_checked = False


def _remaining_of(deadline: float | None) -> float | None:
    """距绝对截止时刻还剩多少秒；`deadline=None` ⇒ `None`（表示"无预算"）。"""
    if deadline is None:
        return None
    return deadline - time.monotonic()


def _get_model(deadline: float | None = None) -> Any | None:
    global _model, _model_checked
    if not RAG_RERANK or _model_checked:
        return _model
    left = _remaining_of(deadline)
    if left is not None and left < MODEL_LOAD_RESERVE_S:
        # 预算不足 ⇒ **不置 `_model_checked`**（这不是"失败"，是"本轮不做"），
        #    下一轮预算够时仍会尝试加载。
        logger.warning(
            "本轮预算仅剩 %.2fs（< %.1fs 加载储备），跳过重排模型加载，本轮用 RRF 排序",
            left,
            MODEL_LOAD_RESERVE_S,
        )
        return None
    _model_checked = True
    try:
        import torch
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(
            RAG_RERANK_MODEL,
            device="cuda" if torch.cuda.is_available() else "cpu",
            # **绝不在请求路径上联网**：不传这个参数时 `CrossEncoder.__init__` 会经 `hf_hub_download`
            # **联网校验**本地缓存 —— 即使 1.2G 缓存完整，经本机代理返回 502 后库内重试+退避会把首访放大到 80.2s；
            # `local_files_only=True` 实测 0.93s（在线 ≈66s）。**不要**改回 `os.environ.setdefault("HF_HUB_OFFLINE", "1")`：它是导入期常量，属永不生效的死代码。
            local_files_only=True,
        )
        logger.info("Reranker 模型加载完成: %s", RAG_RERANK_MODEL)
    except Exception as exc:
        logger.warning("Reranker 不可用，继续使用 RRF 排序: %s", exc)
        _model = None
    return _model


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    """可选重排：**只影响排序，绝不影响 `chunks` 的有无**。
    `deadline`（`time.monotonic()` 绝对时刻，由 `nodes._deadline_after` / `nodes.RAG_TIMEOUT` 派生）：预算不足加载
    （见 `_get_model`）或已耗尽 ⇒ 原样返回 `chunks[:top_k]`（RRF 顺序）；`model.predict` 一旦开始不可中断，故只在开始前把关。"""
    model = _get_model(deadline)
    if model is None or not chunks:
        return chunks[:top_k]
    left = _remaining_of(deadline)
    if left is not None and left <= 0:
        logger.warning(
            "重排预算已耗尽（剩 %.2fs），本轮用 RRF 顺序（%s 条候选）", left, len(chunks)
        )
        return chunks[:top_k]
    try:
        scores = model.predict([(query, chunk.get("content") or "") for chunk in chunks])
        ranked = sorted(
            zip(chunks, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        # 用相对阈值（score >= max_score * 0.5）替代硬阈值 score > 0
        # 硬阈值 score > 0 会丢掉所有负分但相关的结果；相对阈值保留与最高分相近的结果
        max_score = float(ranked[0][1]) if ranked else 0.0
        threshold = max_score * 0.5
        positive = [chunk for chunk, score in ranked if float(score) >= threshold]
        return (positive or [chunk for chunk, _ in ranked])[:top_k]
    except Exception as exc:
        logger.error("Rerank 执行失败，回退 RRF 排序: %s", exc)
        return chunks[:top_k]