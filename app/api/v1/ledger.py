"""运行账本查询（specs/0013 T4）：`GET /api/v1/ledger/runs/{run_id}` 一次返回 run 头 + 全部 LLM 事件 + 全部工具事件（按时间升序）；含运行细节不应对公网开放，如需对外须登记网关 internalPaths 黑名单（待人类确认）。"""

from typing import Any

from fastapi import APIRouter, HTTPException

from app.agents.ledger import fetch_run
from app.common.logger import logger

router = APIRouter()


@router.get("/ledger/runs/{run_id}", tags=["运行账本"])
async def get_run_ledger(run_id: str) -> dict[str, Any]:
    """取一次请求的完整事件流。`run_id` 不存在 → 404。"""
    # 长度上限：DB 列是 varchar(64)，超长的直接挡掉，别让它走到 SQL
    if not run_id or len(run_id) > 64:
        raise HTTPException(status_code=400, detail="run_id 非法")
    try:
        data = await fetch_run(run_id)
    except Exception as exc:  # noqa: BLE001 - 查询接口可以报错，但要转成 500 语义
        logger.error("查询运行账本失败 run_id=%s: %s", run_id, exc)
        raise HTTPException(status_code=500, detail="查询运行账本失败") from exc
    if data is None:
        raise HTTPException(status_code=404, detail="run_id 不存在")
    return data
