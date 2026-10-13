"""降级文案的**单点真源**（`specs/0015` T3）：新增降级路径只能从此处 import，禁止就地写字面量（由 `test_degrade_single_source.py` 看住）；本模块不含 `NEARBY_TYPE_UNKNOWN_NOTICE` / `NEARBY_NO_MORE`（那是用户侧正常分支文案，与"服务降级"根因不同）。"""

#: 任务 → 该块在记录里的 `*_error` 字段名（降级说明写回哪里）。
DEGRADED_ERROR_FIELD: dict[str, str] = {
    "detail": "blogs_error",
    "voucher": "vouchers_error",
    "rag": "chunks_error",
}

#: 任务 → 用户向的说法（只用于拼降级文案，不含内部字段名）。
TASK_LABEL = {"detail": "详情", "voucher": "代金券", "rag": "攻略", "nearby": "位置"}

#: `RAG_TIMEOUT` 被截断时写进 `chunks_error` 的文案（真机 grep 用，**别改字面量**）。
RAG_TIMEOUT_MSG = "细节检索未在预算内完成"

#: 用户向的降级/兜底文案；与用户侧的 `NEARBY_TYPE_UNKNOWN_NOTICE`（用户没说清）必须分开，本句表示服务暂不可用 ⇒ 说服务忙、可重试。
#: 不含内部实现细节（不提模型/额度/接口）；同一根因（入口降级）全项目共用这一句，换节点也不许另起一句。
ROUTE_DEGRADED_NOTICE = "抱歉，我这边刚好有点忙，没能理解你的问题，请稍后再试一次。"


def degraded_block_error(task: str) -> str:
    """熔断短路时该块的**用户向**降级说明（不含内部错误细节）。"""
    return f"这家店的{TASK_LABEL.get(task, task)}暂时查不到，我先用已拿到的信息回答。"


#: 旧名（`nodes` 里用的是带下划线的两个名字）：保留别名是为了让迁移**只改 import**，
#: 调用点一行都不用动 —— 纯重构不该顺手改十几个调用点。
_degraded_block_error = degraded_block_error
_DEGRADED_ERROR_FIELD = DEGRADED_ERROR_FIELD
_TASK_LABEL = TASK_LABEL


__all__ = [
    "DEGRADED_ERROR_FIELD", "TASK_LABEL", "RAG_TIMEOUT_MSG", "ROUTE_DEGRADED_NOTICE",
    "degraded_block_error",
    "_degraded_block_error", "_DEGRADED_ERROR_FIELD", "_TASK_LABEL",
]
