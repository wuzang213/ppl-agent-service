"""基于 LangGraph 的节点编排：trim_history -> entry_router（便宜模型）-> daily_chat | grounding
-> nearby_worker -> shop_worker（Send 并行）-> summarize_answer（逐 token 流式）| summarize（非流式回落）。"""

from app.agents.graph.workflow import build_graph

__all__ = ["build_graph"]
