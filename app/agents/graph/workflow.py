"""图构建：条件边做主骨架，Send 做局部优化。"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from app.agents.graph.nodes import (
    clarify,
    daily_chat,
    dispatch_workers,
    entry_router,
    grounding,
    nearby_worker,
    route_after_grounding,
    route_after_nearby,
    shop_worker,
    summarize,
    trim_history,
)
from app.agents.graph.state import HmdpState

# 允许作为 Send 目标 / 条件边返回值的节点
_FANOUT_TARGETS = ["nearby_worker", "shop_worker", "summarize", "clarify"]


def build_graph(checkpointer=None):
    """
    构建并编译 Agent 图。
    """
    workflow = StateGraph(HmdpState)

    workflow.add_node("trim_history", trim_history)
    workflow.add_node("entry_router", entry_router)
    workflow.add_node("daily_chat", daily_chat)
    workflow.add_node("grounding", grounding)
    workflow.add_node(
        "nearby_worker",
        nearby_worker,
        retry_policy=RetryPolicy(max_attempts=2, initial_interval=0.5),
    )
    workflow.add_node(
        "shop_worker",
        shop_worker,
        retry_policy=RetryPolicy(max_attempts=2, initial_interval=0.5),
    )
    workflow.add_node("clarify", clarify)
    workflow.add_node("summarize", summarize)

    workflow.add_edge(START, "trim_history")
    workflow.add_edge("trim_history", "entry_router")
    # entry_router 通过 Command(goto=...) 自行决定去向，无需条件边

    workflow.add_edge("daily_chat", END)
    workflow.add_conditional_edges("grounding", route_after_grounding, _FANOUT_TARGETS)
    workflow.add_conditional_edges(
        "nearby_worker", route_after_nearby, ["shop_worker", "summarize"]
    )
    workflow.add_edge("shop_worker", "summarize")
    workflow.add_edge("summarize", END)
    workflow.add_edge("clarify", END)

    return workflow.compile(checkpointer=checkpointer, name="hmdp-agent")


__all__ = ["build_graph", "dispatch_workers"]
