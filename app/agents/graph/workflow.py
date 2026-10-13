"""图构建：条件边做主骨架，Send 做局部优化。"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from app.agents.graph.nodes import (
    SUMMARY_NODE,
    booking_confirm,
    capability_chat,
    clarify,
    clarify_slots,
    daily_chat,
    entry_router,
    grounding,
    itinerary,
    nearby_worker,
    plan_slots,
    plan_worker,
    react_loop,
    replace_stop,
    route_after_fanout,
    route_after_grounding,
    route_after_nearby,
    shop_worker,
    suggest,
    summarize_answer,
    trim_history,
)
from app.agents.graph.state import HmdpState

# 允许作为 Send 目标 / 条件边返回值的节点。
# `SUMMARY_NODE` 从 `nodes.py` 唯一导入，不要在此写字面量。
# `booking_confirm` 必须在列，否则条件边查表 `KeyError` 直接 500（店名靠 grounding 解析，见 plans/0027）。
_FANOUT_TARGETS = ["nearby_worker", "shop_worker", SUMMARY_NODE, "clarify", "booking_confirm"]


def build_graph(checkpointer=None):
    """
    构建并编译 Agent 图。
    """
    workflow = StateGraph(HmdpState)

    workflow.add_node("trim_history", trim_history)
    workflow.add_node("entry_router", entry_router)
    workflow.add_node("daily_chat", daily_chat)
    workflow.add_node("capability_chat", capability_chat)
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
    workflow.add_node(SUMMARY_NODE, summarize_answer)
    workflow.add_node("react_loop", react_loop)
    # 计划轨（`specs/0012`）：`plan_slots` 抽槽 → `clarify_slots` 澄清。
    # 两者都用 Command(goto=…) 自行决定去向（`clarify_slots` 可 goto 自身再挂起、或弃权回
    # `entry_router`），故**不需要**静态条件边 —— 与 `entry_router` 同款。
    workflow.add_node("plan_slots", plan_slots)
    workflow.add_node("clarify_slots", clarify_slots)
    # 计划轨的取数与编排（`specs/0012` §12.1/§12.2）：`plan_worker` 并发取候选（**不用 Send**，
    # 见 `nodes.plan_worker` 的说明），`itinerary` 编排，`suggest` 产建议项并收尾。
    # `replace_stop`（§12.5）由 `entry_router` 确定性短路进入，只重排一站。
    workflow.add_node("plan_worker", plan_worker)
    workflow.add_node("itinerary", itinerary)
    workflow.add_node("suggest", suggest)
    workflow.add_node("replace_stop", replace_stop)
    # 预约写路径的 HITL（`plans/0019`）：全图**第二个** `interrupt()` 点，与 `clarify_slots` 互不重叠
    # （澄清在缺槽时进；预约确认需"有行程 + 有登录态 + 点了建议项"）。由 `entry_router`
    # 确定性短路进入 —— 写路径的入口判定绝不交给模型。
    workflow.add_node("booking_confirm", booking_confirm)
    workflow.add_edge(START, "trim_history")
    workflow.add_edge("trim_history", "entry_router")
    # entry_router 通过 Command(goto=...) 自行决定去向（daily_chat / capability_chat /
    # grounding / plan_slots），无需条件边

    workflow.add_edge("daily_chat", END)
    workflow.add_edge("capability_chat", END)
    workflow.add_conditional_edges("grounding", route_after_grounding, _FANOUT_TARGETS)
    workflow.add_conditional_edges(
        "nearby_worker", route_after_nearby, ["shop_worker", SUMMARY_NODE]
    )
    # ReAct 补跳是固定编排之后的第二跳，条件边源节点必须是 shop_worker，**不能在 entry_router 之后**：
    # `_react_candidates` 取第 1 批 `worker_outputs` 里带 `id` 的记录，挂在入口则进环时候选为空、白烧
    # `REACT_MAX_STEPS` 次模型调用。是否进环由 `plan.target_depends_on_result` 决定，与边挂在哪儿无关。
    workflow.add_conditional_edges(
        "shop_worker", route_after_fanout, ["react_loop", SUMMARY_NODE]
    )
    workflow.add_edge("react_loop", SUMMARY_NODE)
    workflow.add_edge(SUMMARY_NODE, END)
    workflow.add_edge("clarify", END)
    # `plan_worker` 返回**普通 dict**（不是 Command）⇒ 必须显式给出边，否则分支会**静默终止**
    #    （图不报错、`state.next` 直接变空、后续节点一个都不跑 —— 排查时毫无线索）。
    workflow.add_edge("plan_worker", "itinerary")
    # 计划轨终态：`suggest` 已自带摘要与状态收尾（`_terminal_cleanup`），**不再接 `SUMMARY_NODE`**
    # —— 后者会无条件用取数结果重写 `answer`，把行程摘要冲掉（见 `nodes.suggest` 的说明）。
    workflow.add_edge("suggest", END)
    # `booking_confirm` 正常路径以 `Command(goto=END)` 收尾，但**兜底必须给静态边**：
    #    与 `plan_worker` 同理，返回 dict 的分支没有边会静默终止。
    workflow.add_edge("booking_confirm", END)

    return workflow.compile(checkpointer=checkpointer, name="hmdp-agent")


__all__ = ["build_graph", "SUMMARY_NODE"]
