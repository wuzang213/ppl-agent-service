"""基于 LangGraph 的节点编排实现。

分层结构：
    第一层  入口：trim_history -> entry_router（意图识别 + 实体抽取，便宜模型）
    第二层  分流：daily_chat（闲聊直出）| grounding（推荐模式实体解析）
    第三层  扇出：nearby_worker（定位）-> shop_worker（详情/优惠/细节/对比，Send 并行）
    第四层  汇总：summarize（主模型）-> END
"""

from app.agents.graph.workflow import build_graph

__all__ = ["build_graph"]
