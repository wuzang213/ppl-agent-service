"""上下文压缩探针：继承 SummarizationMiddleware，记录摘要前后的消息 token。"""

from typing import Any

from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware.types import AgentState, Runtime

compression_events: list[dict[str, int]] = []


def clear_compression_events() -> None:
    compression_events.clear()


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(str(text)) + 1) // 2)


def _count_messages(messages: Any) -> int:
    return sum(count_tokens(getattr(msg, "content", "")) for msg in messages or [])


def _count_state(state: Any) -> int:
    return _count_messages(state.get("messages", []) if isinstance(state, dict) else [])


class SummaryProbeMiddleware(SummarizationMiddleware):
    """在 SummarizationMiddleware 的 before_model 钩子外记录摘要前后 token。"""

    def before_model(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        before_tokens = _count_state(state)
        result = super().before_model(state, runtime)
        if result and result.get("messages") is not None:
            after_tokens = _count_messages(result["messages"])
            if after_tokens < before_tokens:
                compression_events.append(
                    {"before_tokens": before_tokens, "after_tokens": after_tokens}
                )
        return result

    async def abefore_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        before_tokens = _count_state(state)
        result = await super().abefore_model(state, runtime)
        if result and result.get("messages") is not None:
            after_tokens = _count_messages(result["messages"])
            if after_tokens < before_tokens:
                compression_events.append(
                    {"before_tokens": before_tokens, "after_tokens": after_tokens}
                )
        return result
