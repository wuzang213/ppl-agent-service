"""评评哩地生活推荐 Agent。"""

import json
import os
from typing import Any, NotRequired

import aiosqlite
from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
    dynamic_prompt,
)
from langchain.agents.middleware.types import ModelRequest
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.agents.hmdp_metrics import (
    ModelRetryMetricsMiddleware,
    begin_metrics,
    finish_metrics,
    save_metrics,
)
from app.agents.hmdp_mcp import get_mcp_tools
from app.agents.compression_probe import SummaryProbeMiddleware
from app.agents.hmdp_tools import TOOLS, reset_user_context, set_user_context
from app.models.schemas import RagAnswer
from app.common.logger import logger

load_dotenv()

# 评估专用严格模式：关闭推荐/建议/追问，避免干扰 RAG 指标
EVAL_STRICT = os.getenv("AGENT_EVAL_STRICT", "true").lower() == "true"

def _init_model():
    base_url = os.getenv("DASHSCOPE_BASE_URL")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("DASHSCOPE_BASE_URL 和 DASHSCOPE_API_KEY 未配置")
    return init_chat_model(
        model=os.getenv("AGENT_MODEL", "qwen-plus"),
        model_provider="openai",
        base_url=base_url,
        api_key=api_key,
        temperature=float(os.getenv("AGENT_TEMPERATURE", "0.1")),
    )


class HmdpState(AgentState):
    """Agent 状态，记录当前对话使用的模式。"""

    mode: NotRequired[str]


RECOMMEND_PROMPT = """
你是本地生活推荐助手，负责帮用户找店铺、看攻略、比优惠。

# 工作流程
1. 附近找店：必须调用 get_shop_types 确认类型，再调用 get_shops_nearby；不得用 get_shops_by_name 代替附近搜索。
2. 店铺攻略：调用 get_shop_detail 获取店铺信息，调用 get_shop_blogs 获取攻略。
3. 优惠信息：调用 get_shop_vouchers 获取优惠券。
4. 细节体验：用户问价格、酒水、停车、包间、服务态度等细节时，调用 get_shop_rag_context 检索该店铺博客片段。
5. 多店对比：用户要求对比多家店时，调用 get_shops_rag_context 一次检索多个店铺片段，再逐项对比。
6. 综合回答：结合距离、评分、人均价格、攻略口碑和博客片段给出结构化推荐，并说明理由。

# 规则
- 必须优先使用工具获取真实数据，禁止编造店铺、评分、距离、销量、营业时间等片段外信息。
- 当用户提到“附近”或提供坐标时，第一工具必须调用 get_shops_nearby，禁止先调用 get_shops_by_name；只有用户明确问某个具体店名时才允许使用 get_shops_by_name。
- 回答附近店铺时必须包含距离；如果距离不可用，明确说明“距离暂未获取”。
- 营业时间以 get_shop_detail 的官方信息为准，博客信息只作参考；两者冲突时优先官方数据。
- 用户问“是否支持 XX”且检索没有答案时，只回答“根据现有博客暂未查到”，不要输出地址、人均、评分、销量等无关信息。
- 只能依据工具返回内容回答；工具未提供的信息一律不得推断、补充、脑补。
- 禁止使用“未提及=不存在”“未提及=一致”“无负面反馈=体验好”这类推测。
- 检索片段没有答案时，只回答“根据现有博客暂未查到”，然后停止，不要继续推测。
- 不要复制或转述检索原文，只输出提炼后的事实结论。
- 不要输出“如需进一步查询”“需要我帮您”等追问和建议，除非用户主动要求。
- 对比店铺时，先调用 get_shops_rag_context，只基于两家店各自返回的片段逐项对比。
- 用户位置由请求自动注入到上下文中，不要反复向用户要坐标。
- 某类数据查询失败时，明确告诉用户这部分暂时不可用，继续用其余数据回答。

# 输出格式
最终回答必须只输出如下 JSON，不要输出其他内容，不要输出 Markdown 代码块：
{
  "answer": "提炼后的事实结论，自然语言，不含店铺X：或标题：等标签",
  "source": ["博客标题1", "博客标题2"]
}

如果检索不到答案，answer 写“根据现有博客暂未查到”，source 为空数组 []。

# 输出示例
{"answer":"停车很方便，门口有车位。","source":["某某探店博客"]}
{"answer":"根据现有博客暂未查到","source":[]}
{"answer":"店铺A环境安静雅致；店铺B暂未查到环境相关描述。A店更安静。","source":["探店文章A","探店文章B"]}
"""

DAILY_PROMPT = """
你是用户的日常智能助手，可以正常聊天，也可以根据用户问题调用店铺、
博客、优惠券、时间、RAG 检索等工具。

# 规则
- 工具结果只是辅助，不要强制套用推荐模板。
- 用户问本地生活问题时优先使用工具获取真实数据，禁止编造。
- 只能依据工具返回内容回答，未提供的信息不推断、不补充。
- 检索不到时只回答“暂未查到”，不要脑补、不要追加推测。
- 不要复制检索原文，不要输出“是否需要我继续查询”之类的追问。
- 回答用中文，自然、简洁、清晰。
- 最终回答必须输出 JSON：{"answer": "最终回答", "source": ["来源标题"]}
"""


def _strict_output_rules() -> str:
    return """
# 测试模式输出要求
- 只输出检索片段中明确存在的事实，不输出建议、推荐结论或追问。
- 只回答用户问题对应的信息，不要附带其他事实。
- 禁止输出“店铺X：”“《标题》”“标题：”“正文：”等工具原文片段。
- 最终回答必须输出 JSON：{"answer": "事实结论", "source": ["博客标题"]}。
- 禁止把“店铺X：”“《标题》”“标题：”“正文：”等工具原文片段写进 answer。
- 无答案时 answer 写“根据现有博客暂未查到”，source 为空数组。
"""


@dynamic_prompt
def mode_dynamic_prompt(request: ModelRequest) -> str:
    mode = request.state.get("mode", "recommend")
    prompt = RECOMMEND_PROMPT if mode == "recommend" else DAILY_PROMPT
    if EVAL_STRICT:
        prompt += _strict_output_rules()
    return prompt


def _serialize(obj: Any) -> Any:
    if hasattr(obj, "value"):
        return _serialize(obj.value)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, (list, tuple)):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


class HmdpAgent:

    def __init__(self) -> None:
        self.conn: aiosqlite.Connection | None = None
        self.checkpointer: AsyncSqliteSaver | None = None
        self.model = None
        self.agent = None

    async def init(self) -> None:
        self.model = _init_model()
        await self._init_checkpointer()
        mcp_tools = await get_mcp_tools()
        self.agent = create_agent(
            model=self.model,
            tools=[*TOOLS, *mcp_tools],
            checkpointer=self.checkpointer,
            state_schema=HmdpState,
            middleware=[
                mode_dynamic_prompt,
                ModelRetryMetricsMiddleware(),
                ToolRetryMiddleware(max_retries=1),
                ToolCallLimitMiddleware(run_limit=5, exit_behavior="continue"),
                SummaryProbeMiddleware(
                    model=self.model,
                    trigger=("messages", int(os.getenv("AGENT_SUMMARY_TRIGGER", "24"))),
                    keep=("messages", int(os.getenv("AGENT_SUMMARY_KEEP", "6"))),
                ),
            ],
        )
        logger.info("hmdp agent 初始化完成，工具数量: %s", len(TOOLS) + len(mcp_tools))

    async def _init_checkpointer(self) -> None:
        os.makedirs("db", exist_ok=True)
        self.conn = await aiosqlite.connect("db/hmdp_agent.db")
        self.checkpointer = AsyncSqliteSaver(conn=self.conn)
        await self.checkpointer.setup()

    async def close(self) -> None:
        if self.conn is not None:
            await self.conn.close()
        logger.info("hmdp agent 连接已关闭")

    async def extract_answer(self, text: str) -> tuple[str, list[str]]:
        """把模型回答整理成 {answer, source}，只取 answer 作为最终回答。"""
        text = (text or "").strip()
        if not text:
            return "", []

        try:
            cleaned = text
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            data = json.loads(cleaned)
            if isinstance(data, dict) and data.get("answer"):
                answer = str(data["answer"]).strip()
                source = [str(item) for item in (data.get("source") or [])]
                return answer, source
        except json.JSONDecodeError:
            pass

        try:
            structured_model = self.model.with_structured_output(RagAnswer)
            result = await structured_model.ainvoke([
                SystemMessage(
                    "你只负责整理回答。输入可能包含检索片段。"
                    "输出 JSON，answer 必须是提炼后的事实结论，"
                    "不能包含店铺X、标题、正文等标签，source 只放博客标题。"
                ),
                HumanMessage(text),
            ])
            if isinstance(result, dict):
                answer = str(result.get("answer") or "").strip()
                source = [str(item) for item in (result.get("source") or [])]
            else:
                answer = (result.answer or "").strip()
                source = result.source or []
            return answer, source
        except Exception as exc:
            logger.error("结构化回答提取失败，返回原始内容: %s", exc)
            return text, []
    async def generate_sse(
        self,
        thread_id: str,
        message: str,
        user_id: str = "",
        x: float | None = None,
        y: float | None = None,
        mode: str = "recommend",
        interrupt_decision: dict[str, Any] | None = None,
    ):
        tokens = set_user_context(user_id, x, y)
        metrics_token = begin_metrics(thread_id, user_id, mode)
        config = {"configurable": {"thread_id": thread_id}}
        _input = {
            "messages": [HumanMessage(content=message)],
            "mode": mode,
        }
        if interrupt_decision:
            _input = Command(resume={"decisions": [interrupt_decision]})

        success = True
        error = None
        full_parts: list[str] = []
        interrupted = False
        try:
            async for chunk in self.agent.astream(
                _input,
                config=config,
                stream_mode=["messages", "updates"],
                version="v2",
            ):
                event_type = chunk["type"]
                data = chunk["data"]

                if event_type == "messages":
                    token, _ = data
                    content = getattr(token, "content", None)
                    if content:
                        # ----- 新增类型转换 -----
                        if isinstance(content, list):
                            # 将列表中的元素转为字符串并拼接（过滤 None）
                            content = ''.join(str(item) for item in content if item is not None)
                        elif not isinstance(content, str):
                            content = str(content)
                        # ----- 转换结束 -----
                        full_parts.append(content)
                elif event_type == "updates" and "__interrupt__" in data:
                    interrupted = True
                    yield {
                        "event": "interrupt",
                        "data": json.dumps(
                            {
                                "type": "interrupt",
                                "interrupt": {
                                    "reason": "需要人工确认",
                                    "details": _serialize(data["__interrupt__"]),
                                },
                            },
                            ensure_ascii=False,
                            default=str,
                        ),
                    }

            if not interrupted and full_parts:
                answer, source = await self.extract_answer("".join(full_parts))
                yield {
                    "event": "message",
                    "data": json.dumps(
                        {
                            "type": "message",
                            "content": answer,
                            "source": source,
                        },
                        ensure_ascii=False,
                    ),
                }

            yield {
                "event": "done",
                "data": json.dumps(
                    {"type": "done", "content": "处理完成"},
                    ensure_ascii=False,
                ),
            }
        except Exception as exc:
            success = False
            error = str(exc)
            logger.error(f"hmdp SSE 流中断: {exc}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps(
                    {"type": "error", "error": str(exc)},
                    ensure_ascii=False,
                ),
            }
        finally:
            reset_user_context(tokens)
            metrics = finish_metrics(metrics_token, success=success, error=error)
            await save_metrics(metrics)
    async def get_messages(self, thread_id: str) -> dict[str, Any]:
        state = await self.agent.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        if state is None or not state.values:
            return {"messages": []}

        result = []
        for msg in state.values.get("messages", []):
            if not getattr(msg, "content", None):
                continue
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
        return {"messages": result}

    async def clear_messages(self, thread_id: str) -> None:
        await self.checkpointer.adelete_thread(thread_id)


hmdp_agent = HmdpAgent()

__all__ = ["hmdp_agent"]

