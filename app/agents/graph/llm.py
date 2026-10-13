"""模型工厂：主模型负责汇总，轻量模型负责意图识别与日常闲聊。`specs/0014` 后新增计划轨独立模型（`get_planner_model`）与主/轻量各自的可选备用模型（fallback 链，端点抖动自动切换）。模型标识符与连接信息一律来自 env，不得在代码里写死（漏配 `.env` 会悄无声息用错模型）。"""

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from app.common.config import require_env
from app.common.logger import logger

load_dotenv()

# 模型标识符（"用哪个模型"）必须来自 env，缺失即 fail-fast —— 不再静默回退到代码里的
#    "qwen-plus" / "qwen-flash"（漏配 .env 会悄无声息调错模型，且全程不报错）。
#    .env 已配置 AGENT_MODEL / AGENT_LIGHT_MODEL，移除默认值不会让部署环境缺值。
DEFAULT_MAIN_MODEL = require_env("AGENT_MODEL")
DEFAULT_LIGHT_MODEL = require_env("AGENT_LIGHT_MODEL")
#: 抽槽（`SlotFill`）专用模型标识符。**留空 = 跟主模型走**。
#: 抽槽必须留在主模型：轻量模型在抽槽通道实测 >25s 卡死、乱填 0，其余五个通道均可换快模型（快 10~30 倍）。
#: 用 `or` 而非 `os.getenv(k, default)`：键存在但值为空串时默认值不生效（同 `AGENT_IP`）。
DEFAULT_SLOT_MODEL = (os.getenv("AGENT_SLOT_MODEL") or "").strip() or DEFAULT_MAIN_MODEL
#: 计划轨**编排**专用模型标识符（`specs/0014` T1）。**留空 = 回落 `DEFAULT_LIGHT_MODEL`**（用 `or` 兜底空串）：planner 是可选增强，不配即等于现状。
#: 若该模型不支持 `with_structured_output`，调用点会失败并回落规则兜底（见 `_ask_itinerary`）。
DEFAULT_PLANNER_MODEL = (os.getenv("AGENT_PLANNER_MODEL") or "").strip() or DEFAULT_LIGHT_MODEL
#: 主/轻量模型的**备用**模型标识符（`specs/0014` T4）。留空 ⇒ 不构造备用，退回单端点（现状）。
DEFAULT_FALLBACK_MODEL = (os.getenv("AGENT_FALLBACK_MODEL") or "").strip()
DEFAULT_FALLBACK_LIGHT_MODEL = (os.getenv("AGENT_FALLBACK_LIGHT_MODEL") or "").strip()
# 温度属纯调参（非标识符），保留安全默认值：缺省时 0.1 是通用合理值，且 .env 已显式给出。
TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.1"))

#: 是否让模型开启**思考模式**（默认**关**）。
#: 思考模式是"回复慢"的主因（实测开=13.16s / 821 reasoning token，关=2.14s / 0），思考 token 不产出正文却占时间，关掉后答案无实质差异。
#: 用**调用参数**下发而非换模型（模型标识符全来自 env），空串用 `or` 兜底（同 `AGENT_SLOT_MODEL`）。
THINKING_ENABLED = (os.getenv("AGENT_ENABLE_THINKING") or "false").strip().lower() == "true"


def _build_one(model: str, temperature: float, base_url: str = "", api_key: str = ""):
    """构造一个**单端点**的 chat model（fallback 链的每一节都走这里）。"""
    base_url = base_url or os.getenv("DASHSCOPE_BASE_URL")
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("DASHSCOPE_BASE_URL 和 DASHSCOPE_API_KEY 未配置")
    return init_chat_model(
        model=model,
        model_provider="openai",
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        # 防止 dashscope 偶发挂起时整个请求卡死（默认 600s×2 重试最长可阻塞 30 分钟）。
        # 单次调用最多等 60s，失败仅重试 1 次；超时由调用方自行兜底。
        timeout=60,
        max_retries=1,
        # 流式调用**显式**索要 usage（`specs/0013` §11.1）：不下发时"给不给 usage"取决于模型名，轻量模型（六通道共用）的流式聚合体 usage_metadata=None，导致 daily 轨压缩锚点失效、账本 prompt_tokens 恒 0（排障误判没花 token）。
        # 与 `AGENT_ENABLE_THINKING` 同款教训：要么显式下发，要么依赖服务端/库默认，后者换模型时会静默失效。
        stream_usage=True,
        # 思考开关（见 `THINKING_ENABLED`）显式下发而非依赖服务端默认：服务端默认一变，延迟会静默翻数倍且看不出异常。
        # 备用端点若不认此参数，用 `AGENT_FALLBACK_SKIP_THINKING=true` 跳过（只影响备用那一节，不影响主模型）。
        extra_body={} if _fallback_skip_thinking() else {"enable_thinking": THINKING_ENABLED},
    )


def _fallback_skip_thinking() -> bool:
    """备用端点是否**不下发** `enable_thinking`（`specs/0014` §8 Q3）。调用时读 env：运行期开关写成导入期常量就永远改不动（本项目已踩多次）。"""
    raw = os.getenv("AGENT_FALLBACK_SKIP_THINKING")
    if raw is None or not raw.strip():
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _build(model: str, temperature: float, fallback: str = ""):
    """构造模型；给了 `fallback` 就用 LangChain 的 `.with_fallbacks()` 串成一条链。它原生保证"同一份 messages 重试备用"（主模型失败上下文原样传给备用，不丢历史、不换 prompt），自己写 try/except 容易在流式/结构化/bind_tools 上漏传参数。备用端点（`AGENT_FALLBACK_BASE_URL` / `AGENT_FALLBACK_API_KEY`）**留空 = 跟主端点走**（同端点换模型也合法，抗额度耗尽）。"""
    chat = _build_one(model, temperature)
    if not fallback:
        return chat
    fb_url = (os.getenv("AGENT_FALLBACK_BASE_URL") or "").strip()
    fb_key = (os.getenv("AGENT_FALLBACK_API_KEY") or "").strip()
    try:
        backup = _build_one(fallback, temperature, base_url=fb_url, api_key=fb_key)
        return chat.with_fallbacks([backup])
    except Exception as exc:  # noqa: BLE001 - 配错备用不该让服务起不来
        logger.warning("备用模型构造失败（退回单端点 model=%s）: %s", model, exc)
        return chat


@lru_cache(maxsize=4)
def get_main_model():
    """主模型：只在汇总节点调用一次。"""
    return _build(DEFAULT_MAIN_MODEL, TEMPERATURE, fallback=DEFAULT_FALLBACK_MODEL)


@lru_cache(maxsize=4)
def get_slot_model():
    """抽槽专用：默认与主模型同一个（见 `DEFAULT_SLOT_MODEL`）。抽槽每轮只调 1 次、不在热路径上，主模型的 5~6s 可接受；快模型在该通道上会卡死（不返回），必须隔离。"""
    return _build(DEFAULT_SLOT_MODEL, TEMPERATURE)


@lru_cache(maxsize=4)
def get_light_model():
    """轻量模型：入口意图识别、品类归类、细分筛选、行程编排、ReAct 决策、日常闲聊。"""
    return _build(DEFAULT_LIGHT_MODEL, TEMPERATURE, fallback=DEFAULT_FALLBACK_LIGHT_MODEL)


@lru_cache(maxsize=4)
def get_planner_model():
    """计划轨**编排**专用（`specs/0014` T1）：挑店 / 排序 / 起标题。planner 只做结构化编排，可换"更准但更慢"的模型而不拖慢整条轨；`AGENT_PLANNER_MODEL` 留空 ⇒ 返回轻量模型实例（行为与现状完全一致）。"""
    return _build(DEFAULT_PLANNER_MODEL, TEMPERATURE)


def structured(chat: Any, schema: Any):
    """统一的「结构化输出」入口：**显式**指定 `function_calling`。`with_structured_output` 在未知 qwen 模型上落到 json_mode，要求 messages 含 "json" 字样否则 400；本项目提示词不写格式说明 ⇒ 必然 400 并降级（换模型才暴露）。function calling 用 tools 传 schema 绕开；服务端不按 schema 强校验，严格性由客户端 Pydantic 保证。"""
    return chat.with_structured_output(schema, method="function_calling")
