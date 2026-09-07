"""模型工厂：主模型负责汇总，轻量模型负责意图识别与日常闲聊。"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

DEFAULT_MAIN_MODEL = os.getenv("AGENT_MODEL", "qwen-plus")
DEFAULT_LIGHT_MODEL = os.getenv("AGENT_LIGHT_MODEL", "qwen-flash")
TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.1"))


def _build(model: str, temperature: float):
    base_url = os.getenv("DASHSCOPE_BASE_URL")
    api_key = os.getenv("DASHSCOPE_API_KEY")
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
    )


@lru_cache(maxsize=4)
def get_main_model():
    """主模型：只在汇总节点调用一次。"""
    return _build(DEFAULT_MAIN_MODEL, TEMPERATURE)


@lru_cache(maxsize=4)
def get_light_model():
    """轻量模型：入口意图识别、日常闲聊，追求便宜和快。"""
    return _build(DEFAULT_LIGHT_MODEL, TEMPERATURE)
