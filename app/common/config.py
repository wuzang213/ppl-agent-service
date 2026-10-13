"""配置读取：必填项一律从环境变量取，缺失/为空即启动失败，不在代码里静默给默认值。
背景：`llm.py` 曾把 `AGENT_MODEL` 回退成 ``"qwen-plus"``、`AGENT_LIGHT_MODEL` 回退成 ``"qwen-flash"``，RAG 的 `RAG_EMBEDDING_MODEL` / `RAG_RERANK_MODEL` 也各写死模型名 —— 漏配 `.env` 就会悄无声息地用错模型。
适用边界：只用于"标识符 / 连接 / 凭证"这类部署关键配置；纯调参（超时、预算、端口、阈值）仍保留 `os.getenv(X, 安全默认)`。"""

import os


def require_env(name: str) -> str:
    """读取必填环境变量；未设置或为空字符串则抛 `RuntimeError`，返回值已去首尾空白。
    用法：模块顶层、确保已 ``load_dotenv()`` 后调用，如 ``AGENT_MODEL = require_env("AGENT_MODEL")``。
    只用于"标识符 / 连接 / 凭证"这类写错会静默出错的配置；纯调参（超时 / 预算 / 端口 / 阈值）仍用 `os.getenv(X, 默认)`。"""
    val = os.getenv(name)
    if val is None or not val.strip():
        raise RuntimeError(
            f"必填环境变量 {name} 未配置或为空；请从 .env 提供，"
            f"不要在代码里依赖静默默认值（配置真相源是 .env）"
        )
    return val.strip()
