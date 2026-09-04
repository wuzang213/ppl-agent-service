from typing import Optional, List, Dict, Any

from pydantic import BaseModel

class RagAnswer(BaseModel):
    """Agent 结构化回答：answer 为最终事实结论，source 为来源博客标题。"""
    answer: str
    source: list[str] = []

# --- 1. 数据模型 ---
class ChatRequest(BaseModel):
    message: Optional[str] = None
    image_url: Optional[str] = None
    thread_id: str
    # 用户确认的interrupt操作
    interrupt_decision: Optional[Dict[str, Any]] = None