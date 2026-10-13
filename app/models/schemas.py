from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

class RagAnswer(BaseModel):
    """Agent 结构化回答：answer 为最终事实结论，source 为来源博客标题。
    模型在 JSON 模式下对“无值”字段常返回 null，故在 model_validator 里兜底，避免单字段 null 致整条解析失败。
    别用 `with_structured_output(RagAnswer).astream()` 做逐 token 流式：该调用**不做增量解析**（62 chunk 里 57 个空、非空的是裸 JSON 语法）；逐 token 须走纯文本调用，见 `nodes._summarize_plain_stream`。"""

    model_config = ConfigDict(extra="ignore")

    answer: str = ""
    source: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def _coerce_nulls(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if out.get("answer") is None:
            out["answer"] = ""
        if out.get("source") is None:
            out["source"] = []
        return out