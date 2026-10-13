"""历史压缩的 token 预算估算与切分规则（默认只依赖标准库，便于 `test_token_budget.py` 脱离 langgraph/langchain 单独加载）；CJK ≈ 1 token/字、非 CJK ≈ 1/4 字符（可用 `AGENT_HISTORY_ESTIMATE_FACTOR` 校准）；`USE_TIKTOKEN_ESTIMATE` 可选启用 `tiktoken` 真实计数（encoder 默认 `cl100k_base`、可配 `TIKTOKEN_ENCODING`），一律函数内懒加载、不可用时安全回落 CJK，切分规则两套计数完全复用。"""

import os
from typing import Any, Sequence

# 默认 token 预算（估算 token）。它是**一条统一口径**：既决定压缩把切点定在哪里，
# 也决定只看最近上下文的节点（如 `daily_chat`）能注入多少历史。
DEFAULT_TOKEN_BUDGET = 3000
# 默认校准系数：1.0 = 不做缩放，保证开箱行为可预期。
DEFAULT_ESTIMATE_FACTOR = 1.0
#: tiktoken 默认 encoder（模型级可配：`TIKTOKEN_ENCODING`）。
DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"

# CJK / 全角字符的码点区间。这些字符在主流 tokenizer 里基本「一字一 token」，
# 而 ASCII 约「4 字符一 token」，两者量级差 4 倍，必须分开统计。
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),  # CJK 统一表意文字
    (0x3400, 0x4DBF),  # CJK 扩展 A
    (0x3000, 0x303F),  # CJK 标点
    (0xFF00, 0xFFEF),  # 全角字符
)


def _is_cjk(ch: str) -> bool:
    """判断单个字符是否按「一字一 token」估算（CJK 与全角字符）。"""
    code = ord(ch)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def _message_text(message: Any) -> str:
    """取出单条消息的文本内容，兼容 str / list[dict]（新版模型可能返回 list）。
    刻意**不复用** `app/agents/graph/nodes.as_text` —— 那会 import 整个 nodes 模块（连带
    langchain / langgraph），破坏本模块「只依赖标准库」的约束。"""
    content = getattr(message, "content", None)
    if content is None:
        content = message if isinstance(message, str) else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
            if item is not None
        )
    return str(content)


def _is_user_message(message: Any) -> bool:
    """判断消息是否为「用户」角色，用于避免切出孤儿 assistant 消息。
    走鸭子类型（`type` 属性 / 类名）而非 import `langchain_core.messages.HumanMessage`：
    本模块只依赖标准库，且测试里用的是假消息对象。"""
    role = getattr(message, "type", None)
    if isinstance(role, str) and role.lower() in ("human", "user"):
        return True
    return type(message).__name__.lower().startswith("human")


def token_budget() -> int:
    """读取压缩预算 `AGENT_HISTORY_TOKEN_BUDGET`（单位：估算 token，默认 6000）。
    非法值（非整数 / 负数 / 0）一律**回落默认值且不抛异常**（配置写错不该让服务起不来）；
    **运行时读取**而非导入时读，便于评测在同一进程内切换阈值。"""
    try:
        value = int(os.getenv("AGENT_HISTORY_TOKEN_BUDGET", str(DEFAULT_TOKEN_BUDGET)))
    except (TypeError, ValueError):
        return DEFAULT_TOKEN_BUDGET
    return value if value > 0 else DEFAULT_TOKEN_BUDGET


def estimate_factor() -> float:
    """读取估算校准系数 `AGENT_HISTORY_ESTIMATE_FACTOR`（默认 1.0）。
    用于**不改代码**地把「估算 token」校准到「真实 token」（真实值见 `hmdp_metrics.record_usage`）；
    非法值（非数字 / ≤ 0）回落 1.0，不抛异常。"""
    try:
        value = float(os.getenv("AGENT_HISTORY_ESTIMATE_FACTOR", str(DEFAULT_ESTIMATE_FACTOR)))
    except (TypeError, ValueError):
        return DEFAULT_ESTIMATE_FACTOR
    return value if value > 0 else DEFAULT_ESTIMATE_FACTOR


def use_tiktoken() -> bool:
    """是否启用 tiktoken 真实计数（`USE_TIKTOKEN_ESTIMATE`，默认**关**）。
    真值判定：`1/true/yes/on`（大小写不敏感）；其余（含缺失/空串）为关。**运行时读取**。"""
    return os.getenv("USE_TIKTOKEN_ESTIMATE", "").strip().lower() in ("1", "true", "yes", "on")


def tiktoken_encoding_name() -> str:
    """tiktoken encoder 名（`TIKTOKEN_ENCODING`，默认 `cl100k_base`）—— 预留**模型级**配置。"""
    name = os.getenv("TIKTOKEN_ENCODING", "").strip()
    return name or DEFAULT_TIKTOKEN_ENCODING


#: encoder 缓存：键 = encoder 名，值 = tiktoken Encoding 或 `None`（不可用）。
_ENCODER_CACHE: dict[str, Any] = {}


def _get_encoder(name: str) -> Any | None:
    """惰性加载并缓存 tiktoken encoder；任何失败一律返回 `None`（安全回落），绝不抛异常。`import tiktoken` 必须放在函数内，未启用时不得 import（由 `test_token_budget.py` 懒加载断言看住）。"""
    if name in _ENCODER_CACHE:
        return _ENCODER_CACHE[name]
    try:
        import tiktoken  # noqa: PLC0415  （懒加载：仅启用分支才 import）
        encoder: Any = tiktoken.get_encoding(name)
    except Exception:
        encoder = None
    _ENCODER_CACHE[name] = encoder
    return encoder


def estimate_tokens(text: str) -> int:
    """估算文本占用的 token 数。默认（CJK 估算）：CJK 按 1 token/字、非 CJK 按 1/4 字符累加再乘 `estimate_factor()` 向下取整，空文本返回 0、非空至少 1；`USE_TIKTOKEN_ESTIMATE` 启用且可用时返回真实 token 数（不再乘因子），否则安全回落 CJK；`None`/非字符串按空串处理。"""
    if not text:
        return 0
    s = str(text)
    if use_tiktoken():
        encoder = _get_encoder(tiktoken_encoding_name())
        if encoder is not None:
            try:
                return max(1, len(encoder.encode(s)))
            except Exception:
                pass  # 编码失败 ⇒ 回落到 CJK 估算，不抛
    cjk = 0
    other = 0
    for ch in s:
        if _is_cjk(ch):
            cjk += 1
        else:
            other += 1
    return max(1, int((cjk + other / 4.0) * estimate_factor()))


def estimate_messages_tokens(messages: Sequence[Any] | None) -> int:
    """逐条累加估算整段历史占用的 token 数（空输入返回 0）；逐条而非拼成一整段，
    这样每条非空消息至少计 1 token，也与模型按消息边界计费更接近。"""
    if not messages:
        return 0
    return sum(estimate_tokens(_message_text(message)) for message in messages)


def split_index(messages: Sequence[Any] | None, budget: int) -> int:
    """按 token 预算算出压缩候选区条数 `cut`（规格 §2 目标 3 的唯一权威实现）。
    从最后一条往最老累加，累计 > budget 时记下 `cut`；再对齐到保留区以 user 消息开头（避免孤儿 assistant），
    保底 `cut <= len(messages) - 1`。返回 `cut`，候选区 = `messages[:cut]`，`cut == 0` 表示本轮不压缩。"""
    items = list(messages or [])
    total = len(items)
    if total == 0:
        return 0

    cut = 0
    acc = 0
    for idx in range(total - 1, -1, -1):
        acc += estimate_tokens(_message_text(items[idx]))
        if acc > budget:
            cut = idx + 1
            break

    # ③ 对齐 user 边界（只在 total 之内后移，越界交给 ④ 收回）
    while cut < total and not _is_user_message(items[cut]):
        cut += 1
    # ④ 保底：至少保留最后一条消息
    if cut > total - 1:
        cut = total - 1
    return cut


def should_compress(messages: Sequence[Any] | None, budget: int) -> bool:
    """压缩触发器：候选区非空即触发（只看 token，不看消息条数）。
    候选区由 `split_index(messages, budget)` 按 token 预算划出，非空即说明历史确实超出预算。"""
    return split_index(messages, budget) > 0


def recent_messages(messages: Sequence[Any] | None, budget: int) -> list[Any]:
    """按 token 预算取「最近的消息」：从最新往最老累加，返回累计不超预算的最大连续后缀。
    用于给只看最近上下文的节点（如 `daily_chat`）划注入窗口；`split_index` 算的是压缩切点（对齐 user 边界、
    可返回 0），本函数只截窗口且不做边界对齐，但最后一条自身超预算时仍单独返回它。"""
    items = list(messages or [])
    acc = 0
    for idx in range(len(items) - 1, -1, -1):
        acc += estimate_tokens(_message_text(items[idx]))
        if acc > budget:
            kept = items[idx + 1:]
            return kept if kept else items[-1:]
    return items