"""计划轨的**槽位配置**（`specs/0012` §12.3、§12.3.3）。`REQUIRED_SLOTS` 决定问不问、`SLOT_SPECS[*]["options"]` 只做引导候选、`SLOT_SPECS[*]["validate"]` 是唯一收值依据 —— 三者分开配置。用户输入不必匹配 options（如 `people=3` 仍须通过校验）；本模块纯函数、无 IO、无模型调用，`clarify_slots` 在 `interrupt()` 之前只允许调这里的函数（否则重放会执行两次）。"""

from __future__ import annotations

import os
import re
from typing import Any, Sequence


def _valid_text(value: Any) -> bool:
    """非空字符串（时间 / 区域这类自由文本槽）。"""
    return isinstance(value, str) and bool(value.strip())


#: 时间槽的**最小可识别痕迹**：必须含一个时间词或 `HH:MM` / `N点`。
#: 不能只要求非空：实测模型会把「1人」之类填进 `time`（同理 `area`/`budget`），导致标题写成「1人」；时间槽虽是开放文本，但必须真的带时间信息。
_TIME_TRACES = re.compile(
    r"(今天|今日|今儿|今晚|明天|明日|明儿|明晚|后天|大后天|周末|周[一二三四五六日天]|"
    r"下个?周|这周|上午|早上|早晨|中午|下午|傍晚|晚上|夜里|凌晨|"
    r"\d{1,2}\s*[:：]\s*\d{2}|\d{1,2}\s*点)"
)


def _valid_time(value: Any) -> bool:
    """`time` 的值域：非空文本，且**必须含时间痕迹**（见 `_TIME_TRACES`）。"""
    if not _valid_text(value):
        return False
    return _TIME_TRACES.search(str(value)) is not None


def _valid_count(value: Any) -> bool:
    """正整数且 ≤20。必须排除 `bool`：`isinstance(True, int)` 为真，否则 `True` 会被当成 1 通过。"""
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 20


def _valid_budget(value: Any) -> bool:
    """正整数（元）。同样排除 `bool`。"""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_scenes(value: Any) -> bool:
    """至少包含一个非空元素的字符串列表。逐项须"像个活动"：**不含数字**且不长于 8 字，否则模型会把「2个人」整句填进 `scenes`，导致标题错乱成「2个人行程」。"""
    if not isinstance(value, list):
        return False
    items = [str(x).strip() for x in value]
    items = [x for x in items if x]
    if not items:
        return False
    return all(len(x) <= 8 and not any(ch.isdigit() for ch in x) for x in items)


#: 「主观偏好/形容词」高频词 —— 它们**不是区域名**，命中即视为 `area` **未填**（确定性闸门）。
#: 需要它兜底：模型会把「安静些」当区域名填进 `area`，导致不再追问区域、标题变成「安静些行程」（真机约 8% 概率）。判据用**包含**而非相等（脏值带后缀如「安静些的地方」），误伤只是多问一句；本表是兜底，提示词 `SLOT_SYSTEM` 负责提高命中率。
_AREA_PREFERENCE_WORDS: tuple[str, ...] = (
    "安静", "清静", "幽静", "不吵", "人少", "热闹",
    "干净", "卫生", "环境", "氛围", "服务", "档次", "高档",
    "便宜", "实惠", "划算", "性价比",
    "好吃", "好喝", "正宗", "新鲜",
)


def _valid_area(value: Any) -> bool:
    """`area` 的值域：非空文本、**不含数字**、且**不能命中主观偏好词**（命中说明模型抽错）。区域是地名，不可能带数字，故加"不含数字"拦截（真机 id 483：`「1人」`会一路通过并写成「1人行程」）。"""
    if not _valid_text(value):
        return False
    text = str(value)
    if any(ch.isdigit() for ch in text):
        return False
    return not any(word in text for word in _AREA_PREFERENCE_WORDS)


#: 必填槽位（顺序即**提问顺序**，稳定输出便于前端与快照测试）。
REQUIRED_SLOTS: tuple[str, ...] = ("time", "area", "people", "budget", "scenes")

#: 槽位规格：`label` 展示名 / `value_type` 给前端选控件 / `unit` 单位 /
#: `options` **引导用**候选（模型不生成、不修改） / `validate` **唯一**的收值依据。
SLOT_SPECS: dict[str, dict[str, Any]] = {
    "time": {
        "label": "出行时间",
        "value_type": "text",
        "unit": "",
        "options": [
            {"value": "今天下午", "label": "今天下午"},
            {"value": "今天傍晚", "label": "今天傍晚"},
            {"value": "明天上午", "label": "明天上午"},
            {"value": "明天下午", "label": "明天下午"},
        ],
        "validate": _valid_time,
    },
    "area": {
        "label": "活动区域",
        "value_type": "text",
        "unit": "",
        # 区域是开放文本（用户所在城市不同），刻意**不给固定选项**；
        # `_valid_area` 在"非空文本"之外额外排掉**主观偏好词**（见其 docstring）
        "options": [],
        "validate": _valid_area,
    },
    "people": {
        "label": "出行人数",
        "value_type": "number",
        "unit": "人",
        # 值域限制**固定写在配置里**，每次问都带上 —— 这样"超范围"时不必额外记住"上次给了什么"
        # （节点会重放，任何局部变量都会丢；多存一个"上次被拒的槽"就是多一份会漂移的状态）。
        "hint": "需为 1–20 之间的整数",
        "options": [
            {"value": 1, "label": "1人"},
            {"value": 2, "label": "2人"},
            {"value": 4, "label": "4人"},
            {"value": "custom", "label": "其他人数"},
        ],
        "validate": _valid_count,
    },
    "budget": {
        "label": "总预算",
        "value_type": "number",
        "unit": "元",
        "hint": "需为正整数（元）",
        "options": [
            {"value": 100, "label": "100元以内"},
            {"value": 200, "label": "200元左右"},
            {"value": 500, "label": "500元左右"},
        ],
        "validate": _valid_budget,
    },
    "scenes": {
        "label": "想做什么",
        "value_type": "list",
        "unit": "",
        "options": [
            {"value": "吃", "label": "吃"},
            {"value": "喝", "label": "喝"},
            {"value": "玩", "label": "玩"},
            {"value": "看", "label": "看"},
            {"value": "逛", "label": "逛"},
        ],
        "validate": _valid_scenes,
    },
}

#: 模板话术用的口语化问法（**确定性**，不调模型 —— 见 `specs/0012` §12.3.2 的偏差说明）。
_SLOT_ASKS: dict[str, str] = {
    "time": "大概什么时间去",
    "area": "想去哪一带",
    "people": "几个人一起去",
    "budget": "总预算大概多少",
    "scenes": "想安排点什么（比如吃、喝、逛、看）",
}


def max_clarify_retry() -> int:
    """澄清重试上限（默认 2）。调用时读环境，避免导入期常量陷阱（同 `state._task_ttl_turns`）。"""
    try:
        return int(os.getenv("AGENT_CLARIFY_MAX_RETRY", "2"))
    except (TypeError, ValueError):
        return 2


def is_valid(slot: str, value: Any) -> bool:
    """按**值域规则**判断槽值是否可用。未知槽位 / 规则抛异常一律视为不合法（宁可多问一句）。"""
    spec = SLOT_SPECS.get(slot)
    if not spec:
        return False
    try:
        return bool(spec["validate"](value))
    except Exception:
        return False


def missing_slots(draft: dict | None) -> list[str]:
    """`draft.slots` 里**不合法或缺失**的必填槽位，按 `REQUIRED_SLOTS` 顺序返回（纯函数）。齐全返回 `[]` ⇒ 调用方放行、不挂起。"""
    slots = (draft or {}).get("slots") or {}
    if not isinstance(slots, dict):
        return list(REQUIRED_SLOTS)
    return [name for name in REQUIRED_SLOTS if not is_valid(name, slots.get(name))]


def wording(missing: Sequence[str], *, retry: int = 0) -> str:
    """澄清话术（**模板**，同输入同输出）。`retry > 0` 走引导式话术：二次澄清不说"请从选项里选"，而是引导用户给具体值（选项仍由 `questions[].options` 带前端展示）；含值域限制的槽位（人数/预算）每次都把限制说清楚。"""
    asks = [_SLOT_ASKS[n] for n in missing if n in _SLOT_ASKS]
    if not asks:
        return ""
    head = "还差一点信息就能帮你排好行程：" if retry > 0 else "为了帮你把行程排好，想先确认几件事："
    text = head + "；".join(asks) + "？"
    hints = [f"{SLOT_SPECS[n]['label']}{SLOT_SPECS[n]['hint']}" for n in missing
             if n in SLOT_SPECS and SLOT_SPECS[n].get("hint")]
    if hints:
        text += "（" + "；".join(hints) + "）"
    return text


def build_clarification(missing: Sequence[str], task_id: str = "", *, retry: int = 0) -> dict:
    """组 `clarification` 载荷（纯函数；选项**全部来自配置**，模型不参与）。形状见 `specs/0012` §12.3；`text` 给纯文本展示/落库用。"""
    questions: list[dict[str, Any]] = []
    for name in missing:
        spec = SLOT_SPECS.get(name)
        if not spec:
            continue
        questions.append(
            {
                "slot": name,
                "label": spec["label"],
                "required": True,
                "value_type": spec["value_type"],
                "unit": spec.get("unit", ""),
                "hint": spec.get("hint", ""),
                # 深拷贝，避免调用方改坏配置里的选项
                "options": [dict(o) for o in (spec.get("options") or [])],
            }
        )
    return {
        "type": "clarification",
        "task_id": task_id,
        "mode": "slot_fill",
        "questions": questions,
        "text": wording(missing, retry=retry),
    }


#: 与 `scenes` **下标对齐**的归类字段名（`specs/0031`）。它不是必填槽位，只是随 `scenes` 走的附件。
SCENE_TYPES_KEY = "scene_types"


def _clean_scene_types(value: Any) -> list[str]:
    """把归类结果归一成字符串列表（非 list ⇒ `[]`；逐项 `strip()`，非 str ⇒ 空串**保位置**）。"""
    if not isinstance(value, list):
        return []
    return [x.strip() if isinstance(x, str) else "" for x in value]


def aligned_scene_types(raw: dict | None) -> list[str]:
    """取**与 `scenes` 下标对齐**的归类结果（纯函数，`specs/0031` T3）。对齐是硬约束：`scenes` 缺失/为空或长度不等 ⇒ 返回 `[]`（视为无归类）；长度不等绝不强行配对，错位归类（如"唱歌"配到"酒吧"）是静默且难查的坏。"""
    scenes = (raw or {}).get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return []
    types = _clean_scene_types((raw or {}).get(SCENE_TYPES_KEY))
    if len(types) != len(scenes):
        return []
    return types


def sanitize_slots(raw: dict | None) -> dict:
    """把模型抽出的键值对**按预定义槽位过滤**并丢弃非法值（§12.3.4 硬约束 4）。非法值直接丢弃而非保留等下游校验（同 `RoutePlan._clean_sub_intents`：脏字段不该让链路失败，也不能溜进 State）。`scene_types`（`specs/0031`）非必填但跟随 `scenes` 进出：仅当 `scenes` 通过校验且两者等长才带上，否则整体丢弃退回字典/原词。"""
    out: dict[str, Any] = {}
    for name, value in (raw or {}).items():
        if name in SLOT_SPECS and is_valid(name, value):
            out[name] = value
    if "scenes" in out:
        aligned = aligned_scene_types(raw)
        if aligned:
            out[SCENE_TYPES_KEY] = aligned
    return out


__all__ = [
    "REQUIRED_SLOTS",
    "SCENE_TYPES_KEY",
    "SLOT_SPECS",
    "aligned_scene_types",
    "build_clarification",
    "is_valid",
    "max_clarify_retry",
    "missing_slots",
    "sanitize_slots",
    "wording",
]
