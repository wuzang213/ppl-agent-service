# app/core/logger.py
import logging
import sys
import warnings

# 日志格式：时间 - 级别 - 模块 - [trace id] - 消息；`%(trace_id)s` 由 TraceIdFilter 注入，一次请求统一 X-Trace-Id。
# 必须用 `%` 风格占位：用 `{}` + 参数会让 logging 抛 TypeError 并静默吞掉整条日志（最难发现的失效）。
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - [trace=%(trace_id)s] - %(message)s"


class TraceIdFilter(logging.Filter):
    """给每条日志补上当前请求的 trace id（拿不到就 `-`，绝不因为缺字段丢日志）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            try:
                # **延迟导入**：`trace` 会 import 本模块（用它打日志），顶层 import 会成环。
                from app.common.trace import get_trace_id

                record.trace_id = get_trace_id() or "-"
            except Exception:  # noqa: BLE001 - 日志过滤器绝不能成为故障源
                record.trace_id = "-"
        return True


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),  # 输出到控制台
            # logging.FileHandler("app.log")   # 如果需要存到文件可以开启
        ]
    )
    # 过滤器挂在 **handler** 上（不是 logger 上）：子 logger 的记录也会经过 root handler。
    for handler in logging.getLogger().handlers:
        handler.addFilter(TraceIdFilter())
    _silence_pydantic_serializer_noise()


def _silence_pydantic_serializer_noise():
    """压掉 LangChain 结构化输出产生的无害序列化告警（每次调用刷一行，淹掉真故障日志）。
    正则须锚第一行前缀 `Pydantic serializer warnings`：`warnings.filterwarnings` 用 re.match（不跨行），写 `.*field_name=...` 永远匹配不上而静默失效。
    只压这一条，不用 `-W ignore`/simplefilter，以免连同 UserWarning（弃用 API、参数改名等我们需跟进的信号）一起吞掉。"""
    warnings.filterwarnings(
        "ignore",
        message=r"Pydantic serializer warnings",
        category=UserWarning,
    )


# 创建一个全局的 logger 实例
logger = logging.getLogger("personal_chief")
