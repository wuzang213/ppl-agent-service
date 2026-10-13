import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agents.hmdp_agent import hmdp_agent
from app.agents.ledger import LEDGER_TABLES, TABLE_RUNS
from app.rag.hmdp_mq_sync import start_rag_sync_consumer
from app.api.v1 import hmdp, ledger, oss, sessions
from app.common.config import require_env
from app.common.logger import logger, setup_logging
from app.common.mysql import close_mysql_pool, init_mysql_pool, verify_tables
from app.common.trace import TraceMiddleware
from app.models.message import TABLE_NAME as MESSAGE_TABLE
from app.models.session import TABLE_NAME as SESSION_TABLE
from app.nacos_registry import deregister_from_nacos, register_to_nacos
from app.service_discovery import start_discovery, stop_discovery

load_dotenv()

setup_logging()

# 启动时必须存在的表：代码不建表，DDL 由部署方执行 sql/ 下的脚本，这里只做存在性校验 fail-fast。
# 运行账本三张表（specs/0013）一并登记，保证"跑起来就有地方写"，避免静默不落库。
# 只想停止写入而不删表时，设 AGENT_LEDGER_ENABLED=false（表仍在，不影响启动）。
REQUIRED_TABLES = (SESSION_TABLE, MESSAGE_TABLE, *LEDGER_TABLES)

# 启动时必须存在的增量列（sql/ 只增不改）：脚本未执行即启动失败，而非首个请求才抛 Unknown column。
# agent_message.payload 为回看侧结构化载荷（specs/0012）。
# agent_runs 六列（query/意图路由/卡片摘要，specs/0017 T1）同出 sql/agent_ledger_add_context.sql。
REQUIRED_COLUMNS = {
    MESSAGE_TABLE: ("payload",),
    TABLE_RUNS: ("query_text", "scene", "sub_intents", "depends_on", "card_digest", "answer_digest"),
}


#: 就绪状态（GET /ready 判据）：用启动期已建立的状态，而非每次请求去 ping 依赖（specs/0015 §10.1）。
#: 健康检查本身不该成为负载，也不该因一次抖动判死实例；
#: 启动 15–30s 窗口内依赖未建好时 /ready 返回 503，上游据此摘流。
_READY = {"ready": False, "reason": "启动中"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 建池（含库不存在时的可读报错）→ 校验必需表/列存在。两步都 fail fast。
    try:
        await init_mysql_pool()
        await verify_tables(REQUIRED_TABLES, REQUIRED_COLUMNS)
        await hmdp_agent.init()
        await register_to_nacos()
        start_discovery()
        rag_consumer_task = asyncio.create_task(start_rag_sync_consumer())
        _READY["ready"] = True
        _READY["reason"] = ""
    except Exception as exc:  # noqa: BLE001
        # 未就绪原因进日志，**不外泄细节**（错误里可能含连接串/库表信息）
        _READY["ready"] = False
        _READY["reason"] = type(exc).__name__
        logger.error("启动失败，实例保持未就绪: %s", exc)
        raise
    try:
        yield
    finally:
        _READY["ready"] = False
        _READY["reason"] = "关闭中"
        rag_consumer_task.cancel()
        try:
            await rag_consumer_task
        except asyncio.CancelledError:
            pass
        stop_discovery()
        await deregister_from_nacos()
        await hmdp_agent.close()
        await close_mysql_pool()


app = FastAPI(
    title="Heima Agent API",
    description="评评哩智能体API",
    version="0.1.0",
    lifespan=lifespan,
)

# 部署关键：跨域白名单必须显式配置（安全相关），缺失即启动失败，不再静默回退到 localhost:3000。
cors_origins = require_env("CORS_ORIGINS").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hmdp.router, prefix="/api/v1", tags=["对话"])
app.include_router(oss.router, prefix="/api/v1", tags=["申请上传签名url"])
app.include_router(sessions.router, prefix="/api/v1", tags=["会话管理"])
app.include_router(ledger.router, prefix="/api/v1", tags=["运行账本"])

# 链路追踪中间件（`specs/0015` T2）：一次请求一个 `X-Trace-Id`，贯穿日志与账本。
#    用**纯 ASGI** 形态而不是 `BaseHTTPMiddleware`（后者会缓冲 SSE 流式响应）。
app.add_middleware(TraceMiddleware)


# 健康端点**必须注册在兜底路由 `/{path:path}` 与 `static` 挂载之前**（`specs/0015` R1）：
#    那两个会吞掉后注册的同名路径 —— 顺序错了健康检查会一直返回前端 index（且看起来"没报错"）。
@app.get("/health", tags=["运维"], include_in_schema=True)
async def health() -> dict:
    """存活（liveness）：进程活着就 200，**不看依赖**（否则依赖一抖就被判死、反复重启）。"""
    return {"status": "ok"}


@app.get("/ready", tags=["运维"], include_in_schema=True)
async def ready(response: Response) -> Response:
    """就绪（readiness）：依赖（MySQL 池 / 表校验 / checkpoint / Nacos）建立完成才返回 200；未就绪返回 503 供上游摘流，原因只进日志不回传。"""
    if _READY["ready"]:
        return JSONResponse({"status": "ready"})
    logger.warning("/ready 未就绪：%s", _READY["reason"])
    return JSONResponse({"status": "not_ready"}, status_code=503)


static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


@app.get("/{path:path}", include_in_schema=False)
async def serve_frontend(path: str):
    if path.startswith("api/"):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Not Found"}, status_code=404)
    file_path = os.path.join(static_dir, path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "你的专属助手上线了", "status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=os.getenv("AGENT_HOST", "0.0.0.0"), port=int(os.getenv("AGENT_PORT", "8001")), reload=True)