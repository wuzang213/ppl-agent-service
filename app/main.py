import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agents.hmdp_agent import hmdp_agent
from app.rag.hmdp_mq_sync import start_rag_sync_consumer
from app.api.v1 import hmdp, oss, sessions
from app.common.logger import setup_logging
from app.common.mysql import close_mysql_pool, init_mysql_pool, verify_tables
from app.models.message import TABLE_NAME as MESSAGE_TABLE
from app.models.session import TABLE_NAME as SESSION_TABLE
from app.nacos_registry import deregister_from_nacos, register_to_nacos
from app.service_discovery import start_discovery, stop_discovery

setup_logging()

# 启动时必须存在的表。**代码不建表** —— DDL 由部署方执行 sql/ 下的脚本（见 sql/README.md），
# 这里只做存在性校验：缺表就 fail fast，避免运行时第一个请求才报错。
REQUIRED_TABLES = (SESSION_TABLE, MESSAGE_TABLE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 建池（含库不存在时的可读报错）→ 校验必需表存在。两步都 fail fast。
    await init_mysql_pool()
    await verify_tables(REQUIRED_TABLES)
    await hmdp_agent.init()
    await register_to_nacos()
    start_discovery()
    rag_consumer_task = asyncio.create_task(start_rag_sync_consumer())
    yield
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

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

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