"""Agent 服务注册到 Nacos，供网关 lb://agent-service 转发。"""

import os
import socket

from dotenv import load_dotenv
from v2.nacos import NacosNamingService, RegisterInstanceParam, ClientConfigBuilder, DeregisterInstanceParam

from app.common.config import require_env
from app.common.logger import logger

load_dotenv()

# 部署关键：连哪台 Nacos 必须显式配置，缺失即启动失败（不再静默回退到固定地址）。
NACOS_SERVER_ADDRS = require_env("NACOS_SERVER_ADDRS")


def _client_config():
    builder = ClientConfigBuilder().server_address(
        NACOS_SERVER_ADDRS
    ).namespace_id(os.getenv("NACOS_NAMESPACE", ""))
    username = os.getenv("NACOS_USERNAME", "")
    password = os.getenv("NACOS_PASSWORD", "")
    if username:
        builder.username(username).password(password)
    return builder.build()


def _instance_ip() -> str:
    """注册到 Nacos 的 IP：显式配置优先，否则自动探测本机 IP。
    空串须等价于「未设置」：os.getenv 键存在但值为空串时返回空串（不触发 fallback），照模板配置会把注册 IP 写成空串导致网关连不上，故用 `or` 兜底（None/未设置/空串/空白一律自动探测）。
    显式配了值就原样用，过期 IP 应留空而非写死。"""
    explicit = (os.getenv("AGENT_IP") or "").strip()
    return explicit or socket.gethostbyname(socket.gethostname())


def _instance_port() -> int:
    return int(os.getenv("AGENT_PORT", "8001"))


async def register_to_nacos() -> None:
    try:
        service = await NacosNamingService.create_naming_service(_client_config())
        await service.register_instance(
            RegisterInstanceParam(
                ip=_instance_ip(),
                port=_instance_port(),
                service_name=os.getenv("NACOS_SERVICE_NAME", "agent-service"),
                group_name=os.getenv("NACOS_GROUP_NAME", "DEFAULT_GROUP"),
                ephemeral=True,
                metadata={"preserved.heart.beat.interval": "5"},
            )
        )
        logger.info("Agent 已注册到 Nacos: %s:%s", _instance_ip(), _instance_port())
    except Exception as exc:
        logger.error("Nacos 注册失败: %s", exc)


async def deregister_from_nacos() -> None:
    try:
        service = await NacosNamingService.create_naming_service(_client_config())
        await service.deregister_instance(
            DeregisterInstanceParam(
                ip=_instance_ip(),
                port=_instance_port(),
                service_name=os.getenv("NACOS_SERVICE_NAME", "agent-service"),
                group_name=os.getenv("NACOS_GROUP_NAME", "DEFAULT_GROUP"),
                ephemeral=True,
            )
        )
        await service.shutdown()
        logger.info("Agent 已从 Nacos 注销")
    except Exception as exc:
        logger.error("Nacos 注销失败: %s", exc)