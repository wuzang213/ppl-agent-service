"""Agent 服务注册到 Nacos，供网关 lb://agent-service 转发。"""

import os
import socket

from v2.nacos import NacosNamingService, RegisterInstanceParam, ClientConfigBuilder, DeregisterInstanceParam

from app.common.logger import logger


def _client_config():
    builder = ClientConfigBuilder().server_address(
        os.getenv("NACOS_SERVER_ADDRS")
    ).namespace_id(os.getenv("NACOS_NAMESPACE", ""))
    username = os.getenv("NACOS_USERNAME", "")
    password = os.getenv("NACOS_PASSWORD", "")
    if username:
        builder.username(username).password(password)
    return builder.build()


def _instance_ip() -> str:
    return os.getenv("AGENT_IP", socket.gethostbyname(socket.gethostname()))


def _instance_port() -> int:
    return int(os.getenv("AGENT_PORT", "8001"))


async def register_to_nacos() -> None:
    if not os.getenv("NACOS_SERVER_ADDRS"):
        logger.warning("NACOS_SERVER_ADDRS 未配置，跳过 Nacos 注册")
        return
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
    if not os.getenv("NACOS_SERVER_ADDRS"):
        return
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