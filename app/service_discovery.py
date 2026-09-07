"""Nacos 服务发现：为 agent 出站调用 Java 微服务提供动态实例解析。

设计要点：
- 默认优先使用 .env 中显式配置的静态地址（SHOP/BLOG/VOUCHER_SERVICE_URL，如 localhost:8082），
  该地址由使用者验证可达，是最稳妥的选择。
- 仅当 .env 未配置（为空）时，才回落到 Nacos 命名服务拉取的 *健康* 实例地址
  （自带 LB + 故障转移 + 地址动态）。
- 本地缓存 + 定时刷新；Nacos 不可用或解析不到时，不影响已配置的静态地址。
- resolve() 是同步、无网络 IO、永不抛异常的纯查表操作，保证出站请求绝不被阻塞。

注意：Spring Cloud 默认用机器局域网 IP（如 172.x.x.x / 192.168.x.x）注册到 Nacos，该地址在
agent 运行时未必可达（防火墙 / 仅监听回环 / 跨容器网络）。因此 .env 静态地址（localhost）优先，
避免"Nacos 有实例却连不通"反而劣于写死地址的情况。

"""

import asyncio
import os
import threading
from typing import Optional

from v2.nacos import ClientConfigBuilder, ListInstanceParam, NacosNamingService

from app.common.logger import logger

# 逻辑服务 key -> Nacos 中 Java 服务的注册名（可经 env 覆盖，默认与 Spring 注册名一致）
SERVICE_NAMES = {
    "shop": os.getenv("NACOS_SHOP_SERVICE", "shop-service"),
    "blog": os.getenv("NACOS_BLOG_SERVICE", "blog-service"),
    "voucher": os.getenv("NACOS_VOUCHER_SERVICE", "voucher-service"),
}

# 兜底静态地址（Nacos 解析不到时使用，直接读 env，不依赖本模块外的常量）
FALLBACK_URLS = {
    "shop": os.getenv("SHOP_SERVICE_URL", "").rstrip("/"),
    "blog": os.getenv("BLOG_SERVICE_URL", "").rstrip("/"),
    "voucher": os.getenv("VOUCHER_SERVICE_URL", "").rstrip("/"),
}

REFRESH_SECONDS = int(os.getenv("NACOS_DISCOVERY_REFRESH", "10"))

_cache: dict[str, str] = {}
_cache_lock = threading.Lock()
_stop = threading.Event()
_thread: Optional[threading.Thread] = None


def _client_config():
    builder = ClientConfigBuilder().server_address(
        os.getenv("NACOS_SERVER_ADDRS", "")
    ).namespace_id(os.getenv("NACOS_NAMESPACE", ""))
    username = os.getenv("NACOS_USERNAME", "")
    password = os.getenv("NACOS_PASSWORD", "")
    if username:
        builder.username(username).password(password)
    return builder.build()


def _build_url(inst) -> str:
    # 实例 metadata 里若带了 scheme（如 https）则用之，否则默认 http
    scheme = (getattr(inst, "metadata", None) or {}).get("scheme") or "http"
    return f"{scheme}://{inst.ip}:{inst.port}"


async def _refresh(svc) -> None:
    group = os.getenv("NACOS_GROUP_NAME", "DEFAULT_GROUP")
    for key, name in SERVICE_NAMES.items():
        try:
            instances = await svc.list_instances(
                ListInstanceParam(service_name=name, group_name=group, healthy_only=True)
            )
        except Exception as exc:
            logger.warning("Nacos 拉取 %s 实例失败: %s", name, exc)
            continue
        if instances:
            # 简单策略：取第一个健康实例。如需加权随机/轮询可在此扩展。
            url = _build_url(instances[0])
            with _cache_lock:
                _cache[key] = url
            logger.debug("Nacos 发现 %s -> %s（健康实例 %s 个）", name, url, len(instances))


async def _run() -> None:
    svc = None
    while not _stop.is_set():
        if svc is None:
            try:
                svc = await NacosNamingService.create_naming_service(_client_config())
            except Exception as exc:
                logger.warning(
                    "Nacos 服务发现连接失败，%ss 后重试，期间使用静态地址: %s",
                    REFRESH_SECONDS, exc,
                )
                await asyncio.sleep(REFRESH_SECONDS)
                continue
        try:
            await _refresh(svc)
        except Exception as exc:
            logger.warning("Nacos 服务发现刷新异常，下次重建连接: %s", exc)
            try:
                await svc.shutdown()
            except Exception:
                pass
            svc = None
        await asyncio.sleep(REFRESH_SECONDS)
    if svc is not None:
        try:
            await svc.shutdown()
        except Exception:
            pass


def start_discovery() -> None:
    """应用启动时调用（main.py lifespan）。NACOS_SERVER_ADDRS 未配置则静默不启用。"""
    global _thread
    if not os.getenv("NACOS_SERVER_ADDRS"):
        logger.info("NACOS_SERVER_ADDRS 未配置，出站调用使用 .env 静态地址")
        return
    _stop.clear()
    _thread = threading.Thread(
        target=lambda: asyncio.run(_run()), name="nacos-discovery", daemon=True
    )
    _thread.start()
    logger.info("Nacos 服务发现已启动（刷新间隔 %ss）", REFRESH_SECONDS)


def stop_discovery() -> None:
    _stop.set()


def resolve(key: str) -> str:
    """返回逻辑服务的 base_url：优先 .env 静态地址，否则回落 Nacos 发现的实例。

    同步、无 IO、永不抛异常——保证出站请求不被阻塞。

    优先级说明：
    - .env 中的 SHOP/BLOG/VOUCHER_SERVICE_URL 优先（使用者已验证可达，如 localhost:8082）；
    - 仅当 .env 对应地址为空时，才使用 Nacos 服务发现解析到的健康实例地址。
    """
    fallback = FALLBACK_URLS.get(key, "")
    if fallback:
        return fallback
    with _cache_lock:
        url = _cache.get(key)
    if url:
        return url
    return ""
