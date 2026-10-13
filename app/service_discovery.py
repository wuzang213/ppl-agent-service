"""Nacos 服务发现：为 agent 出站调用 Java 微服务提供动态实例解析。
优先用 .env 显式配置的静态地址（SHOP/BLOG/VOUCHER_SERVICE_URL，如 localhost:8082，使用者已验证可达），仅当为空时才回落 Nacos 拉取的健康实例（自带 LB + 故障转移），本地缓存 + 定时刷新；`resolve()` 同步、无 IO、永不抛异常。
Spring Cloud 默认把机器局域网 IP（如 10.x.x.x）注册到 Nacos，agent 运行时未必可达（防火墙 / 仅监听回环 / 跨容器网络），故静态地址优先。"""

import asyncio
import os
import threading
from typing import Optional

from dotenv import load_dotenv
from v2.nacos import ClientConfigBuilder, ListInstanceParam, NacosNamingService

from app.common.config import require_env
from app.common.logger import logger

load_dotenv()

# 部署关键：连哪台 Nacos 必须显式配置，缺失即启动失败（不再静默回退到固定地址）。
NACOS_SERVER_ADDRS = require_env("NACOS_SERVER_ADDRS")

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
        NACOS_SERVER_ADDRS
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
    """应用启动时调用（main.py lifespan）。NACOS_SERVER_ADDRS 缺失会在导入期即 fail-fast，
    这里不再做“空则跳过”——连接失败由 _run 优雅降级为使用 .env 静态地址。"""
    global _thread
    _stop.clear()
    _thread = threading.Thread(
        target=lambda: asyncio.run(_run()), name="nacos-discovery", daemon=True
    )
    _thread.start()
    logger.info("Nacos 服务发现已启动（刷新间隔 %ss）", REFRESH_SECONDS)


def stop_discovery() -> None:
    _stop.set()


def resolve(key: str) -> str:
    """返回逻辑服务的 base_url：优先 .env 静态地址（`SHOP/BLOG/VOUCHER_SERVICE_URL`，如 localhost:8082），
    仅当地址为空时才用 Nacos 服务发现解析到的健康实例。同步、无 IO、永不抛异常 —— 保证出站请求不被阻塞。"""
    fallback = FALLBACK_URLS.get(key, "")
    if fallback:
        return fallback
    with _cache_lock:
        url = _cache.get(key)
    if url:
        return url
    return ""
