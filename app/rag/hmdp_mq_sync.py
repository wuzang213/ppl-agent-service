"""监听 Canal 的缓存同步 fanout 交换机，增量更新 RAG 索引。"""

import asyncio
import json
import os

import aio_pika

from app.common.logger import logger
from app.rag.hmdp_rag import sync_blog_by_id

RAG_MQ_SYNC = os.getenv("RAG_MQ_SYNC", "true").lower() == "true"
EXCHANGE = os.getenv("RABBITMQ_CACHE_SYNC_EXCHANGE", "cache.sync.fanout")
QUEUE = os.getenv("RAG_SYNC_QUEUE", "agent.rag.sync.queue")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER",)
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST")


async def _on_message(message: aio_pika.abc.AbstractIncomingMessage) -> None:
    try:
        body = json.loads(message.body.decode("utf-8"))
        if body.get("type") == "BLOG" and body.get("id"):
            result = await asyncio.to_thread(sync_blog_by_id, int(body["id"]))
            logger.info("RAG 增量同步完成: %s", result)
        await message.ack()
    except Exception as exc:
        logger.error("RAG 增量同步失败，重新入队: %s", exc)
        await message.reject(requeue=True)


async def _consume_forever() -> None:
    connection = await aio_pika.connect_robust(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        login=RABBITMQ_USER,
        password=RABBITMQ_PASSWORD,
        virtualhost=RABBITMQ_VHOST,
    )
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        exchange = await channel.declare_exchange(
            EXCHANGE,
            aio_pika.ExchangeType.FANOUT,
            durable=True,
        )
        queue = await channel.declare_queue(QUEUE, durable=True)
        await queue.bind(exchange)
        await queue.consume(_on_message)
        logger.info("RAG MQ 增量同步已启动: %s -> %s", EXCHANGE, QUEUE)
        # aio‑pika 9.x 替代 await connection.closing
        await asyncio.Future()


async def start_rag_sync_consumer() -> None:
    if not RAG_MQ_SYNC:
        logger.info("RAG MQ 增量同步已关闭")
        return
    while True:
        try:
            await _consume_forever()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("RAG MQ 连接失败，5 秒后重试: %s", exc)
            await asyncio.sleep(5)