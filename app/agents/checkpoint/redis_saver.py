"""零 Redis 模块依赖的 LangGraph checkpoint saver：只用 Hash / ZSet / Set / String（官方实现要求 Redis Stack 的 ``FT.CREATE``）。
key（``p`` = 前缀，各段经 URL 编码）：``{p}:ckpt:{t}:{ns}:{c}`` / ``{p}:ckptids:{t}:{ns}`` / ``{p}:wk:{t}:{ns}:{c}:{k}`` / ``{p}:wtids:{t}:{ns}:{c}`` / ``{p}:ns:{t}`` / ``{p}:threads``。
ZSet 全员同分、靠成员字典序当有序索引；不设 TTL（清理走 ``adelete_thread``，即 ``DELETE /sessions/{id}``），``alist`` 跨 thread 顺序与官方 SQL 版略有差异。"""

from __future__ import annotations

import asyncio
import base64
import json
import random
from collections.abc import AsyncIterator, Awaitable, Iterator, Sequence
from typing import Any, cast
from urllib.parse import quote

from app.common.logger import logger

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

_SEP = b"\x00"


def _pack(type_tag: str, payload: bytes) -> bytes:
    """``type 标签 + \\x00 + base64(序列化结果)``。

    base64 是为了让 ``redis-cli HGET`` 直接看到内容（二进制本也能存，Redis 二进制安全），代价体积约 +33%。"""
    return type_tag.encode("utf-8") + _SEP + base64.b64encode(payload)


def _unpack(blob: bytes) -> tuple[str, bytes]:
    type_tag, _, encoded = blob.partition(_SEP)
    return type_tag.decode("utf-8"), base64.b64decode(encoded)


def _pack_write(channel: str, type_tag: str, payload: bytes) -> bytes:
    return channel.encode("utf-8") + _SEP + _pack(type_tag, payload)


def _unpack_write(blob: bytes) -> tuple[str, str, bytes]:
    channel, _, rest = blob.partition(_SEP)
    type_tag, value = _unpack(rest)
    return channel.decode("utf-8"), type_tag, value


def _as_text(raw: Any) -> str:
    """redis-py 在 ``decode_responses=False`` 下返回 bytes，这里统一成 str。"""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _metadata_matches(metadata: dict[str, Any], criteria: dict[str, Any] | None) -> bool:
    """复刻 SQL 版 ``search_where`` 的语义：SQL 走 ``json_extract(metadata, '$.key') <op> ?``，
    布尔值会被转成 1/0；这里对已解析的 dict 做等价判断。"""
    if not criteria:
        return True
    for key, expected in criteria.items():
        actual = metadata.get(key)
        if expected is None:
            if actual is not None:
                return False
        elif isinstance(expected, bool):
            if actual != (1 if expected else 0) and actual is not expected:
                return False
        else:
            if actual != expected:
                return False
    return True


class AsyncRedisCheckpointSaver(BaseCheckpointSaver[str]):
    """只用基础 Redis 命令实现的异步 checkpoint saver；用法：``AsyncRedisCheckpointSaver(client, prefix=...)``
    → ``await asetup()`` → ``builder.compile(checkpointer=saver)``，停机时 ``await aclose()``。"""

    def __init__(
        self,
        redis_client: Any,
        *,
        prefix: str = "agent:checkpoint",
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self.redis = redis_client
        self.prefix = prefix
        self._setup_done = False
        # 同步方法需要用「构造时所在的事件循环」把协程调度出去（与官方
        # AsyncSqliteSaver 的做法一致）。构造时不在循环里则退化为按需新建循环。
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    # ------------------------------------------------------------------
    # key 构造
    # ------------------------------------------------------------------
    def _seg(self, part: str) -> str:
        """URL 编码单个 key 段：``checkpoint_ns`` 形如 ``node:task_id`` 本身带 ``:``，
        不编码会让不同 (thread, ns) 组合撞到同一个 key。"""
        return quote(part, safe="")

    def _ckpt_key(self, thread: str, ns: str, cid: str) -> str:
        return f"{self.prefix}:ckpt:{self._seg(thread)}:{self._seg(ns)}:{self._seg(cid)}"

    def _ckpt_ids_key(self, thread: str, ns: str) -> str:
        return f"{self.prefix}:ckptids:{self._seg(thread)}:{self._seg(ns)}"

    def _writes_key(self, thread: str, ns: str, cid: str, task_id: str) -> str:
        return (f"{self.prefix}:wk:{self._seg(thread)}:{self._seg(ns)}"
                f":{self._seg(cid)}:{self._seg(task_id)}")

    def _write_task_ids_key(self, thread: str, ns: str, cid: str) -> str:
        return f"{self.prefix}:wtids:{self._seg(thread)}:{self._seg(ns)}:{self._seg(cid)}"

    def _namespaces_key(self, thread: str) -> str:
        return f"{self.prefix}:ns:{self._seg(thread)}"

    def _threads_key(self) -> str:
        return f"{self.prefix}:threads"

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def asetup(self) -> None:
        """只做一次连通性探测，不建索引/建表 —— 官方 ``AsyncRedisSaver`` 会在这里 ``FT.CREATE``，普通 Redis 直接失败。"""
        if self._setup_done:
            return
        await self.redis.ping()
        self._setup_done = True

    def setup(self) -> None:
        self._run_sync(self.asetup())

    async def aclose(self) -> None:
        await self.redis.aclose()

    # ------------------------------------------------------------------
    # 异步：读
    # ------------------------------------------------------------------
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        await self.asetup()
        thread = str(config["configurable"]["thread_id"])
        ns = config["configurable"].get("checkpoint_ns", "")
        cid = get_checkpoint_id(config)
        if cid is None:
            # 没指定 checkpoint_id → 取该 (thread, ns) 下最新的一个
            latest = await self.redis.zrevrange(self._ckpt_ids_key(thread, ns), 0, 0)
            if not latest:
                return None
            cid = _as_text(latest[0])

        raw = await self.redis.hgetall(self._ckpt_key(thread, ns, cid))
        if not raw:
            return None
        parent = _as_text(raw.get(b"parent")) or None
        type_tag, payload = _unpack(raw[b"data"])
        metadata_json = _as_text(raw.get(b"metadata"))

        out_config: RunnableConfig = config
        if get_checkpoint_id(config) is None:
            # 与官方 AsyncSqliteSaver 行为保持一致：只有「按最新查询」时才把
            # 解析出来的 checkpoint_id 回填进 config
            out_config = {
                "configurable": {
                    "thread_id": thread,
                    "checkpoint_ns": ns,
                    "checkpoint_id": cid,
                }
            }

        return CheckpointTuple(
            out_config,
            cast(Checkpoint, self.serde.loads_typed((type_tag, payload))),
            cast(CheckpointMetadata, json.loads(metadata_json) if metadata_json else {}),
            (
                {
                    "configurable": {
                        "thread_id": thread,
                        "checkpoint_ns": ns,
                        "checkpoint_id": parent,
                    }
                }
                if parent
                else None
            ),
            await self._load_writes(thread, ns, cid),
        )

    async def _load_writes(self, thread: str, ns: str, cid: str) -> list[tuple[str, str, Any]]:
        """读取 pending writes，顺序等价于 SQL 的 ``ORDER BY task_id, idx``。"""
        task_ids = await self.redis.zrange(self._write_task_ids_key(thread, ns, cid), 0, -1)
        if not task_ids:
            return []
        writes: list[tuple[str, str, Any]] = []
        for raw_task in task_ids:
            task_id = _as_text(raw_task)
            fields = await self.redis.hgetall(self._writes_key(thread, ns, cid, task_id))
            if not fields:
                continue
            # Hash field 是无序的，按 idx 数值升序排回来
            for _, blob in sorted(fields.items(), key=lambda kv: int(_as_text(kv[0]))):
                channel, type_tag, payload = _unpack_write(blob)
                writes.append((task_id, channel, self.serde.loads_typed((type_tag, payload))))
        return writes

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        await self.asetup()
        before_id = get_checkpoint_id(before) if before is not None else None

        if config is not None:
            thread = str(config["configurable"]["thread_id"])
            ns_in_config = config["configurable"].get("checkpoint_ns")
            namespaces = (
                [ns_in_config]
                if ns_in_config is not None
                else [_as_text(n) for n in await self.redis.smembers(self._namespaces_key(thread))]
            )
            # 按 (thread, ns) 分组，组内再倒序；同一个组天然有序，跨组顺序不稳定
            targets = [(thread, ns) for ns in sorted(namespaces)]
        else:
            threads = sorted(_as_text(t) for t in await self.redis.smembers(self._threads_key()))
            targets = []
            for thread in threads:
                for ns in sorted(
                    _as_text(n) for n in await self.redis.smembers(self._namespaces_key(thread))
                ):
                    targets.append((thread, ns))

        only_id = get_checkpoint_id(config) if config is not None else None
        yielded = 0
        for thread, ns in targets:
            cids = await self.redis.zrevrange(self._ckpt_ids_key(thread, ns), 0, -1)
            for raw_cid in cids:
                if limit is not None and yielded >= limit:
                    return
                cid = _as_text(raw_cid)
                if only_id is not None and cid != only_id:
                    continue
                # SQL 语义：checkpoint_id < before_id
                if before_id is not None and cid >= before_id:
                    continue
                raw = await self.redis.hgetall(self._ckpt_key(thread, ns, cid))
                if not raw:
                    # 索引成员比 Hash 活得久（例如被 TTL 单独清掉），跳过并提示
                    continue
                metadata_json = _as_text(raw.get(b"metadata"))
                metadata = json.loads(metadata_json) if metadata_json else {}
                if not _metadata_matches(metadata, filter):
                    continue
                parent = _as_text(raw.get(b"parent")) or None
                type_tag, payload = _unpack(raw[b"data"])
                yielded += 1
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": thread,
                            "checkpoint_ns": ns,
                            "checkpoint_id": cid,
                        }
                    },
                    cast(Checkpoint, self.serde.loads_typed((type_tag, payload))),
                    cast(CheckpointMetadata, metadata),
                    (
                        {
                            "configurable": {
                                "thread_id": thread,
                                "checkpoint_ns": ns,
                                "checkpoint_id": parent,
                            }
                        }
                        if parent
                        else None
                    ),
                    await self._load_writes(thread, ns, cid),
                )

    # ------------------------------------------------------------------
    # 异步：写
    # ------------------------------------------------------------------
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        await self.asetup()
        thread = str(config["configurable"]["thread_id"])
        ns = config["configurable"].get("checkpoint_ns", "")
        cid = checkpoint["id"]
        parent = config["configurable"].get("checkpoint_id") or ""
        type_tag, payload = self.serde.dumps_typed(checkpoint)
        metadata_json = json.dumps(
            get_checkpoint_metadata(config, metadata), ensure_ascii=False
        )

        # 用事务把「写 checkpoint + 更新倒序索引 + 记录线程/命名空间」打包，
        # 避免出现「Hash 写成功但索引没更新」的中间态
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(
                self._ckpt_key(thread, ns, cid),
                mapping={
                    b"parent": parent.encode("utf-8"),
                    b"type": type_tag.encode("utf-8"),
                    b"data": _pack(type_tag, payload),
                    b"metadata": metadata_json.encode("utf-8"),
                },
            )
            pipe.zadd(self._ckpt_ids_key(thread, ns), {cid: 0})
            pipe.sadd(self._namespaces_key(thread), ns)
            pipe.sadd(self._threads_key(), thread)
            # Redis 不可用时不能把异常抛给上层：异常会一路穿到 generate_sse 的 async for，导致整条 SSE 中断、接口 500 且 traceback 吐给客户端。
            # checkpoint 是可丢的旁路（丢只本轮不落库，下轮照常），故降级为 error 日志 + 放行。
            try:
                await pipe.execute()
            except Exception as exc:
                logger.error(
                    "checkpoint 写入失败（已降级，本轮不落库）: thread=%s ckpt=%s err=%s: %s",
                    thread, cid, type(exc).__name__, exc,
                )

        return {
            "configurable": {
                "thread_id": thread,
                "checkpoint_ns": ns,
                "checkpoint_id": cid,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread = str(config["configurable"]["thread_id"])
        ns = config["configurable"].get("checkpoint_ns", "")
        cid = str(config["configurable"]["checkpoint_id"])
        if not writes:
            return

        # 与官方实现保持一致的覆盖语义：
        # 全部 channel 都在 WRITES_IDX_MAP 里 → 可覆盖（HSET）；
        # 否则 → 已存在就不覆盖（HSETNX），对应 SQL 的 INSERT OR REPLACE / INSERT OR IGNORE
        replace = all(channel in WRITES_IDX_MAP for channel, _ in writes)

        async with self.redis.pipeline(transaction=True) as pipe:
            for idx, (channel, value) in enumerate(writes):
                write_idx = WRITES_IDX_MAP.get(channel, idx)
                type_tag, payload = self.serde.dumps_typed(value)
                blob = _pack_write(channel, type_tag, payload)
                key = self._writes_key(thread, ns, cid, task_id)
                if replace:
                    pipe.hset(key, str(write_idx), blob)
                else:
                    pipe.hsetnx(key, str(write_idx), blob)
            pipe.zadd(self._write_task_ids_key(thread, ns, cid), {task_id: 0})
            pipe.sadd(self._namespaces_key(thread), ns)
            pipe.sadd(self._threads_key(), thread)
            # 同 `aput`：节点写通道失败同样只丢"这一轮的可续跑状态"，不该打断回答。
            try:
                await pipe.execute()
            except Exception as exc:
                logger.error(
                    "checkpoint writes 写入失败（已降级，本轮不落库）: thread=%s ckpt=%s task=%s err=%s: %s",
                    thread, cid, task_id, type(exc).__name__, exc,
                )

    # ------------------------------------------------------------------
    # 异步：删除
    # ------------------------------------------------------------------
    async def adelete_thread(self, thread_id: str) -> None:
        thread = str(thread_id)
        namespaces = [_as_text(n) for n in await self.redis.smembers(self._namespaces_key(thread))]
        keys: list[str] = [self._namespaces_key(thread)]
        for ns in namespaces:
            cids = await self.redis.zrevrange(self._ckpt_ids_key(thread, ns), 0, -1)
            keys.append(self._ckpt_ids_key(thread, ns))
            for raw_cid in cids:
                cid = _as_text(raw_cid)
                keys.append(self._ckpt_key(thread, ns, cid))
                keys.append(self._write_task_ids_key(thread, ns, cid))
                for raw_task in await self.redis.zrange(
                    self._write_task_ids_key(thread, ns, cid), 0, -1
                ):
                    keys.append(self._writes_key(thread, ns, cid, _as_text(raw_task)))
        if keys:
            # 分批 DEL，避免单次 pipeline 撑爆
            for start in range(0, len(keys), 500):
                await self.redis.delete(*keys[start:start + 500])
        await self.redis.srem(self._threads_key(), thread)

    # ------------------------------------------------------------------
    # 版本号
    # ------------------------------------------------------------------
    def get_next_version(self, current: str | None, channel: None) -> str:
        """与官方 AsyncSqliteSaver 同一算法：``{递增序号:032}.{随机数:016}``。
        同一次写入内序号相同、随机后缀不同 ⇒ 单调递增且同版本内可区分先后（LangGraph 版本比较依赖此约定）。"""
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    # ------------------------------------------------------------------
    # 同步桥接
    # ------------------------------------------------------------------
    def _run_sync(self, coro: Awaitable[Any]) -> Any:
        if self._loop is None:
            return asyncio.run(coro)
        running: asyncio.AbstractEventLoop | None
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            raise RuntimeError(
                "AsyncRedisCheckpointSaver 的同步方法不能在事件循环线程内调用，"
                "请改用异步接口（aget_tuple / alist / aput / aput_writes / adelete_thread）"
            )
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self._run_sync(self.aget_tuple(config))

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self._run_sync(self.aput(config, checkpoint, metadata, new_versions))

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._run_sync(self.aput_writes(config, writes, task_id, task_path))

    def delete_thread(self, thread_id: str) -> None:
        self._run_sync(self.adelete_thread(thread_id))

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield self._run_sync(aiter_.__anext__())  # type: ignore[union-attr]
            except StopAsyncIteration:
                break
