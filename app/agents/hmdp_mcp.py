"""MCP 工具加载：time 默认启用，kiwi 可选。"""

import os

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.common.logger import logger


async def get_mcp_tools() -> list:
    connections = {}

    if os.getenv("ENABLE_TIME_MCP", "true").lower() == "true":
        connections["time"] = {
            "transport": "stdio",
            "command": "uvx",
            "args": ["mcp-server-time", "--local-timezone=Asia/Shanghai"],
        }

    if os.getenv("ENABLE_KIWI_MCP", "false").lower() == "true":
        connections["kiwi"] = {
            "transport": "http",
            "url": "https://mcp.kiwi.com",
        }

    if not connections:
        return []

    client = MultiServerMCPClient(connections, tool_name_prefix=True)
    tools = []
    for server_name in connections:
        try:
            tools.extend(await client.get_tools(server_name=server_name))
        except Exception as exc:
            logger.error(f"MCP {server_name} 加载失败: {exc}")
    return tools