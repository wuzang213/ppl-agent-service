# 本地生活智能问答Agent

## 项目简介

对接本地生活微服务体系的AI对话服务，基于网关统一鉴权转发，支持生活推荐、日常助手双模式对话。基于LangGraph构建智能Agent，融合RAG检索与业务工具调用，实现智能化问答、流式交互与多轮会话持久化。

## 技术栈

Python、FastAPI、LangChain、LangGraph、Chroma、BM25、RRF、Cross\-Encoder、RabbitMQ、Nacos、SSE

## 核心功能

- **高精度RAG问答**：融合多路检索与重排算法，MQ增量同步知识库，搭建问答评估体系，优化问答准确率与忠实度。

- **工程化Agent**：基于LangGraph实现任务编排、调用重试限流；优化长会话Token开销，封装Java微服务调用，打通AI与业务服务。

- **会话权限管理**：支持会话持久化存储、动态提示词切换，结合网关实现会话级权限校验。

- **可观测流式交互**：全链路指标采集监控，支持SSE流式输出、对话中断，优化用户交互体验。

## 环境要求

Python ≥ 3.13，包管理器使用 [uv](https://docs.astral.sh/uv/)。

## 快速开始

```bash
uv sync                 # 安装依赖
cp .env.example .env    # 填入模型 API Key 与各服务地址
uv run python -m app.main
```

服务默认监听 `8001` 端口，可通过 `.env` 中的 `AGENT_HOST` / `AGENT_PORT` 修改。如需 LangGraph 调试面板，可改用 `uv run langgraph dev`。

## 关联项目

- [ppl-cloud](https://github.com/wuzang213/ppl-cloud)：本地生活微服务后端。本项目的业务工具调用依赖其中的店铺、博客、优惠券服务，需先启动并通过网关暴露接口。

> 本项目仅向 Nacos 注册实例，供网关以 `lb://agent-service` 转发，不读取 Nacos 中的任何配置 DataId，自身配置全部来自 `.env`。所需的 Nacos 共享配置见 ppl-cloud 仓库的 `nacos-config/` 目录。

