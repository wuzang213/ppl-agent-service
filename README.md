# 本地生活智能问答Agent

## 项目简介

对接本地生活微服务体系的AI对话服务，基于网关统一鉴权转发，支持生活推荐、日常助手双模式对话。基于LangGraph构建智能Agent，融合RAG检索与业务工具调用，实现智能化问答、流式交互与多轮会话持久化。

Agent 采用**多节点图编排**而非单模型全流程：入口用轻量模型做意图识别与实体抽取，再按意图分流——闲聊走轻链路流式直出，推荐走「Grounding 解析店铺 → 并行 Worker 取数 → 主模型汇总」，整条推荐链路只调用 2 次大模型。

## 技术栈

Python、FastAPI、LangChain、LangGraph、Chroma、BM25、RRF、Cross-Encoder、RabbitMQ、Nacos、SSE

## 核心功能

- **意图路由与并行编排**：LangGraph 条件边做主骨架、`Send` 做店铺级并行扇出；流程约束写进代码而非硬塞提示词，链路可控且首字更快。

- **高精度RAG问答**：融合多路检索与重排算法，MQ增量同步知识库，搭建问答评估体系，优化问答准确率与忠实度。

- **长会话历史压缩**：对话超过阈值（默认 8 轮）后，先用轻量模型把最早的若干轮压缩成滚动摘要再删除原文，实测 token 节省约 92%；压缩失败时沿用旧摘要，绝不因压缩异常丢失记忆。

- **服务发现与优雅降级**：出站调用 Java 微服务优先使用 `.env` 中已验证可达的静态地址，未配置时才回落到 Nacos 拉取健康实例；Nacos 不可用不阻塞任何请求。

- **会话权限管理**：支持会话持久化存储、动态提示词切换，结合网关实现会话级权限校验。

- **可观测流式交互**：全链路指标采集监控，支持SSE流式输出、对话中断，优化用户交互体验。

## Agent 架构

```
START → trim_history → entry_router ─┬─ daily     → daily_chat → END
                                     └─ recommend → grounding ─┬─ 有 shop_id → dispatch_workers → shop_worker ×N → summarize → END
                                                               ├─ 无 shop_id → nearby_worker → 按需扇出 → summarize → END
                                                               └─ 无法解析   → clarify → END
```

| 节点 | 作用 | 是否调 LLM |
|---|---|---|
| `trim_history` | 历史超过阈值时压缩早期消息为滚动摘要并删除原文 | 仅超阈值时（轻量模型） |
| `entry_router` | 意图识别 + 实体抽取，一次调用输出结构化路由计划 | 是（轻量模型） |
| `daily_chat` | 日常闲聊，流式直出 | 是（轻量模型） |
| `grounding` | 用规则/查库把用户表述解析成 `shop_ids`（含指代消解、店名规范化） | 否 |
| `dispatch_workers` | 以 `Send` 并行扇出，每家店一个 Worker | 否 |
| `shop_worker` / `nearby_worker` | 取店铺详情、评价、优惠、RAG 片段，只返回紧凑 JSON | 否 |
| `summarize` | 主模型把 Worker 结果整理成自然语言，流式输出 | 是（主模型） |
| `clarify` | 信息不足时向用户追问 | 否 |

**关键设计**

- 推荐链路**只调 2 次 LLM**（入口 + 汇总），中间全是数据节点，对比旧版 5–8 次工具循环；
- 单店/多店由 `len(shop_ids)` 推导，它是事实不是意图，不额外占用一次模型调用；
- `focus_shops` 入 State，解决「这家 / 第二家 / 它」的跨轮指代；
- `worker_outputs` 用完即清（可序列化字符串哨兵 `__CLEAR__`），避免被 checkpoint 带入下一轮导致上下文膨胀。

## 历史压缩

长会话直接硬截断会丢上下文，全量保留又撑爆 token。这里的做法是**截断前先语义压缩**：

1. 消息数超过 `AGENT_HISTORY_KEEP`（默认 8 轮 = 16 条）时，取最早一批待删消息；
2. 轻量模型结合已有摘要压缩成新的 running summary，写入 State 的 `history_summary`；
3. 用 `RemoveMessage` 删除原文，摘要再注入 `entry_router` / `daily_chat` / `summarize` 的提示词，弥补被删掉的早期上下文；
4. 压缩调用异常时**沿用旧摘要**并照常删除原文，保证服务可用。

## 服务发现（Nacos）

出站调用（shop / blog / voucher）的地址解析优先级：

1. `.env` 中显式配置的 `SHOP_SERVICE_URL` / `BLOG_SERVICE_URL` / `VOUCHER_SERVICE_URL` —— **优先**，这是使用者已验证可达的地址；
2. 未配置时，回落到 Nacos 命名服务定时（默认 10s）拉取的**健康实例**，自带负载均衡与故障转移；
3. 都拿不到则返回空，由调用方按原有逻辑处理。

`NACOS_SERVER_ADDRS` 未配置时服务发现静默不启用，后台线程不会启动，最坏情况等价于写死地址。

> 之所以让静态地址优先：Spring Cloud 默认以机器局域网 IP 注册到 Nacos，该地址在 agent 运行时未必可达，动态优先反而会出现「Nacos 有实例却连不通」。

## 环境要求

Python ≥ 3.13，包管理器使用 [uv](https://docs.astral.sh/uv/)。

## 快速开始

```bash
uv sync                 # 安装依赖
cp .env.example .env    # 填入模型 API Key 与各服务地址
uv run python -m app.main
```

服务默认监听 `8001` 端口，可通过 `.env` 中的 `AGENT_HOST` / `AGENT_PORT` 修改。如需 LangGraph 调试面板，可改用 `uv run langgraph dev`。

## 目录结构

```
agent-service/
├── app/
│   ├── main.py                 # FastAPI 入口：路由注册、Nacos 注册、服务发现启停
│   ├── service_discovery.py    # Nacos 服务发现（后台刷新 + 本地缓存 + 静态地址优先）
│   ├── nacos_registry.py       # 本服务实例注册到 Nacos
│   ├── agents/
│   │   ├── graph/              # LangGraph 图
│   │   │   ├── state.py        # 最小状态集
│   │   │   ├── llm.py          # 主模型 / 轻量模型工厂
│   │   │   ├── prompts.py      # 拆分后的短提示词
│   │   │   ├── nodes.py        # 各节点实现（含历史压缩）
│   │   │   └── workflow.py     # build_graph：条件边骨架 + Send 扇出
│   │   ├── hmdp_agent.py       # 对外门面：generate_sse / get_messages / clear_messages
│   │   ├── hmdp_tools.py       # Java 微服务调用封装（fetch_* 纯函数）
│   │   ├── hmdp_mcp.py         # MCP 工具
│   │   └── hmdp_metrics.py     # 全链路指标采集
│   ├── rag/                    # 检索：Chroma + BM25 + 重排 + MQ 增量同步
│   ├── api/v1/                 # HTTP 接口：对话、会话、OSS 预签名
│   ├── models/                 # 数据模型
│   └── common/                 # 日志
├── db/                         # 运行期数据（checkpoint / 索引 / 指标，已 gitignore）
└── langgraph.json              # LangGraph 调试面板配置
```

## 主要配置项

完整列表见 `.env.example`，以下为架构相关项：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DASHSCOPE_BASE_URL` / `DASHSCOPE_API_KEY` | 无 | 模型服务地址与密钥（OpenAI 兼容协议），缺失时启动即报错 |
| `AGENT_MODEL` | `qwen-plus`（代码内默认） | 主模型，只在汇总节点调用 |
| `AGENT_LIGHT_MODEL` | `qwen-flash`（代码内默认） | 轻量模型，用于意图识别、闲聊、历史压缩 |
| `AGENT_TEMPERATURE` | `0.1` | 采样温度 |
| `AGENT_HISTORY_KEEP` | `8` | 历史压缩阈值（轮），运行时可改 |
| `AGENT_MAX_SHOPS` | `3` | 单次最多并行处理的店铺数 |
| `AGENT_NEARBY_TOP` | `5` | 「附近找店」候选数量 |
| `AGENT_SSE_STATUS` / `AGENT_SSE_DELTA` | `true` / `false` | 是否推送进度事件 / 是否逐 token 推送 |
| `SHOP/BLOG/VOUCHER_SERVICE_URL` | `http://localhost:8082` 等 | Java 微服务静态地址，优先使用；留空则走 Nacos 发现 |
| `NACOS_SERVER_ADDRS` | 无 | Nacos 地址，留空则不启用服务发现 |

## 关联项目

- [ppl-cloud](https://github.com/wuzang213/ppl-cloud)：本地生活微服务后端。本项目的业务工具调用依赖其中的店铺、博客、优惠券服务，需先启动并通过网关暴露接口。

> 本项目与 Nacos 的关系是**双向的**：启动时把自身注册为 `agent-service` 供网关以 `lb://agent-service` 转发；同时可读取 Nacos 上 Java 服务的健康实例用于出站调用。但它**不读取任何配置 DataId**，自身配置全部来自 `.env`。所需的 Nacos 共享配置见 ppl-cloud 仓库的 `nacos-config/` 目录。
