# agblearn — AI Agent 学习实验仓库

跟随开源书 [《深入理解 AI Agent：设计原理与工程实践》](https://github.com/bojieli/ai-agent-book)（本地路径 `D:\Github\ai-agent-book`）的学习实验代码。

原书核心公式：**Agent = LLM + 上下文 + 工具**。本仓库按章节亲手实现书中实验的最小可运行版本，目标是**用最少的代码把每个概念跑通**，而非复刻原书仓库（原书实验代码在 `ai-agent-book/chapterN/`，可对照阅读）。

## 学习进度

| 章 | 主题 | 本仓库实现 | 状态 |
| :--: | --- | --- | :--: |
| 1 | Agent 入门（LLM + 上下文 + 工具） | `chapter1/` | ✅ |
| 2 | 上下文工程 | `chapter2/` | 🚧 |
| 3 | 用户记忆和知识库 | `chapter3/` | 🚧 |
| 4 | 工具（MCP 协议等） | — | |
| 5 | Coding Agent 与通用 Agent | — | |
| 6 | 交互：观察与动作空间的扩展 | — | |
| 7 | Agent 的评估 | — | |
| 8 | 模型后训练 | — | |
| 9 | Agent 的持续进化 | — | |
| 10 | 多 Agent 协作 | — | |

## 目录结构

```
agblearn/
├── common/               # 跨章节共用代码（API 配置等）
├── chapter1/             # 第 1 章实验（Agent 基础知识）
├── chapter2/             # 第 2 章实验（上下文工程）
├── ...                   # 后续章节按需添加
├── .env                  # API 密钥配置（git 忽略，需自行填写）
├── AGENTS.md             # AI 编码助手工作约定
└── pyproject.toml        # uv 项目定义与依赖
```

每个 `chapterN/` 目录包含该章的实验脚本，带序号命名（如 `01_xxx.py`、`02_xxx.py`），按学习顺序排列。

## 环境准备

- Python ≥ 3.12（uv 会自动下载）
- [uv](https://docs.astral.sh/uv/getting-started/installation/) 包管理器

```bash
uv sync   # 创建 .venv 并安装依赖（langchain、python-dotenv 等）
```

## API 配置

模型走 **OpenAI 兼容接口**（DeepSeek / 通义 / Kimi / OpenAI 等均可），配置集中在根目录 `.env`（已被 git 忽略，密钥不会进仓库）：

```bash
# .env
LLM_API_KEY=sk-xxx          # DeepSeek 在 https://platform.deepseek.com/api_keys 创建
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat     # deepseek-chat (V3) 支持工具调用；deepseek-reasoner 为 R1 推理模型
```

- 配置由 `common/config.py` 统一加载（python-dotenv），**所有章节脚本共用**，不在各章节重复配置
- 同名环境变量优先级高于 `.env` 文件值
- 更换厂商时只改 `.env` 三个值，代码不用动

## 运行示例

```bash
# 第 1 章实验示例
uv run chapter1/01_agent.py       # 规则版：无需 API key，演示 Agent 骨架
uv run chapter1/02_agent_llm.py   # LLM 版：需先填好 .env（使用 LangChain）
uv run chapter1/06_react.py       # ReAct 循环：while 循环 Agent
```

更多实验脚本见各 `chapterN/` 目录，脚本头部 docstring 有详细说明。

## 代码规范

- **每章一个目录**：`chapterN/`，实验脚本带序号命名（`NN_主题.py`），可独立运行
- **共用代码进 `common/`**：配置、工具函数等跨章节复用的代码放这里
- **LLM/Agent 场景使用 LangChain 生态**：统一接口，便于切换模型和工具
- **教学优先**：代码保持最小可运行，每个实验聚焦一个概念
- **配置只走 `.env`**：任何脚本不硬编码密钥；读取一律通过 `common/config.py`
- **依赖管理**：新依赖用 `uv add <pkg>` 添加，不用 pip 手动装
- 详细约定见 [AGENTS.md](AGENTS.md)

## 常见问题

**终端中文乱码（Windows）**：Windows 控制台默认 GBK，重定向/管道输出中文可能乱码。交互式终端一般正常；如需管道输出，用：

```bash
uv run python -X utf8 chapter1/02_agent_llm.py
```

**401 / 403 报错**：401 = key 无效（检查 `.env`）；403 = 网关拒绝（确认 BASE_URL 指向正确的厂商地址）。

## 相关资源

- 原书在线阅读：https://bojieli.github.io/ai-agent-book
- 原书仓库：https://github.com/bojieli/ai-agent-book
- 本仓库学习约定见 [AGENTS.md](AGENTS.md)
