# agblearn 项目约定

> 本文件记录项目核心约定，供 AI 助手和跨设备开发参考。

## 项目定位

跟随《深入理解 AI Agent》（https://bojieli.github.io/ai-agent-book）的学习实验仓库。目标是**最小可运行的教学实现**，聚焦概念理解而非工程完备性。

## 项目结构

```
agblearn/
├── common/               # 跨章节共用代码（API 配置等）
├── chapterN/             # 每章一个目录，实验脚本带序号（NN_主题.py）
├── .env                  # API 密钥（git 忽略）
├── AGENTS.md             # AI 编码助手工作约定（详细版）
└── pyproject.toml        # uv 项目定义
```

- **章节脚本命名**：`NN_主题.py`（如 `01_agent.py`、`02_agent_llm.py`），两位序号保证排序即学习顺序
- **共用代码**：配置统一走 `common/config.py`，章节脚本用 `sys.path` 引导后 `from common.xxx import`
- **禁止**：各章节新建配置文件、硬编码密钥

## 章节复习 README（chapterN/README.md）

每个章节目录下维护一个 README.md 作为**复习卡**（不是文档堆砌），每学完一节就同步补充。

```markdown
# 第 N 章 · 主题
> 复习卡：主线 → 记忆卡 → 对照表 → 自测。比喻是锚点，数字是钩子。
## 一章主线     ← 一段话 + ASCII 演进图串起全章逻辑（章末回写/修订）
## 记忆卡       ← 每节一小节，内部分行标注三个维度：
                  比喻/类比（马具=Dockerfile）、数字钩子（384 token、跌 30%）、
                  实证（实验里实测过什么，含踩坑的原始报错）
## 对照表       ← 易混淆概念并排对比：KV vs Prompt Cache、五组成↔四角色、
                  铁律↔反模式、防御三板斧；模式/术语带"后端熟语"列与"后续章节落点"列
## 自测         ← 3-7 题"遮住括号自测"式，答案藏括号；优先选实验里实测过的坑
```

维护规则：
- **每次学完新的一节（Day），同步追加/修订该章 README**——实验脚本提交前先更新复习卡
- 记忆卡按维度分行标注（比喻/数字钩子/实证），不写成一段长文；不复制代码、不复述细节
- 对照表用于"易混淆对"：凡是在学习中发现两个概念容易混，就并排成表
- 原书金句优先直接引用（"聪明新员工读完还不知道怎么做，模型也不知道"）
- 一章学完后回写"一章主线"（含 ASCII 演进图），检验能否一段话讲清全章逻辑；讲不清 = 还没学透
- 自测题答案藏在括号里，方便遮住自测；题目优先选"实验里实测过的坑"（如漏发 assistant → 400）

## 技术栈

### LLM/Agent 场景使用 LangChain 生态

- `langchain-core` + `langchain-openai`（或对应厂商的 langchain 包）
- `ChatOpenAI` 作为 LLM 统一接口
- `@tool` 装饰器定义工具，`bind_tools` 绑定到 LLM
- 教学场景优先手写 `while` 循环展示 ReAct 原理，也可用 `create_react_agent`
- **使用最新语法**：优先使用 LangChain 最新版本的语法和功能（如 LCEL、`@tool`、`bind_tools`）；遇到较新语法时用简短注释说明（如 `# LCEL 链式调用`）

### 配置

- 接口协议：OpenAI 兼容，当前用 DeepSeek（`deepseek-chat`）
- 配置变量：`LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`（根目录 `.env`）
- 厂商中立：换厂商只改 `.env`，LangChain 抽象层屏蔽差异

## 代码风格

### 文件头说明概念

每个实验脚本头部用 docstring 讲清：
- 本实验讲什么概念
- 与前面实验什么关系
- 概念落在代码哪里

范例：`chapter1/03_harness.py`

### 代码内注释从简

- 中文注释，只在概念落点或必要处点到"为什么"
- 不逐行复述代码

### 语言生动易记

本仓库核心是学习，说明性内容用容易理解、方便记忆的生动语言：
- 多用比喻和类比（如 Harness 原意"马具"：不是限制马跑，而是把力量引导成可靠执行）
- 拒绝干瘪的术语复读

### 教学优先

- 每个实验最小可运行、聚焦单一概念
- 不加与当前实验无关的错误处理、抽象或兼容层
- 能在 50 行内讲清的不写 200 行

## 验证要求

- 新增/修改实验脚本后必须实际运行验证
- 空配置应友好提示、假 key 应到达真实 API 报 401
- 未经真实 key 端到端验证时，明确告知用户

## 常用命令

```bash
uv sync                          # 安装依赖
uv add <pkg>                     # 新增依赖
uv run chapterN/xxx.py           # 运行实验脚本
uv run python -X utf8 <script>   # 管道输出中文时（Windows GBK）
```

## 相关资源

- 原书在线阅读：https://bojieli.github.io/ai-agent-book
- 原书仓库：https://github.com/bojieli/ai-agent-book
- 详细约定：[AGENTS.md](../AGENTS.md)
