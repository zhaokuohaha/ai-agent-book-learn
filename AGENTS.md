# AGENTS.md — AI 编码助手工作约定

本仓库是跟随《深入理解 AI Agent》（bojieli/ai-agent-book，本地路径 `D:\Github\ai-agent-book`）的学习实验仓库。用户在逐章实现书中的实验代码。原书实验可对照阅读，但本仓库定位是**最小可运行的教学实现**，不照搬原书工程结构。

## 项目结构约定

- `chapterN/`：每章一个目录，实验脚本可直接运行（`uv run chapterN/xxx.py`）
- 实验脚本命名带序号：`NN_主题.py`（如 `01_agent.py`、`02_agent_llm.py`），两位序号即实验先后顺序——无序号时文件按字母排序会打乱学习顺序
- `common/`：跨章节共用代码。API 配置统一在 `common/config.py`，从根目录 `.env` 读取
- 章节脚本导入共用代码时，开头需要 sys.path 引导（uv 按脚本方式运行，无包安装）：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import API_KEY, BASE_URL, MODEL
```

- **禁止**在各章节新建独立配置文件或直接解析 `.env`——一律走 `common/config.py`
- **禁止**把密钥写进任何代码或让 `.env` 进入 git（`.gitignore` 已覆盖 `.env`）

## API / 模型

- **LLM/Agent 场景使用 LangChain 生态**：`langchain-core` + `langchain-openai`（或对应厂商的 langchain 包），统一接口便于切换模型和工具
- **使用最新语法**：优先使用 LangChain 最新版本的语法和功能（如 LCEL、`@tool` 装饰器、`bind_tools` 等）；遇到较新或不常见的语法时，用简短注释说明其作用（如 `# LCEL 链式调用`）
- 接口协议：OpenAI 兼容，当前用 DeepSeek（`deepseek-chat`）
- 配置变量：`LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`（根目录 `.env`，环境变量可覆盖）
- 厂商中立：换厂商只改 `.env`，不改代码；LangChain 抽象层屏蔽厂商差异
- 模型输出是动态的：验证脚本时检查执行轨迹（工具是否被调用、闭环是否走通），不要断言模型的固定文案

## 常用命令

```bash
uv sync                          # 安装依赖
uv add <pkg>                     # 新增依赖（写入 pyproject.toml）
uv run chapter1/01_agent.py      # 运行实验脚本
uv run python -X utf8 <script>   # 需要管道/重定向输出中文时（Windows GBK）
```

## 代码风格

- **文件头说明概念**：每个实验脚本头部用 docstring 讲清核心概念与代码重点——本实验讲什么概念、与前面实验什么关系、概念落在代码哪里；读者应先看头、再看码（范例：`chapter1/03_harness.py`）
- **代码内注释从简**：中文注释，只在概念落点或必要处点到"为什么"，不逐行复述代码
- **语言生动易记**：本仓库核心是学习，docstring/注释等一切说明性内容用容易理解、方便记忆的生动语言——多用比喻和类比（如 Harness 原意"马具"：不是限制马跑，而是把力量引导成可靠执行；Model 是能力强的第三方服务，Harness 是你的网关+治理层），拒绝干瘪的术语复读
- 教学优先：每个实验最小可运行、聚焦单一概念；不加与当前实验无关的错误处理、抽象或兼容层
- **遵循 Python 规范与最佳实践，注重代码优雅**：教学简洁不等于随意——遵循 PEP 8；优先惯用法（推导式、标准库、解包）而非手写循环；同一逻辑只写一遍，重复即坏味道；函数单一职责，依赖经参数/构造器传入而非依赖模块级全局；模块级副作用最小化（模型实例化等放进 `__main__`）；在"最小可运行"约束下选最优雅的写法，但不为优雅引入多余抽象
- Agent 工具模式：使用 LangChain 的 `@tool` 装饰器或 `StructuredTool`，工具定义与 schema 合一
- Agent 循环模式：可使用 LangChain 的 `create_react_agent` 或手写 `while` 循环 + `bind_tools`；教学场景优先手写循环以展示原理
- **类型标注按功能分层**：`@tool` / `StructuredTool` 函数必须完整标注参数与返回类型——LangChain 依赖注解生成工具 schema，缺标注无法生成（功能需求，非风格选择）；其余函数/方法参数默认可省（教学脚本聚焦概念、减少噪音），只在类型不明显、容易误传处点到（如 `tool_list: tuple[BaseTool, ...]`）；返回值标注一般省略；不回头补标旧脚本

## 环境

- Windows + Git Bash + uv；Python ≥ 3.12
- `pyproject.toml` 中 `[tool.uv] package = false`（纯脚本项目，不打包安装）
- 交互式终端中文正常；**管道捕获中文输出会 GBK 乱码**，验证时用 `uv run python -X utf8 <script>`
- `uv run -X utf8 <script>` 无效（Python 参数须跟在 `python` 后）

## 验证要求

- 新增/修改实验脚本后必须实际运行验证（空配置应友好提示、假 key 应到达真实 API 报 401，即接线正确）
- 未经用户真实 key 的端到端验证时，明确告知用户"未用真实凭据跑通"
