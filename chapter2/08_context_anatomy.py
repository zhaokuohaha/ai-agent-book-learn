"""chapter2/08_context_anatomy.py — 上下文为何是上限：无状态 API 与上下文解剖

核心概念（原书第 2 章 · W2 Day 8）：

永远的新员工
    天才工程师入职：功底深厚，但对产品架构、业务规则、技术债一无所知，很难
    发挥价值。Agent 就是这个"永远的新员工"——每轮醒来都失忆，表现上限不由
    模型智商决定，而由你塞给它的资料决定。所以"中等模型 + 精心组织的上下文"
    常胜过"顶级模型在信息匮乏下盲目摸索"。Coding Agent 最低信息需求三类：
    实时代码上下文、流程规范（Git/提交/CI）、环境信息（测试/部署）。
    推论：对远程协作友好的团队也对 Agent 友好——信息公开、可检索、结构化，
    构建 AI 原生团队先是一场文档化运动（Linux 内核三十年协作的秘诀）。

模型 API 无状态
    像没有 session 的 REST 接口：服务端不保存对话状态，客户端自己维护、每次
    全量重传——更像 JWT 而不是 Cookie-Session。"模型记得我"是错觉，记忆的
    每一帧都是你重新发过去的 messages。本实验用对照实验验证：同一个问题，
    带不带历史，答案天差地别。

上下文构成 = 静态前缀 + 动态轨迹
    system → 系统提示词（开发者规则）             ┐ 静态前缀：全程不变
    tools  → 工具定义（请求顶层独立字段）          ┘
    user      → 用户消息（环境返回的观察）
    assistant → 模型回复（含工具调用请求）         │ 动态轨迹：只增不减
    tool      → 工具执行结果（tool_call_id 配对）  ┘
    "前面不能动、后面在变长"——这是后续讲 KV Cache 与上下文压缩的地基：
    前缀字节级稳定才能命中缓存，压缩只敢动轨迹。

与前面实验的关系：
    第 1 章说"上下文五组成"（系统提示词/工具定义/用户消息/模型回复/工具结果），
    今天从 API 视角重看（图 2-4）：前四者就是 system/user/assistant/tool 四种
    消息角色，工具定义是请求顶层的 tools 字段（不是消息）。两套分类，同一个请求。

作业落点（今日三问）：
    1. 四角色 + tools 一句话作用   → CONTEXT_MAP
    2. 静态 / 动态分界线           → static_prefix 与 trajectory 两段拼接
    3. 与第一章五组成一一对应      → CONTEXT_MAP 每行末尾"第一章"列

运行：uv run python -X utf8 chapter2/08_context_anatomy.py（需 .env，真实 API 两次小调用）
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL

# ---------- 作业 1+3：五个组成部分一览（API 分类 vs 第一章分类） ----------
#   (API 角色/字段, 所属区,    一句话作用,                                        第一章对应物)
CONTEXT_MAP = [
    ("system",    "静态前缀", "系统提示词：开发者写死的规则与角色，Agent 的'岗位说明书'", "系统提示词"),
    ("tools",     "静态前缀", "工具定义：请求顶层独立字段（不是消息），交代'手脚菜单'",   "工具定义"),
    ("user",      "动态轨迹", "用户消息：任务起点，也常夹带 RAG 检索来的外部知识",       "用户消息"),
    ("assistant", "动态轨迹", "模型回复：文字答案或工具调用请求（已采取的行动）",         "模型回复"),
    ("tool",      "动态轨迹", "工具执行结果：靠 tool_call_id 与 assistant 的调用配对",   "工具执行结果"),
]

model = ChatOpenAI(
    api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0,
)

# ---------- 作业 2：上下文 = 静态前缀（全程不变）+ 轨迹（只增不减） ----------

static_prefix = [
    {"role": "system", "content": "你是严谨的后端编程助手"},
]

trajectory = [
    {"role": "user", "content": "我叫小明，请记住这个名字"},
    {"role": "assistant", "content": "好的小明，我记住了"},
    {"role": "user", "content": "我叫什么名字？"},
]

# 无状态的本质：模型每步"推理"的输入 = 本次请求的全部 messages。
# 真实请求还有与 messages 平级的顶层字段（model、tools 等），Day 9 会用到。
full_messages = static_prefix + trajectory

if __name__ == "__main__":
    print("[作业 1+3] 上下文五组成：API 分类 ←→ 第一章分类")
    for role, zone, duty, ch1 in CONTEXT_MAP:
        print(f"  {role:<10}│ {zone} │ {duty} ← 第一章·{ch1}")

    print("\n[解剖] 本次请求发给模型的完整上下文（静态前缀 + 轨迹）：")
    print(json.dumps(full_messages, ensure_ascii=False, indent=2))

    # ---------- 对照实验：模型只看这次请求，不记得上一轮 ----------
    print("\n[对照 A] 模拟失忆：只发最后一句，不带任何历史")
    print(f"  回答：{model.invoke([trajectory[-1]]).content}")

    print("\n[对照 B] 全量重发：静态前缀 + 完整轨迹")
    print(f"  回答：{model.invoke(full_messages).content}")

    print("\n同一个问题两个答案——差的就是这次请求里带没带历史。")
    print("所谓'模型记得我'，是客户端每轮全量重发历史制造的错觉（JWT，不是 Cookie-Session）。")
