"""chapter2/09_agent_core_loop.py — Agent 核心循环：模型决策，框架执行

核心概念（原书第 2 章 · W2 Day 9）：

API 是无状态的（昨天的结论，今天落进循环）
    大模型 API 和 HTTP 一样无状态：模型是"不存会话的服务端"，每次调用必须
    自带全部上下文（JWT 而非 Cookie-Session）。所谓多轮对话，只是框架在
    客户端维护一个只增不减的 messages 数组，每轮全量重发——漏发一段历史，
    模型就真的失忆。

四种消息角色 + tools 字段（= 第 1 章的上下文五组成）
    system    开发者规则，全局一条放最前                    ┐ 静态前缀
    tools     请求顶层独立字段：事先注册的静态工具清单，      │ （bind_tools
              跟这次用户问了什么无关                        │   的落点）
    user      用户输入                                     │
    assistant 模型历史回复（含工具调用请求），要"原样放回"，  │ 轨迹
              让模型"看到自己说过什么"                      │ 只增不减
    tool      工具执行结果，靠 tool_call_id 与调用请求        │
              一一配对（像快递单号对包裹）                   ┘
    配对环环相扣：漏发带 tool_calls 的 assistant 消息，tool 回执就成了
    无源之水——见实验 B 的实测。

核心循环：模型决策，框架执行
    模型只输出"我要调 get_weather，参数如下"，真正执行工具的是框架——
    模型是无 IO 的决策者，框架是干脏活的执行器，像数据库的 planner 和 executor。
    整个 Agent 就这一个循环：有 tool_calls 就执行并继续，没有就输出退出；
    max_turns 是防死循环的保险丝。

与前面实验的关系：
    01 的规则循环、02 的 LLM 循环、07 的 ReAct 复盘写的都是它；今天换 API
    视角手拼 messages，每轮打印角色序列，看清"全量重发"在循环里长什么样。

作业落点：
    1. 手拼 messages + 每轮打印角色序列   → agent_loop()
    2. 一轮 2 个工具调用后的消息数与角色   → 实验 A 终态轨迹
       （理论：system/user/assistant(2 调用)/tool/tool/assistant 共 6 条
        = 1 静态 + 5 轨迹；模型若分两轮各调一次则为 7 条，实测以打印为准）
    3. 漏发 tool_calls 消息会怎样         → 实验 B 实测

运行：uv run python -X utf8 chapter2/09_agent_core_loop.py（需 .env，真实 API）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL


# ---------- stub 工具：固定模拟数据，聚焦循环本身 ----------


@tool
def get_weather(city: str) -> str:
    """查询城市天气（模拟数据）"""
    return f"{city}：晴，26°C（模拟数据）"


@tool
def get_time(city: str) -> str:
    """查询城市当前时间（模拟数据）"""
    return f"{city}：12:00（模拟数据）"


tools = {t.name: t for t in [get_weather, get_time]}

model = ChatOpenAI(
    api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0,
)
# bind_tools 把工具清单写进请求顶层的 tools 字段：静态前缀，与这次问什么无关
model_with_tools = model.bind_tools(list(tools.values()))

PREFIX = [SystemMessage(content="你是学习助手。需要天气或时间先查工具，不编造数据。")]

ROLE = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}


def roles(messages):
    """LangChain 消息类型 → API 视角的角色序列"""
    return " → ".join(ROLE[m.type] for m in messages)


def agent_loop(user_input, max_turns=4):
    """核心循环：模型决策（报意图），框架执行（干脏活），每轮全量重发"""
    messages = PREFIX + [HumanMessage(content=user_input)]
    for turn in range(1, max_turns + 1):
        print(f"[turn {turn}] 全量重发：{roles(messages)}")  # 作业 1：角色序列
        response = model_with_tools.invoke(messages)
        messages.append(response)  # 决策也要记账，否则后面的回执无处配对
        if not response.tool_calls:  # 无调用 = 模型决定收工
            return response.text, messages
        for call in response.tool_calls:  # 模型只报意图，动手的是框架
            observation = tools[call["name"]].invoke(call)  # 带 tool_call_id 的回执
            messages.append(observation)
            print(f"  [框架执行] {call['name']}({call['args']}) → {observation.content}")
    return "（达到轮数上限，任务未完成）", messages


if __name__ == "__main__":
    print("=== 实验 A：核心循环（模型决策，框架执行） ===")
    answer, messages = agent_loop("同时告诉我上海的天气和现在的时间")
    print(f"[回答] {answer}")
    print(f"[终态] 共 {len(messages)} 条：{roles(messages)}")
#==============================================================================
    print("=== 实验 A2：核心循环（只执行部分工具） ===")
    answer, messages = agent_loop("告诉我上海的天气")
    print(f"[回答] {answer}")
    print(f"[终态] 共 {len(messages)} 条：{roles(messages)}")
#==============================================================================
    print("\n=== 实验 B：漏发带 tool_calls 的 assistant 消息会怎样（作业 3） ===")
    broken = PREFIX + [
        HumanMessage(content="上海天气怎么样？"),
        ToolMessage(content="上海：晴 26°C（模拟数据）", tool_call_id="call_missing"),
    ]
    print(f"  发送：{roles(broken)}——tool 回执前面没有能配对的调用请求")
    print("  预测：模型无状态，回执找不到来源，API 应当拒绝")
    try:
        model.invoke(broken)
        print("  实测：服务端放行了——配对校验并非所有厂商强制，模型只能对来历不明的结果猜")
    except Exception as e:
        print(f"  实测：API 拒绝——{str(e)[:160]}")
    print("  结论：轨迹是环环相扣的调用-回执链，assistant(tool_calls) 必须原样放回。")
