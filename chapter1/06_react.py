"""chapter1/06_react.py — 第 1 章实验日：跑通你的第一个 while 循环 Agent（LangChain）

Day 5 核心概念（原书第 1 章）：

核心公式：
    上下文 = 静态前缀（系统提示词 + 工具定义） + 轨迹（消息历史，只增不改）
    每一轮：拼上下文 → 调模型 → 有工具调用就执行并把结果追加进轨迹 → 循环
    模型输出没有工具调用 → 最终答案，循环结束

ReAct 循环（Reason + Act）：
    模型先推理（Reason：要不要调工具？调哪个？），再行动（Act：执行工具），
    拿到观察结果（Observe：工具返回），写回上下文，下一轮。循环直到模型认为
    任务完成（不再调工具）。
    后端类比：轨迹就是一张 append-only 的请求日志表，每轮全量读出拼成 prompt；
    max_turns 就是超时熔断，防死循环打爆预算。

代码重点：
    - 静态前缀：system prompt + tools（LangChain @tool 装饰器）
    - 轨迹：messages 列表，只增不改（append-only）
    - 循环：for + max_turns 熔断（防死循环）
    - 工具执行：模型返回 tool_calls → 执行 → 结果追加进 messages
    - 终止条件：模型输出没有 tool_calls → 返回最终答案

使用 LangChain 生态：ChatOpenAI + @tool + bind_tools。

运行：uv run chapter1/06_react.py（需先填好 .env）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 定位项目根目录

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL

# ---------- LLM 初始化（LangChain 统一接口） ----------

llm = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL)

# ---------- 工具定义（LangChain @tool 装饰器，静态前缀的一部分） ----------


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气"""
    return f"{city} 晴 26°C"


# 工具注册表（用于按名称查找执行）
tools_registry = {"get_weather": get_weather}
tools_list = [get_weather]
llm_with_tools = llm.bind_tools(tools_list)  # 绑定工具到 LLM

# ---------- 静态前缀：系统提示词 ----------

SYSTEM_PROMPT = "你是一个天气查询助手。根据用户问题，调用 get_weather 工具查询天气，然后给出简洁回答。"

# ---------- ReAct 循环：while 循环 Agent ----------


def agent_loop(user_input: str, max_turns: int = 5, system_prompt: str = None) -> str:
    """ReAct 循环：拼上下文 → 调模型 → 有工具调用就执行并追加进轨迹 → 循环"""
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    # 轨迹：append-only 的消息历史（LangChain Message 对象）
    messages = [
        SystemMessage(content=system_prompt),  # 静态前缀
        HumanMessage(content=user_input),      # 用户输入
    ]

    for turn in range(1, max_turns + 1):
        print(f"\n[turn {turn}] 调用模型...")
        resp = llm_with_tools.invoke(messages)  # 全量读出轨迹拼成 prompt

        # 终止条件：模型输出没有 tool_calls → 最终答案
        if not resp.tool_calls:
            print(f"[turn {turn}] 无工具调用，返回最终答案")
            return resp.content

        # 有工具调用：执行并追加进轨迹（append-only）
        messages.append(resp)  # 把"要调工具"这个决策追加进轨迹
        for call in resp.tool_calls:  # LangChain tool_calls 是 dict
            tool_name = call["name"]
            tool_args = call["args"]
            result = tools_registry[tool_name].invoke(tool_args)
            print(f"  [tool] {tool_name}({tool_args}) -> {result}")
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    return "（达到最大轮数，熔断）"


if __name__ == "__main__":
    print("=" * 50)
    print("场景 1：正常流程（工具成功）")
    print("=" * 50)
    print(agent_loop("上海天气怎么样？"))

    print("\n" + "=" * 50)
    print("场景 2：断环实验（工具持续失败 + 强制重试 → max_turns 熔断）")
    print("=" * 50)

    # 替换工具为不稳定版本：总是返回错误
    @tool
    def get_weather(city: str) -> str:
        """不稳定的天气工具：总是返回错误"""
        return "错误：天气服务暂时不可用"

    tools_registry["get_weather"] = get_weather
    llm_with_tools = llm.bind_tools([get_weather])

    # 强制模型重试：system prompt 要求"必须重试直到成功"
    retry_prompt = "你是一个天气查询助手。如果工具返回错误，你必须重试，直到成功为止。不要放弃。"
    print(agent_loop("上海天气怎么样？", max_turns=3, system_prompt=retry_prompt))
