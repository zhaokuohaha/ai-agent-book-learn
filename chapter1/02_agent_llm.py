"""chapter1/02_agent_llm.py — 接入真实模型的 Agent（LangChain + 工具调用）

与 01_agent.py 的区别：llm() 从手写规则换成了真实 API，agent_loop 结构不变。
使用 LangChain 生态：统一接口，便于切换模型和工具。

运行：uv run chapter1/02_agent_llm.py
配置：编辑项目根目录 .env（所有章节共用，需 OpenAI 兼容接口）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 定位项目根目录

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL

# ---------- LLM 初始化（LangChain 统一接口） ----------

llm = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL)

# ---------- 工具定义（LangChain @tool 装饰器） ----------


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气"""
    return f"{city} 晴 26°C"  # 真实场景：调天气 API


tools = [get_weather]
llm_with_tools = llm.bind_tools(tools)  # 绑定工具到 LLM

# ---------- Agent 循环（手写 while 循环，展示原理） ----------


def agent_loop(user_input: str, max_turns: int = 10) -> str:
    messages = [{"role": "user", "content": user_input}]
    for _ in range(max_turns):  # 防止模型无限循环调工具
        resp = llm_with_tools.invoke(messages)  # 大脑思考：真实模型决策

        if not resp.tool_calls:  # 无动作 = 任务完成
            return resp.content

        messages.append(resp)  # 把"要调工具"这个决策写回上下文
        for call in resp.tool_calls:  # 手脚执行（LangChain tool_calls 是 dict）
            result = get_weather.invoke(call["args"])
            print(f"  [tool] get_weather({call['args']}) -> {result}")
            messages.append({  # 观察写回上下文
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result,
            })
    return "（达到最大轮数，任务未完成）"


if __name__ == "__main__":
    print(agent_loop("上海天气怎么样？"))
