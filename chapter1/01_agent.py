"""最简单的 Agent：大脑（LLM 决策）+ 手脚（工具执行）+ 循环（观察写回上下文）"""

CITIES = ("北京", "上海", "广州", "深圳")

def llm(messages: list) -> dict:
    # 真实场景：调 OpenAI 兼容 API，由模型决定是否调用工具
    last = messages[-1]
    if last["role"] == "tool":
        # 拿到工具返回的观察结果，组织最终回答
        return {"content": f"查询结果：{last['content']}"}
    if "天气" in last["content"]:
        for city in CITIES:
            if city in last["content"]:
                return {"tool": "get_weather", "args": {"city": city}}
    return {"content": "今天适合学习 AI Agent"}

def get_weather(city: str) -> str:
    return f"{city} 晴 26°C"  # 真实场景：调天气 API

tools = {"get_weather": get_weather}

def agent_loop(user_input: str) -> str:
    messages = [{"role": "user", "content": user_input}]
    while True:
        resp = llm(messages)                          # 大脑思考
        if "tool" not in resp:                        # 无动作 = 任务完成
            return resp["content"]
        result = tools[resp["tool"]](**resp["args"])  # 手脚执行
        messages.append({"role": "tool", "content": result})  # 观察写回上下文

if __name__ == "__main__":
    print(agent_loop("上海天气怎么样？"))  # 走工具调用路径
    print(agent_loop("今天干什么好？"))    # 不需要工具，直接回答
