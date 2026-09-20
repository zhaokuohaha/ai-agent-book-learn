"""Day 7 · W1 一页知识图谱：三件套 → Harness → 设计模式。


          Agent = Model + Harness
                   │
   ┌───────────┬───┴────────┬───────────┐
   ▼           ▼            ▼           ▼
 大脑LLM     眼睛上下文    手脚工具     环境
 只决策     前缀 + 轨迹   观察/动作    状态+反馈
   └─────  ReAct: 想→做→看  ─────┘
                   │
   五要素: 上下文·工具·约束·验证·纠正
                   │
   五模式: 提议-审核 / 渐进披露 / 只增不改
           边界+保留 / 最小diff可回滚


主线图（对照 01/02 的骨架、03 的治理、04/05 的双角色、06 的循环）：
    Agent = LLM（大脑：决定做什么）+ 上下文（眼睛：看到什么）+ 工具（手脚：能做什么）
      └─ 换个工程视角：Agent = Model + Harness，不是两套互相替代的公式
           ├─ 上下文：备齐资料    工具：提供接口
           └─ 约束：门卫放行      验证：质检验收      纠正：返工或停机
    Agent ←→ Environment：工具接口属于 Harness，背后的数据库、网页等属于环境。
    模型不变时，按需扩展眼睛和手脚往往比反复改提示词更有效，但权限也要跟上。

上下文像办公桌：固定手册 + 不断追加的工作流水账。
    静态前缀 = 系统提示词 + 工具定义；轨迹 = 用户消息 + 模型回复 + 工具结果。
    本例 PREFIX 保存系统消息，bind_tools 提供工具定义，trajectory 只追加消息。
    前缀稳定有利于 KV Cache 复用，但是否命中仍取决于服务端的缓存机制。
    模型调用不会自动记住上轮消息，需要重新带上历史；这不等于输出是确定的纯函数。
    工具既能读取观察，也能改变世界；本例只返回模拟天气，不接真实天气服务。

闭环口诀：想 → 做 → 看 → 再想；“手脚做了”不等于“大脑知道了”。
    HumanMessage → AIMessage(tool_calls) → ToolMessage → AIMessage(最终回答)
    没有工具调用表示模型决定结束，不保证业务成功；验收还要核对工具轨迹与结果。
    max_turns 是轮数保险丝，不是超时器，也不限制单轮工具次数或总 token 数。

消融复盘：拿走一块零件，再看哪里失灵（结果要实测，不背固定输出）。
    去掉工具定义：在本例中没有工具调用入口，但模型仍可能拒答或编造答案。
    屏蔽结果反馈：模型可能重复尝试、放弃或编造；关键是失去了真实观察。
    06 的“服务报错后重试”仍有反馈，不等于反馈缺失。若实验要屏蔽结果内容，
    应保留 ToolMessage 与调用 ID 的配对；直接删掉回执可能先触发 API 协议错误。

五大模式，记住后端熟语就能串起来：
    提议者-审核者：写代码和审 PR 分工；审核者看产物，不共享提议者的推理上下文。
    渐进式披露：先看目录再翻正文，像分页和懒加载，省上下文也减少选错信息。
    只增不改：流水账只追加，像 event sourcing；便于回放，但不自动等于防篡改。
    边界集 + 保留集：既测该变好的样本，也测不能变坏的样本，像回归测试。
    最小 diff + 可回滚：小步修改、记来源、能撤回，出错才知道是哪一步引起的。

代码重点：约 30 行核心逻辑只复现 ReAct，不硬塞五种模式或完整生产治理。
    @tool 从类型与 docstring 生成参数说明；bind_tools 只给模型“工具菜单”。
    invoke(call) 传入完整调用字典，会得到带 tool_call_id 的 ToolMessage 回执。
    最终回答也追加进轨迹；函数返回回答与轨迹，验收不靠猜模型的固定文案。

自测（先预测，再运行）：
    1. 一次工具调用后回答：轨迹应有 4 条消息，另有 1 条静态 system 消息。
    2. max_turns=1 且第一轮调用工具：结果虽已写回，但没有下一轮供模型读取。
    3. 不查天气、只打招呼：允许直接回答；没有工具调用本身不是错误。
    用自己的话补一句：我以前把 Agent 当作____，现在发现可靠执行还需要____。

来源：https://bojieli.github.io/ai-agent-book/book/chapter1/
API：https://docs.langchain.com/oss/python/langchain/models#tool-calling
运行：uv run python -X utf8 chapter1/07_w1_review.py（真实模型 + 模拟天气，需 .env）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL


@tool
def get_weather(city: str) -> str:
    """查询城市的模拟天气，仅供学习，不代表实时天气。"""
    return f"{city}：晴，26°C（模拟数据）"


tools = {get_weather.name: get_weather}
model = ChatOpenAI(
    api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0,
)
model_with_tools = model.bind_tools(list(tools.values()))  # 给菜单，不替模型执行
PREFIX = [SystemMessage(content="你是学习助手。天气问题先查工具，不编造数据；其他问题简洁回答。")]


def agent_loop(user_input, max_turns=3):
    trajectory = [HumanMessage(content=user_input)]
    turn = 0
    while turn < max_turns:
        turn += 1
        print(f"[turn {turn}] 静态 system=1，轨迹={len(trajectory)} 条")
        response = model_with_tools.invoke(PREFIX + trajectory)
        trajectory.append(response)  # 最终回答也要记账，不能先 return
        if response.invalid_tool_calls:
            raise ValueError("工具调用参数无法解析，不能当作正常结束")
        if not response.tool_calls:
            return response.text, trajectory
        for call in response.tool_calls:
            observation = tools[call["name"]].invoke(call)  # 完整调用 → 带 ID 的回执
            trajectory.append(observation)  # 只执行不回传，就像干完活却没交报告
            print(f"  [tool] {call['name']}({call['args']}) → {observation.content}")
    return "达到模型调用轮数上限，任务未完成。", trajectory


if __name__ == "__main__":
    answer, trajectory = agent_loop("请查询上海的模拟天气，并提醒我是否需要防晒。")
    print(f"[回答] {answer}")
    print("[轨迹] " + " → ".join(message.type for message in trajectory))
