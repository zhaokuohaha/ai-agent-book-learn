"""chapter3/16.2_agent_memory.py — MemoryStore × 玩具 Agent：第二次会话"记得你"

核心概念（16 的选做作业，承接 16.1 与 common/memory_store.py）：

记忆注入闭环
    16.1 验证了"存取层"跨会话生效；本实验把 D02 的玩具 Agent（手写 while 循环 +
    bind_tools）接上同一个 MemoryStore，验证完整闭环：会话结束 → 记忆入库 →
    新会话 system 注入 store.render() → Agent 不问自答。

对照实验（无记忆 vs 有记忆）
    同一句"天气怎么样？"跑两遍：无记忆组不知道用户在哪（反问或瞎猜）；有记忆组
    从档案读到 city=上海，直接 get_weather(上海)。验收看工具调用轨迹不看文案
    （模型输出动态，D03 起的老规矩）。

幕 A 入库是手动一行 store.remember——真实系统由 LLM 提取（15 的 ETL），本实验
    聚焦"接上之后闭环是否成立"，不重复提取部分。

运行：uv run python -X utf8 chapter3/16.2_agent_memory.py（需 .env，约 3 次真实调用）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL
from common.memory_store import MemoryStore

STORE_PATH = Path(__file__).parent / "agent_memory.json"  # 与 16.1 的主实验记忆文件分开
TS_A, TS_B = "2026-09-27", "2026-09-28"  # 两幕剧的会话日期


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气"""
    return f"{city} 晴 26°C"  # 真实场景：调天气 API（同 D02 的玩具工具）


def agent_loop(model, store, user_input, ts, max_turns=10):
    """D02 的手写 while 循环原样复刻，只加一处：system 注入 store.render()
    记忆档案（记忆是跨会话档案，system 是它在消息序列里的座位）。
    返回 (最终答复, 工具调用参数列表)——轨迹用于验收，不看文案看动作"""
    messages = [
        SystemMessage(content=f"你是私人助理。今天是 {ts}。\n\n{store.render()}"),
        HumanMessage(content=user_input),
    ]
    llm_with_tools = model.bind_tools([get_weather])
    calls = []
    for _ in range(max_turns):
        resp = llm_with_tools.invoke(messages)
        if not resp.tool_calls:  # 无动作 = 任务完成
            return resp.content, calls
        messages.append(resp)
        for call in resp.tool_calls:
            result = get_weather.invoke(call["args"])
            print(f"  [tool] get_weather({call['args']}) -> {result}")
            calls.append(call["args"])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    return "（达到最大轮数，任务未完成）", calls


def agent_demo(model):
    """两幕剧验证"第二次会话记得你"：幕 A 首次会话学到"用户在上海"，幕 B
    重启后只说"天气怎么样？"——Agent 该靠记忆直接查上海；另设无记忆对照组"""
    STORE_PATH.unlink(missing_ok=True)  # 每次从头演示两幕

    print(f"\n--- 幕 A（{TS_A}，首次会话）---")
    store_a = MemoryStore(STORE_PATH)  # 空库
    print("用户：我在上海，以后查天气别再问我城市。上海现在天气怎么样？")
    answer, _ = agent_loop(model, store_a, "我在上海，以后查天气别再问我城市。上海现在天气怎么样？", TS_A)
    print(f"助手：{answer[:80]}")
    # 会话结束 → ETL 入库：真实系统由 LLM 提取（见 15），此处手动入库聚焦存取闭环
    store_a.remember("city", "上海", "semantic", TS_A)
    print(f"[会话结束 → 入库] city = 上海（semantic，{TS_A}）→ save")

    print(f"\n--- 幕 B（{TS_B}，重启后的第二次会话）---")
    print("对照组（无记忆，新用户视角）——用户：天气怎么样？")
    store_none = MemoryStore(STORE_PATH.with_name("empty.json"))  # 空文件：对照组的"失忆"
    answer_none, calls_none = agent_loop(model, store_none, "天气怎么样？", TS_B)
    print(f"助手：{answer_none[:80]}")
    print("实验组（load 记忆档案）——用户：天气怎么样？")
    store_b = MemoryStore(STORE_PATH)  # 新进程视角：load 恢复
    answer, calls_mem = agent_loop(model, store_b, "天气怎么样？", TS_B)
    print(f"助手：{answer[:80]}")

    queried = [c.get("city") for c in calls_mem]
    ok = "上海" in queried
    print(f"\n[验收] 对照组查询城市：{[c.get('city') for c in calls_none] or '未调工具'}（不知道用户在哪）")
    print(f"[验收] 实验组查询城市：{queried} —— "
          + ("记忆跨会话生效，第二次会话'记得你' ✓" if ok else "未查上海（模型输出动态，可重跑）"))


if __name__ == "__main__":
    model = ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0,
    )
    print("=== 选做：D02 玩具 Agent × MemoryStore——第二次会话\"记得你\" ===")
    agent_demo(model)

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑：
#   1. 幕 A（TS_A，空库）：用户自报上海并问天气 → D02 手写循环调 get_weather →
#      会话结束手动入库 city=上海（ETL 的简化）→ save 落盘
#   2. 幕 B（TS_B）：对照组用空文件 store（无记忆）问"天气怎么样？"；
#      实验组 load 记忆档案后问同一句——验收看两组的工具调用轨迹差异
# 验证内容：
#   - 幕 A 工具轨迹 city=上海（用户明说了城市，理应查对）
#   - 幕 B 对照组反问或瞎猜城市（或未调工具）；实验组直接 get_weather(上海)——
#     记忆跨会话生效（模型输出动态，验收看轨迹不断言文案）
