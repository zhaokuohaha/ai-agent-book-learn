"""chapter7/43.2_mini_tau2_loop.py — 迷你 τ² 环：双侧 LLM 对话 + 四维验证器

本实验新增的主概念是 **评估的执行与打分**：43.1 解剖了任务的静态结构，这里
让它动起来——Agent（真实 LLM + 运营商侧工具）↔ 用户模拟器（另一个 LLM 实例
+ 设备侧工具，双控），跑完整对话循环后用四维验证器打分。

与真实 τ²-bench 的对齐：用户模拟器同样由 LLM 扮演、同样带独立工具集、同样
受 known_info（渐进式透露）与 grounding（事实锚定）双重约束；Agent 政策里
同样写着「一次只发一个工具调用」——能不能守住，验证器数给你看。

打分设计（作业①的落地）：**reward 与报告分离**——
- reward：仍按 reward_basis=ENV_ASSERTION 给二元 0/1（口径不动，跨模型可比）；
- 报告：四维逐项 ✓/✗ + 政策违规计数（诊断信息全保留，指出问题出在哪）。
二元奖励的代价（过程违规不被扣分）不是靠把 reward 改复杂来解决，而是靠
把诊断放进报告层——数字负责排名，报告负责归因。

运行：uv run python -X utf8 chapter7/43.2_mini_tau2_loop.py（需 .env，双 LLM）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL

AGENT_POLICY = "政策：一次只发一个工具调用，等结果返回再决定下一步。"

AGENT_SYS = f"""你是电信客服 Agent，处理工单：{AGENT_POLICY}
运营商侧工具只能查线路；用户的手机（飞行模式/漫游/信号）你看不到也改不了，只能引导用户自己操作。
排障指南：出境上不了网，优先怀疑设备侧（飞行模式误开、数据漫游未开），请用户查看状态栏确认。
把根因明确告知用户，并让用户测速；用户确认解决即可收尾。{AGENT_POLICY}"""

USER_SYS = """你扮演一位真实用户：在法国出差的中国游客，手机突然上不了网，找客服。
你只知道：姓名张三、号码 +33612345678、人在法国。故障原因你不知道——别假装知道，也别主动猜。
铁律（grounding）：凡是涉及手机状态（飞行模式/漫游/信号）或测速结果，必须先调用你的设备工具、照实转述结果，绝不编造；客服没问到的信息不主动说。
验收口径：run_speed_test 返回 excellent 才算解决，你才可以确认「修好了」并结束对话。
像真实用户一样用一句口语回复，不要提及工具或扮演设定。"""

MAX_TURNS = 14


def make_env():
    """环境状态：carrier 归 Agent 侧，device 归用户侧——双控的数据基础"""
    env = {"device": {"airplane": True, "roaming": False, "last_speed": None},   # initial_state：故障注入
           "carrier": {"line_ok": True, "roaming_plan": True}}
    return env


def make_agent_tools(env):
    @tool
    def get_line_status(phone: str) -> str:
        """（运营商侧）查询号码的线路状态与国际漫游套餐"""
        c = env["carrier"]
        return f"线路正常={c['line_ok']} 国际漫游套餐已开通={c['roaming_plan']}（运营商侧无异常）"

    @tool
    def get_troubleshooting_guide(symptom: str) -> str:
        """（运营商侧）按症状查排障指南"""
        return "出境无法上网：线路侧正常时，根因通常在用户设备侧（飞行模式误开/数据漫游未开）。请引导用户查看状态栏并逐项确认。"

    return [get_line_status, get_troubleshooting_guide]


def make_user_tools(env):
    @tool
    def check_status_bar() -> str:
        """（设备侧）查看手机状态栏：飞行模式/漫游开关/信号"""
        d = env["device"]
        sig = "无" if d["airplane"] else ("有" if d["roaming"] else "弱")
        return f"飞行模式={'开' if d['airplane'] else '关'} 数据漫游={'开' if d['roaming'] else '关'} 信号={sig}"

    @tool
    def toggle_airplane_mode() -> str:
        """（设备侧）切换飞行模式开/关"""
        env["device"]["airplane"] = not env["device"]["airplane"]
        return f"飞行模式已{'打开' if env['device']['airplane'] else '关闭'}"

    @tool
    def toggle_roaming() -> str:
        """（设备侧）切换数据漫游开/关"""
        env["device"]["roaming"] = not env["device"]["roaming"]
        return f"数据漫游已{'打开' if env['device']['roaming'] else '关闭'}"

    @tool
    def run_speed_test() -> str:
        """（设备侧）测速——验收口径的唯一裁判"""
        d = env["device"]
        d["last_speed"] = "excellent" if (not d["airplane"] and d["roaming"]) else "poor"
        return f"测速结果: {d['last_speed']}"

    return [check_status_bar, toggle_airplane_mode, toggle_roaming, run_speed_test]


def act(llm, msgs, tools, trace, who):
    """一侧的行动：有 tool_calls 就执行并续轮（≤2 次），否则文本发言返回"""
    by_name = {t.name: t for t in tools}
    r = llm.bind_tools(tools).invoke(msgs)
    msgs.append(r)
    while r.tool_calls:
        if who == "agent":
            trace["tool_calls_per_turn"].append(len(r.tool_calls))   # 政策探测点：>1 即违规
            for tc in r.tool_calls:
                trace["agent_tools"].append(tc["name"])
        else:
            for tc in r.tool_calls:
                trace["user_tools"].append(tc["name"])
        for tc in r.tool_calls:  # 逐个配对（37.2 教训：并行 tool_call 全要补 ToolMessage）
            obs = by_name[tc["name"]].invoke(tc["args"])
            msgs.append(ToolMessage(content=obs, tool_call_id=tc["id"]))
            print(f"    [{who} 工具] {tc['name']}({tc['args']}) → {obs}")
        r = llm.bind_tools(tools).invoke(msgs)
        msgs.append(r)
    text = r.content if isinstance(r.content, str) else str(r.content)
    trace["dialog"].append(f"{who}: {text}")
    print(f"  {who}: {text}")
    return text


def verify(env, trace):
    """四维校验 + reward/报告分离：数字负责排名（口径不动），报告负责归因"""
    d = env["device"]
    env_ok = [("airplane_mode == False", not d["airplane"]),
              ("roaming == True", d["roaming"]),
              ("last_speed == 'excellent'", d["last_speed"] == "excellent")]
    violations = sum(1 for n in trace["tool_calls_per_turn"] if n > 1)
    told = any("飞行模式" in m and "漫游" in m for m in trace["dialog"] if m.startswith("agent"))
    report = {
        "env_assertions": env_ok,
        "actions": [("check_status_bar 发生过", "check_status_bar" in trace["user_tools"])],
        "communicate_info": [("根因已告知（Agent 对话含'飞行模式'与'漫游'）", told)],
        "nl_assertions": [("grounding：用户状态回答出自工具（user_tools 非空）", len(trace["user_tools"]) > 0)],
        "policy": [(f"一次一个工具调用（违规 {violations} 次，不计入 reward）", violations == 0)],
    }
    reward = 1.0 if all(ok for _, ok in report["env_assertions"]) else 0.0  # reward_basis=ENV_ASSERTION
    return reward, report


if __name__ == "__main__":
    llm = lambda: ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=90, max_retries=2)
    agent_llm, user_llm = llm(), llm()
    env = make_env()
    agent_tools, user_tools = make_agent_tools(env), make_user_tools(env)
    trace = {"dialog": [], "agent_tools": [], "user_tools": [], "tool_calls_per_turn": []}
    a_msgs, u_msgs = [SystemMessage(content=AGENT_SYS)], [SystemMessage(content=USER_SYS)]

    msg = "（工单开始）用户来电，请开场。"
    done = False
    for turn in range(1, MAX_TURNS + 1):
        print(f"\n── 轮 {turn} " + "─" * 40)
        a_msgs.append(HumanMessage(content=msg))
        msg = act(agent_llm, a_msgs, agent_tools, trace, "agent")
        if "excellent" in msg or env["device"]["last_speed"] == "excellent":
            done = True  # 测速已达验收口径——收尾（继续跑只是重复确认）
        u_msgs.append(HumanMessage(content=f"客服说：{msg}"))
        reply = act(user_llm, u_msgs, user_tools, trace, "user")
        if done or ("解决" in reply or "好了" in reply) and env["device"]["last_speed"] == "excellent":
            break
        msg = reply

    reward, report = verify(env, trace)
    print("\n" + "═" * 62 + "\n四维校验报告（诊断层——指出问题出在哪）\n" + "═" * 62)
    for dim, items in report.items():
        for name, ok in items:
            print(f"  [{dim}] {'✓' if ok else '✗'} {name}")
    print(f"\nreward = {reward}（口径 reward_basis=ENV_ASSERTION，二元 0/1）")
    print("对照：若政策违规 >0 而 reward=1.0——这就是二元奖励的固有代价：")
    print("  过程违规不被扣分（以过程颗粒度换跨模型可比的单一数字）。")
    print("  本实验的解法：reward 与报告分离——数字排名，报告归因，口径不动。")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（真实双 LLM，轨迹动态但结构确定）：
#   1. make_env() 即 initial_state：飞行模式=开、漫游=关（同一起点）
#   2. 循环 ≤14 轮：Agent（运营商工具+政策约束）发言/调工具 → 文本传给
#      用户模拟器（设备工具+known_info+grounding）→ 回复传回（双控对话）
#   3. act() 记录 tool_calls_per_turn——政策「一次一个工具调用」的探测器
#   4. verify() 四维校验 → reward（二元）+ 报告（诊断）分离输出
# 验证内容（看轨迹不看文案）：
#   - 期望轨迹：get_line_status 全正常 → get_troubleshooting_guide →
#     引导用户 check_status_bar（用户侧发起工具调用——第 30 条消息同款）
#     → toggle_airplane_mode + toggle_roaming → run_speed_test=excellent
#   - grounding 实证：用户回复中的状态数字必须与工具返回一致（不编造）
#   - 政策探测：违规计数 = 并行发多个 tool_call 的轮数。本次实跑 0 违规
#     （满分轨迹，reward=1.0）；若某轮并行则 >0 而 reward 仍 =1.0——二元
#     奖励代价的现场证据（课件满分轨迹同款现象：过程违规不扣分）
#   - 简化注：nl_assertions 用关键词/非空近似（生产用 LLM 裁判）；对话
#     截断条件用 excellent/「解决」关键词（生产按环境终态+最大轮数）
