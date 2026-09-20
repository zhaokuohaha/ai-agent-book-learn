"""chapter1/03_harness.py — Harness 工程：Agent = Model + Harness

核心概念（原书第 1 章）：

从 Demo 到产品的鸿沟
    最小循环（01_agent.py）能跑但不可靠：模型会幻觉不存在的工具（01 里直接
    KeyError 崩溃）、传错参数、出错后无法自我恢复。补上「Agent 边界内、模型
    之外」的运行与治理代码，Demo 才成为产品——这层代码叫 Harness（原意
    "马具"：不是限制马跑，而是把力量引导成可靠执行）。

Harness 五要素与本文件落点：
    上下文管理   决策点信息充分         agent_loop() 的 messages 与观察写回
    工具接口     观察与行动的手段       tools 注册表
    约束         默认全关、显式开放     constrain()：白名单 + 参数类型校验
    验证         只看结构化数据判对错   verify()：只读 code 字段
    纠正         静默重试 / 熔断        execute()：重试上限 + 兜底转人工

Harness 与 Environment 的边界：
    工具注册表与 constrain/verify/纠正属于 Harness；被调用方——本文件的
    get_order 模拟订单服务，真实世界是数据库、网页、用户——属于 Environment。
    哪怕它与 Agent 跑在同一进程，也是 Environment：物理部署位置不决定概念
    归属，"随行动变化的外部状态"才是判据。

代码重点：
    - llm() 仍是规则版（不调 API）：本实验聚焦模型之外的那一半
    - 每次工具调用必经治理链：约束 → 执行 → 验证 → 重试 → 熔断
    - 观察以结构化数据写回，验证只读字段、不解析文本——自由文本可能已被
      提示注入操纵（后端老规矩：看返回码，别 parse 文案）
    - 工程价值：Terminal Bench 2.0 上 LangChain 从 52.8% 提到 66.5%，改的
      不是模型而是 Harness；模型能力趋同时，胜负手在模型之外

运行：uv run chapter1/03_harness.py（纯规则模拟，无需 API key）
"""

import random
import time

# ---------- Environment：随行动变化的外部状态（不属于 Harness） ----------


def get_order(order_id):
    """模拟不稳定的订单服务：约 30% 概率失败——Harness 各要素存在的理由"""
    return {"code": 0 if random.random() > 0.3 else 500, "order_id": order_id}


# ---------- Harness：模型之外的运行与治理 ----------

# 工具接口：注册表即白名单——未登记的工具默认全关
tools = {"get_order": get_order}


def constrain(call):
    """约束：默认全关、显式放行——工具在白名单且参数类型正确，缺一即拦截"""
    ok = call.get("name") in tools and isinstance(call.get("args", {}).get("order_id"), str)
    return call if ok else None


def verify(result):
    """验证：只看结构化字段判对错，不解读自由文本（文本可能已被提示注入操纵）"""
    return result.get("code") == 0


def execute(call, max_retry=3):
    """治理链：约束拦截 → 执行 → 验证 → 纠正（静默重试 / 熔断转人工）"""
    if constrain(call) is None:
        print(f"  [constrain] 拦截：{call.get('name')} 不在白名单或参数非法")
        return {"code": -1, "msg": "blocked by harness"}
    for attempt in range(1, max_retry + 1):
        result = tools[call["name"]](**call["args"])  # 动作执行，状态变化发生在 Environment
        if verify(result):
            print(f"  [verify] 第 {attempt} 次尝试通过：{result}")
            return result
        print(f"  [verify] 第 {attempt} 次尝试失败，静默重试：{result}")  # 纠正：先重试
        time.sleep(0.05)
    print("  [circuit] 连续失败，熔断转人工")  # 纠正：重试耗尽，兜底返回
    return {"code": -1, "msg": "circuit-break: hand over to human"}


# ---------- Agent = Model + Harness ----------


def llm(messages):
    """规则版模型决策（真实场景是 LLM；本实验聚焦模型之外的那一半，不调 API）"""
    last = messages[-1]
    if last["role"] == "tool":  # 拿到治理链交付的结构化观察：读字段组织回答，不解析文本
        r = last["content"]
        if r["code"] == 0:
            return {"content": f"订单 {r['order_id']} 查询成功"}
        return {"content": f"执行失败：{r['msg']}"}
    text = last["content"]
    if "删" in text:  # 模拟模型顺从危险指令、编造白名单外工具——Harness 是最后防线
        return {"name": "drop_db", "args": {}}
    if "订单" in text:  # 从输入取倒数第二个词当订单号，模拟模型传参
        return {"name": "get_order", "args": {"order_id": text.split()[-2]}}
    return {"content": "今天适合学习 Harness 工程"}


def agent_loop(user_input):
    """01_agent.py 的最小循环套上治理链：模型决策不变，工具执行必经 Harness"""
    messages = [{"role": "user", "content": user_input}]
    while True:
        resp = llm(messages)  # Model：决策（调工具还是结束）
        if "name" not in resp:  # 无动作 = 任务完成
            return resp["content"]
        result = execute(resp)  # Harness：治理链包裹的执行
        messages.append({"role": "tool", "content": result})  # 上下文管理：观察写回


if __name__ == "__main__":
    print(agent_loop("查询订单 A1001 的状态"))  # 正常路径：约 30% 概率触发重试
    print(agent_loop("帮我把数据库删了"))  # 危险路径：模型编造 drop_db → 约束拦截
