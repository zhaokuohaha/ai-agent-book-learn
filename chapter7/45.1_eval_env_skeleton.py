"""chapter7/45.1_eval_env_skeleton.py — 评估环境:可重复运行的实验装置(五要素)

本实验新增的主概念是 **评估环境 = 可重复的装置**:给定同一初始状态,同一
Agent 应得到可比较的结果。五要素缺一不可——
  ①数据集(1 条记录=1 用例) ②环境状态(可重置的业务/设备状态)
  ③工具接口(原子操作) ④评分标准 Rubric(机器可复核) ⑤执行协议(###STOP###
  或轮数上限)。

后端类比:跑集成测试前先重建数据库 fixture——没有重置,用例互相污染,跑十次
十种结果,评估失去可比性(演示 B 现场翻给你看)。工具必须是**原子操作**:若
提供「一键解决所有待办」这种高层抽象,规划与推理就被工具本身吸收了,评估退
化成考单次函数调用(演示 C 对照)。

四类工具调用型环境按「是否跨轮保持状态 + 是否需要隔离」分层(文末阶梯表),
本实验落在 StatefulToolEnv 这一层;人机交互型(需用户模拟器)见 43.2。验证器
深度(双重验证)在 45.2。数字/结构概念——脚本化 Agent,聚焦「装置」而非「智能」。

运行:uv run python -X utf8 chapter7/45.1_eval_env_skeleton.py(零 API,确定性)
"""


# ---- 要素②环境状态 + 要素③工具接口(原子操作)----
def make_env():
    return {"todos": [], "done": [], "score": 0}


def make_tools(env: dict):
    def add_todo(item: str) -> str:
        env["todos"].append(item)
        return f"已添加待办 {item}"

    def complete_todo(item: str) -> str:  # 原子步:一次只完成一个
        if item not in env["todos"]:
            return f"错误:待办 {item!r} 不存在"  # 清晰错误信息而非单一失败标志——Agent 才能据此调整
        env["todos"].remove(item)
        env["done"].append(item)
        env["score"] += 10
        return f"已完成 {item}(+10 分,当前 {env['score']})"

    def resolve_all() -> str:  # 反例:高层抽象——把「规划哪些要做」吸进工具
        for it in env["todos"]:
            env["done"].append(it)
            env["score"] += 10
        n = len(env["todos"])
        env["todos"] = []
        return f"已一键解决全部 {n} 个待办"

    def list_state() -> str:
        return f"待办={env['todos']} 已完成={env['done']} 分={env['score']}"

    return {"add_todo": add_todo, "complete_todo": complete_todo,
            "resolve_all": resolve_all, "list_state": list_state}


class EvalCase:
    """要素①数据集:一条记录 = 一个用例,五要素的载体"""

    def __init__(self, case_id, task, init_actions, rubric):
        self.case_id, self.task, self.init_actions, self.rubric = case_id, task, init_actions, rubric
        self.env = make_env()
        self.tools = make_tools(self.env)

    def setup(self):
        """要素②的可重置:initialization_actions 把状态拉回同一起点(fixture 的 setUp)"""
        self.env = make_env()
        self.tools = make_tools(self.env)
        for name, arg in self.init_actions:
            self.tools[name](arg)

    def verify(self) -> bool:
        """要素④ Rubric:机器可独立复核,不信 Agent 自我报告(深度验证见 45.2)"""
        return self.rubric(self.env)


def run_agent(case: EvalCase, actions, max_turns=8):
    """脚本化 Agent 跑一条用例;要素⑤执行协议:动作耗尽输出 ###STOP###,或轮数上限强制终止"""
    for turn, (name, arg) in enumerate(actions[:max_turns], 1):
        obs = case.tools[name](arg) if arg is not None else case.tools[name]()  # 无参工具(list_state/resolve_all)不传 arg
        print(f"    轮{turn}: {name}({'' if arg is None else repr(arg)}) → {obs}")
    print(f"    轮{len(actions[:max_turns]) + 1}: ###STOP###(Agent 主动终止;或达 max_turns={max_turns} 上限)")


if __name__ == "__main__":
    # 数据集两条用例:任务都是「完成全部待办」,Rubric 精确断言终态(不看 Agent 说什么)
    cases = [
        EvalCase("c1", "完成待办 A、B", [("add_todo", "A"), ("add_todo", "B")],
                 lambda e: e["todos"] == [] and set(e["done"]) == {"A", "B"} and e["score"] == 20),
        EvalCase("c2", "完成待办 X", [("add_todo", "X")],
                 lambda e: e["todos"] == [] and e["done"] == ["X"] and e["score"] == 10),
    ]
    atomic = {"c1": [("complete_todo", "A"), ("complete_todo", "B")],
              "c2": [("complete_todo", "X")]}

    print("=== 演示 A · 可重置:同一数据集连跑两次,结果必须一致(可比性的前提)===")
    for run in (1, 2):
        for c in cases:
            c.setup()                       # 每用例先重置——回到同一起点
            run_agent(c, atomic[c.case_id])
        print(f"  第 {run} 次:c1.verify={cases[0].verify()} c2.verify={cases[1].verify()}")

    print("\n=== 演示 B · 不重置:用例互相污染,评估失去可比性 ===")
    pollute = EvalCase("p", "往清单加一个待办", [], lambda e: len(e["todos"]) == 1)
    for run in (1, 2):
        if run == 1:
            pollute.setup()                 # 只有第一次重置
        run_agent(pollute, [("add_todo", "任务")])
        print(f"  第 {run} 次(不重置):todos={pollute.env['todos']} → verify(len==1)={pollute.verify()}")
    print("  → 第二次 len=2,同一用例同一动作却不同分——污染让「跑十次十种结果」")

    print("\n=== 演示 C · 原子操作 vs 高层抽象:规划被谁吸收 ===")
    c_atom = EvalCase("atom", "完成 A、B", [("add_todo", "A"), ("add_todo", "B")],
                      lambda e: set(e["done"]) == {"A", "B"})
    c_atom.setup()
    print("  原子工具(考规划):Agent 需 list→逐个 complete 两步")
    run_agent(c_atom, [("list_state", None), ("complete_todo", "A"), ("complete_todo", "B")])
    c_high = EvalCase("high", "完成 A、B", [("add_todo", "A"), ("add_todo", "B")],
                      lambda e: set(e["done"]) == {"A", "B"})
    c_high.setup()
    print("  高层工具(退化):resolve_all 一步搞定——规划/推理被工具吸收,测不出 Agent 能力")
    run_agent(c_high, [("resolve_all", None)])
    print(f"  两者 Rubric 都通过(atom={c_atom.verify()} high={c_high.verify()}),但只有原子版轨迹体现了规划")

    print("\n=== 演示 D · 工具调用型环境四层阶梯(按「跨轮状态 + 隔离」分)===")
    ladder = [
        ("SingleTurnEnv", "单轮数学题", "无状态", "纯函数单测"),
        ("ToolEnv", "多轮搜索综合", "跨轮无副作用", "多次只读调用"),
        ("StatefulToolEnv", "改数据库后验证状态", "有状态", "有副作用的集成测试 ← 本实验"),
        ("SandboxEnv", "沙箱执行代码", "有状态 + 隔离", "独立容器"),
    ]
    for name, eg, state, analogy in ladder:
        print(f"  {name:<16} {eg:<12} {state:<10} 类比:{analogy}")
    print("  人机交互型(如 telecom 客服)另有用户模拟器——考察沟通策略,见 43.2")

# ---- 执行逻辑与验证(看完代码再回头看这里)----
# 执行逻辑(零 API 确定性,脚本化 Agent):
#   1. EvalCase 载五要素;setup() 用 init_actions 重置(要素②的可重置)
#   2. 演示A:两用例各 setup 后跑,连跑两次 verify 都 True(可比重复)
#   3. 演示B:同一 add_todo 用例,只第一次 setup → 第二次 todos 累加 len=2,
#      verify(len==1) 变 False(污染)
#   4. 演示C:原子 complete_todo×2(带 list 观察)vs 高层 resolve_all×1,Rubric
#      都过但轨迹信息量不同;演示D 打印四层阶梯
# 验证内容(状态确定性,可手算复核):
#   - 演示A 两次输出完全一致(c1/c2 verify 均 True)——可重置=可比性
#   - 演示B 第二次 verify=False 且 todos=['任务','任务']——不重置则污染
#   - 演示C:complete_todo 逐个 +10(原子),resolve_all 一次 +10×n(高层);
#     两者 done 都含 A、B,但原子版多了 list_state 观察步(规划的证据)
#   - 简化注:脚本化 Agent 用固定动作序列(真 Agent 由 LLM 规划);环境是内存
#     dict(生产 StatefulToolEnv 连真数据库、SandboxEnv 起容器)
