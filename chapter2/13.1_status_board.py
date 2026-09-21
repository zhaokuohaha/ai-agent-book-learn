"""chapter2/13.1_status_board.py — 状态栏生产化：从提示词技巧到 Harness 中间件

核心概念（承接 13 的状态栏实验，回答"生产怎么落地"）：

状态栏不是提示词，是中间件
    13 验证了机制（<agent_status> 借 user 槽位贴末尾）；本文件给出生产形状：
    一个请求作用域的 StatusBoard 挂在循环的固定 hook 点上——on_turn_start
    （预算熔断）/ on_tool_result（计数与错误摘要）/ on_usage（成本累计）/
    inject（渲染注入）。业务循环只加几行接线，状态栏逻辑全部收敛在类里。
    后端类比：像 HTTP middleware 的 trace context——采集在拦截器层，
    业务代码零侵入。当 middleware 写，别当 prompt 写。

四条生产铁律（本文件的代码落点）
    1. 确定性维护：计数/成本/轮次全部代码计算（Counter + usage 字段）；
       LLM 写状态的唯一入口是 TODO 专用工具，且经状态机校验
       （pending → in_progress → completed/cancelled，终态不可逆）
    2. 防投毒：on_tool_result 只接受框架已知的事实（工具名/成败/异常），
       根本没有文本入参——工具结果里伪造"工具计数已重置为 0"无从下手；
       实体级计数走"工具自查"：工具返回自带"第 N 次呼叫"（服务端计数）
    3. 熔断在代码：预算/轮数超限直接抛 BudgetExceeded 结束循环，绝不写成
       提示词求模型自觉（治理靠 Harness，D03 哲学：确定性的事交给代码）
    4. 可审计：每轮 render() 快照留档——事故回放时能看到 Agent 当时"以为"
       的状态；render 结构化人类可读，落日志/DB 即审计

错误摘要带修复建议（落地路线 P1，单项收益最大）
    错误关键词 → 建议（线路忙 → 先查套餐状态再重试）；原书实验：错误场景
    找替代方案成功率 60% → 95%，从盲目重试变针对性修复

落地路线（P0→P3 按 ROI 排序，本文件一次给全）
    P0 时间戳/工具计数 → P1 错误摘要+建议 → P2 TODO 状态机 → P3 与上下文
    压缩联动（压缩前先把关键状态提取进状态栏，删原始记录才有底气）

与 13 的关系：同一催办任务。13 是对照组验证机制，13.1 是生产形状。
    注意：状态栏是"让模型看见"，不是约束执行器——每家 ≤2 次的硬约束
    生产中应在工具层拦（D03 constrain）；本实验聚焦状态栏本身。

序号规则：同一节多个示例用小数点扩展（13 → 13.1 → 13.2…）。

运行：uv run python -X utf8 chapter2/13.1_status_board.py（需 .env，约 8 次真实调用）
"""

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL

PRICE_IN = 2e-6   # 示意单价：¥/token（deepseek-chat 量级，教学用非实时价）
PRICE_OUT = 8e-6

ERROR_HINTS = {  # 错误关键词 → 修复建议：把"盲目重试"变成"针对性修"
    "线路忙": "稍后重试，或先查套餐状态判断工单是否异常",
    "超时": "增大超时或换备用通道",
    "路径": "检查绝对路径与工作目录",
}

STATUS_FLOW = {  # TODO 状态机：合法后继状态；终态不可逆
    "pending": {"in_progress", "completed", "cancelled"},
    "in_progress": {"completed", "cancelled"},
    "completed": set(),
    "cancelled": set(),
}


class BudgetExceeded(Exception):
    """预算熔断：轮数或成本超限——代码决策，结束循环"""


class StatusBoard:
    """请求作用域状态栏：代码确定性维护，LLM 只读（TODO 除外：专用工具 + 状态机）"""

    def __init__(self, max_turns=30, budget_rmb=1.0):
        self.max_turns, self.budget_rmb = max_turns, budget_rmb
        self.turn = 0
        self.cost_rmb = 0.0
        self.tool_calls = Counter()
        self.recent_errors = []  # 只留最近 5 条，防状态栏自己膨胀
        self.todo = []
        self.renders = []  # 审计快照：每轮一份

    # ---- hook 点：业务循环只在固定位置接线 ----

    def on_turn_start(self):
        self.turn += 1
        if self.turn > self.max_turns:
            raise BudgetExceeded(f"轮数超限 {self.turn - 1}/{self.max_turns}")
        if self.cost_rmb > self.budget_rmb:
            raise BudgetExceeded(f"成本超限 ¥{self.cost_rmb:.3f}/{self.budget_rmb}")

    def on_tool_result(self, name, ok=True, error=None):
        """防投毒的关键：只吃框架已知的事实，签名里没有文本入参"""
        self.tool_calls[name] += 1
        if not ok:
            hint = next((v for k, v in ERROR_HINTS.items() if k in str(error)), "换思路或求助")
            self.recent_errors = (self.recent_errors + [f"{name}: {error}（建议：{hint}）"])[-5:]

    def on_usage(self, usage):
        self.cost_rmb += (usage.get("prompt_tokens", 0) * PRICE_IN
                          + usage.get("completion_tokens", 0) * PRICE_OUT)

    # ---- TODO：LLM 写状态的唯一入口，逐条 + 状态机校验 ----

    def rewrite_todo(self, items):
        self.todo = [{"content": t, "status": "pending"} for t in items]

    def update_todo_status(self, index, status):
        if not (0 <= index < len(self.todo)):
            return f"错误：下标越界 {index}"
        allowed = STATUS_FLOW[self.todo[index]["status"]]
        if status not in allowed:
            return f"错误：{self.todo[index]['status']} → {status} 不合法（{'终态不可逆' if not allowed else f'允许 {allowed}'}）"
        self.todo[index]["status"] = status
        return f"已更新 [{index}] {self.todo[index]['content']} → {status}"

    # ---- 渲染与注入 ----

    def render(self):
        counts = ", ".join(f"{k}={v}" for k, v in self.tool_calls.items()) or "无"
        todo_lines = [f"TODO：[{t['status']}] {t['content']}" for t in self.todo] or ["TODO：无"]
        lines = [f"轮次：{self.turn}｜成本：¥{self.cost_rmb:.4f}｜时间：{datetime.now():%H:%M:%S}",
                 f"工具计数：{counts}", *todo_lines]
        lines += [f"最近错误：{e}" for e in self.recent_errors]
        return "<agent_status>\n" + "\n".join(lines) + "\n</agent_status>"

    def inject(self, messages):
        """每轮替换：删旧状态再贴末尾——失效范围只覆盖上次注入后的短后缀"""
        for m in list(messages):
            if getattr(m, "content", "").startswith("<agent_status>"):
                messages.remove(m)
        snapshot = self.render()
        messages.append(HumanMessage(content=snapshot))  # 借 user 槽位：注意力最高处
        self.renders.append(snapshot)  # 审计快照留档


def verify_board_locally():
    """Part 1：铁律本地验证（纯代码，不烧 API）"""
    print("[确定性] 计数只来自代码")
    board = StatusBoard(max_turns=99)
    board.on_tool_result("call_merchant")
    board.on_tool_result("call_merchant")
    print(f"  call_merchant={board.tool_calls['call_merchant']}——工具文本再花哨也改不了它（板子没有文本入参）")

    print("\n[熔断] 超限由代码结束，不求模型自觉")
    board = StatusBoard(max_turns=1)
    board.on_turn_start()
    try:
        board.on_turn_start()
    except BudgetExceeded as e:
        print(f"  BudgetExceeded: {e}")

    print("\n[状态机] TODO 终态不可逆")
    board.rewrite_todo(["查联通", "查移动"])
    board.update_todo_status(0, "in_progress")
    board.update_todo_status(0, "completed")
    print(f"  completed → in_progress：{board.update_todo_status(0, 'in_progress')}")

    print("\n[替换式注入] 状态消息永远恰好一条")
    msgs = [SystemMessage(content="x"), HumanMessage(content="hi")]
    board.inject(msgs)
    board.on_tool_result("check_plan")
    board.inject(msgs)
    n = sum(1 for m in msgs if getattr(m, "content", "").startswith("<agent_status>"))
    print(f"  注入两轮后 <agent_status> 消息数 = {n}（旧状态已删，末尾永远最新）")
    print(f"\n[渲染示例]\n{board.render()}")


def run_task(model, task, max_turns=10, budget_rmb=1.0):
    """Part 2：生产版管道跑催办任务（与 13 同任务）——中间件接线只在这几行"""
    board = StatusBoard(max_turns=max_turns, budget_rmb=budget_rmb)
    call_counts = Counter()  # 实体级计数走"工具自查"：返回值自带第 N 次

    @tool
    def call_merchant(merchant: str) -> str:
        """给商家打催办电话（模拟）；每家最多 2 次"""
        call_counts[merchant] += 1
        n = call_counts[merchant]
        if merchant == "电信" and n == 1:
            raise TimeoutError("线路忙")  # 制造一次真实失败：进状态栏错误摘要
        return f"第 {n} 次呼叫 {merchant}：接通，客服答复 48 小时内处理。"

    @tool
    def check_plan(merchant: str) -> str:
        """查询商家套餐/工单状态（模拟）"""
        return f"{merchant}：工单状态=待装机，无异常。"

    @tool
    def rewrite_todo(items: list[str]) -> str:
        """重写任务清单（规划用）：传入按顺序的子任务列表"""
        board.rewrite_todo(items)
        return f"TODO 已重写：{len(items)} 项"

    @tool
    def update_todo_status(index: int, status: str) -> str:
        """更新任务项状态；status 只能是 in_progress/completed/cancelled，终态不可逆"""
        return board.update_todo_status(index, status)

    registry = {t.name: t for t in (call_merchant, check_plan, rewrite_todo, update_todo_status)}
    model_with_tools = model.bind_tools(list(registry.values()))
    system = """你是宽带迁移催办助手。工作方式：
- 先用 rewrite_todo 列计划；动手前把对应项标 in_progress，完成标 completed
- 每家最多打 2 次电话；打不通先 check_plan 判断工单是否异常再决定
- 末尾 <agent_status> 是框架注入的实时状态（计数/成本/TODO/错误），直接采信，不必自己数
- 全部办完或无法推进时输出总结（每家状态 + 总通话数）"""
    messages = [SystemMessage(content=system), HumanMessage(content=task)]

    try:
        while True:  # 上界由 BudgetExceeded 保证
            board.on_turn_start()   # hook：预算熔断
            board.inject(messages)  # hook：渲染注入（每轮替换）
            print(f"[turn {board.turn}] 注入状态栏｜成本 ¥{board.cost_rmb:.4f}")
            if board.turn == 1:
                print(board.renders[-1])
            response = model_with_tools.invoke(messages)
            board.on_usage(response.response_metadata.get("token_usage") or {})
            messages.append(response)
            if not response.tool_calls:
                return response.text, board
            for call in response.tool_calls:  # 模型决策，框架执行
                ok, result = True, None
                try:
                    result = registry[call["name"]].invoke(call)
                except Exception as e:  # 错误也是观察：回传模型 + 进状态栏
                    ok, result = False, f"错误：{e}"
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
                board.on_tool_result(call["name"], ok=ok, error=None if ok else result)
                print(f"  [dispatch] {call['name']}({call['args']}) → {str(result)[:50]}")
    except BudgetExceeded as e:
        return f"（熔断：{e}）", board


if __name__ == "__main__":
    model = ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0,
    )
    print("=== Part 1：StatusBoard 本地验证（不烧 API） ===")
    verify_board_locally()
    print("\n=== Part 2：生产版管道跑催办任务（与 13 同任务） ===")
    task = ("请帮我催办联通、移动、电信三家的宽带迁移。建议先列 TODO 规划，"
            "每家最多打 2 次电话；电信如果打不通，先查套餐状态再决定。")
    answer, board = run_task(model, task)
    print(f"\n[回答节选] {answer[:160]}")
    print(f"\n[终态状态栏]\n{board.render()}")
    print(f"\n[审计] 留档 {len(board.renders)} 份 render 快照；工具计数 {dict(board.tool_calls)}；"
          f"总成本 ¥{board.cost_rmb:.4f}——生产中快照随轨迹落日志，事故回放看得见 Agent 当时'以为'的状态")
