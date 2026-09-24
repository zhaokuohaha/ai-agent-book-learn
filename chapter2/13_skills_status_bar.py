"""chapter2/13_skills_status_bar.py — Agent Skills 渐进式披露 + 状态栏

核心概念（原书第 2 章 · W2 Day 13）：

Skills：Agent 世界的 npm 包（作业 1 → 实验 A）
    系统提示词越塞越胖 = 上下文腐化（token 浪费 + 注意力稀释）。Skills 的
    解法是渐进式披露（Progressive Disclosure）三层：
    ① 元数据目录常驻上下文——SKILL.md 的 frontmatter（name+description），
       成本极低；description 要写成路由条件（"何时该用我"），不是功能介绍，
       "help with backend" 这种宽泛描述等于没有路由，还要给反例防误触发
    ② 核心流程按需加载——模型读目录判断需要后调 load_skill；Claude Code
       的关键细节：tool result 只是"正在启动"占位符，正文作为 user 消息在
       调用位置注入（本实验按此实现，见 <skill> 注入）
    ③ 细则再按需深入（本实验最小版不展开）
    "对 KV Cache 友好"≠零成本：目录/正文首次加载都要算，但之后前缀稳定可复用。

状态栏：只有一半的检索引擎（作业 2+3 → 实验 B）
    上下文学习像检索不像推理：注意力"找"原始记录很强，"数数/归纳"很弱——
    规则要求每家最多打 2 次电话，Agent 却常打到第 4 次：次数以原始记录
    散落 KV Cache，模型每次现场重数。解法：框架把隐式状态提前算好
    （工具调用计数器 + 时间），包 <agent_status> 借 user 槽位追加到末尾
    ——为什么借 user 槽位？改 system 毁整个前缀缓存（D10 铁律），追加
    末尾则前缀全保；末尾位置 = 注意力权重最高处。
    本实验用"每轮替换"实现（删旧状态再追加，只失效一轮后缀）。
    实验数据：加状态栏后小模型准确率接近前沿大模型，思考 token/延迟/
    成本降约一个数量级（无栏时思考量随上下文持续增长，有栏后基本恒定）。

三条踩坑警示（写进记忆）：
    ① 模型几乎无条件相信状态栏——状态必须由代码确定性维护，绝不让 LLM
       批量统计（LLM 数数不可靠，且状态栏可能被投毒）
    ② 状态栏是原始上下文的有损投影——只提前算了"预想会被问到"的维度，
       删原始记录要谨慎
    ③ 无侵入性优点：不用微调，任何模型直接生效

作业落点：
    1. 迷你 Skill 加载器 → 实验 A：parse_skill 解析 frontmatter 生成目录、
       目录常驻 system、正文经 load_skill 触发后以 user 消息注入
    2. 状态栏 → 实验 B：call_merchant 计数器 + 时间，<agent_status> 注入
    3. 对照实验 → 实验 B：同一任务有/无状态栏各跑一遍，数每家电话次数

运行：uv run python -X utf8 chapter2/13_skills_status_bar.py（需 .env，约 10 次真实调用）
"""

import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL


# ---------- 实验 A：两个测试 Skill（SKILL.md = frontmatter 路由条件 + 正文） ----------

GIT_COMMIT_SKILL_MD = """---
name: git-commit
description: 何时用：用户让你写/润色 Git 提交信息时。何时不适用：代码审查、分支管理、写 CHANGELOG。
---
# Git 提交规范
- 格式：<type>: <一句话主题>（type 只用 feat/fix/refactor/docs/test/chore）
- 主题 ≤ 50 字符，祈使语气（"添加"而不是"添加了"），不加句号
- body 可省略；有则空一行，说清"为什么改"而不是"改了什么"
- 禁止：把多个不相关改动塞进一个提交；主题写"WIP"、"update"等无信息词
- 正例：fix: 修复订单查询在空 ID 时的 500 错误
- 反例：update: 改了一些东西
"""

SQL_REVIEW_SKILL_MD = """---
name: sql-review
description: 何时用：用户给出 SQL 让你审查安全性/正确性时。何时不适用：从零写 SQL、性能调优、ORM 配置。
---
# SQL 审查要点
- 注入：拼接用户输入即红灯，必须参数化
- 权限：UPDATE/DELETE 必须带 WHERE；影响行数大的要有 LIMIT 保护意识
- 事务：转账类双写必须同事务；DDL 前确认是否锁全表
- 输出：逐条给"风险 + 建议"，别只说"有问题"
"""


def parse_skill(md):
    """SKILL.md → (name, description, body)：frontmatter 是路由层，正文是流程层"""
    _, meta, body = md.split("---", 2)
    fields = dict(line.split(":", 1) for line in meta.strip().splitlines())
    return fields["name"].strip(), fields["description"].strip(), body.strip()


PARSED = [parse_skill(s) for s in (GIT_COMMIT_SKILL_MD, SQL_REVIEW_SKILL_MD)]
SKILL_BODIES = {name: body for name, _, body in PARSED}
CATALOG = "\n".join(f"- {name}: {desc}" for name, desc, _ in PARSED)  # 目录常驻，正文不进


@tool
def load_skill(name: str) -> str:
    """按需加载一项技能的完整说明；name 必须来自技能目录中的名称。"""
    if name not in SKILL_BODIES:
        return f"错误：技能不存在 {name}（可选：{', '.join(SKILL_BODIES)}）"
    return f"Skill '{name}' 正在启动，完整说明见紧接着注入的 user 消息。"  # 占位符，不承载正文


SKILL_SYSTEM = f"""你是开发助手。可用技能目录如下（目录常驻；判断当前任务需要某项技能时，
先调用 load_skill 加载完整说明，再按说明行事；无关任务直接回答）：

<skills_catalog>
{CATALOG}
</skills_catalog>"""


def skill_loader_demo(model):
    """实验 A：目录常驻 → 模型自主触发 → 占位符回执 + 正文以 user 消息注入"""
    task = "我刚重构完配置加载模块，帮我写个提交信息。"
    print(f"[任务] {task}")
    model = model.bind_tools([load_skill])  # 给菜单：模型才知道有 load_skill 可调
    messages = [SystemMessage(content=SKILL_SYSTEM), HumanMessage(content=task)]
    for turn in range(3):
        response = model.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            print(f"[回答] {response.text}")
            break
        for call in response.tool_calls:
            observation = load_skill.invoke(call)
            messages.append(ToolMessage(content=observation.content, tool_call_id=call["id"]))
            print(f"[turn {turn + 1}] load_skill({call['args']}) → 占位符回执")
            body = SKILL_BODIES[call["args"]["name"]]  # Claude Code 做法：正文以 user 消息在调用位置注入
            messages.append(HumanMessage(content=f"<skill name='{call['args']['name']}'>\n{body}\n</skill>"))
    roles = " → ".join({HumanMessage: "user", SystemMessage: "system", ToolMessage: "tool"}.get(type(m), "assistant") for m in messages)
    print(f"[轨迹角色] {roles}")
    print("[结构验证] 目录在 system 常驻；正文注入后出现在轨迹中部——不是塞进 system（前缀不动，D10）")


# ---------- 实验 B：状态栏对照（计数器 + 时间，<agent_status> 注入末尾） ----------

MERCHANTS = ("联通", "移动", "电信")
CALL_HISTORY = Counter()  # (商家) → 已拨次数；代码确定性维护，绝不让 LLM 数数


@tool
def call_merchant(merchant: str) -> str:
    """给指定商家打催办电话（模拟）。每家最多允许打 2 次。"""
    CALL_HISTORY[merchant] += 1
    n = CALL_HISTORY[merchant]
    if n == 1:
        return f"第 {n} 次呼叫 {merchant}：无人接听，已留语音留言。"
    return f"第 {n} 次呼叫 {merchant}：接通，客服答复 48 小时内处理。"


@tool
def check_plan(merchant: str) -> str:
    """查询在指定商家的套餐状态（模拟）。"""
    return f"{merchant}：宽带迁移单状态=待装机，工单无异常。"


WORK_TOOLS = {t.name: t for t in (call_merchant, check_plan)}

WORK_SYSTEM = """你是宽带迁移催办助手。规则：每家商家最多打 2 次电话；
打不通可以先查套餐状态判断工单是否异常，再决定下一步；全部办完或确定
无法推进时，汇报总结（每家当前状态 + 共打了几通电话）。"""


def status_bar():
    """状态栏：隐式状态 → 显式知识。代码维护（Counter），LLM 只读不数"""
    counts = ", ".join(f"{m}={CALL_HISTORY[m]}/2 次" for m in MERCHANTS)
    return (
        "<agent_status>\n"
        f"当前时间：{datetime.now():%H:%M:%S}\n"
        f"电话计数：{counts}\n"
        f"约束检查：{'；'.join(f'{m} 已达上限' for m in MERCHANTS if CALL_HISTORY[m] >= 2) or '均未达上限'}\n"
        "</agent_status>"
    )


def run_task(model, task, use_status_bar, max_turns=8):
    """同一任务：use_status_bar=False 对照组 / True 实验组（每轮替换注入）"""
    CALL_HISTORY.clear()
    messages = [SystemMessage(content=WORK_SYSTEM), HumanMessage(content=task)]
    injected = None  # 上一轮状态消息：每轮替换 = 删旧追加，只失效一轮后缀
    for _ in range(max_turns):
        if use_status_bar:
            if injected is not None:
                messages.remove(injected)
            injected = HumanMessage(content=status_bar())
            messages.append(injected)  # 借 user 槽位贴末尾：注意力最高处
        response = model.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return response.text
        for call in response.tool_calls:  # 模型决策，框架执行
            observation = WORK_TOOLS[call["name"]].invoke(call)
            messages.append(ToolMessage(content=observation.content, tool_call_id=call["id"]))
    return "（达到轮数上限）"


def status_bar_suite(model):
    """实验 B：同一任务跑两遍，数每家电话次数、查是否超限"""
    model = model.bind_tools(list(WORK_TOOLS.values()))  # 工具菜单：决策权在模型
    task = ("请帮我催办联通、移动、电信三家的宽带迁移进度。每家最多打 2 次电话，"
            "打不通就先查一下套餐状态再决定。")
    results = {}
    for use in (False, True):
        label = "有状态栏" if use else "无状态栏"
        print(f"\n--- [{label}] ---")
        answer = run_task(model, task, use)
        print(f"  汇报节选：{answer[:120]}")
        print(f"  电话计数：{dict(CALL_HISTORY)}")
        results[use] = dict(CALL_HISTORY)
    violated = lambda c: any(v > 2 for v in c.values())
    print("\n[对照结论]")
    for use in (False, True):
        label = "有状态栏" if use else "无状态栏"
        c = results[use]
        print(f"  {label}：总通话 {sum(c.values())} 通，每家 {sorted(c.values())}，"
              f"{'违反上限！' if violated(c) else '未违反上限'}")
    print("  机制：状态栏组无需从原始记录里数数——约束检查由代码提前算好贴在末尾（检索引擎补上提炼层）。")
    print("  实测留档：本组两组均未超限——任务规模小、模型较强时数数不难；原书实验中")
    print("  0.6B 小模型无栏时频繁打到第 4 次，有栏后稳定遵守——轨迹越长/模型越小，状态栏价值越大。")


if __name__ == "__main__":
    model = ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0,
    )
    print("=== 实验 A：迷你 Skill 加载器（目录常驻，正文按需） ===")
    skill_loader_demo(model)
    print("\n=== 实验 B：状态栏对照（同一任务，无状态栏 vs 有状态栏） ===")
    status_bar_suite(model)

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑：
#   1. 实验 A（skill_loader_demo）：目录常驻 system（frontmatter 解析出的
#      name+description）→ 模型自主调 load_skill → 占位符回执 + 正文以
#      user 消息在调用位置注入 → 按 Skill 规范产出提交信息
#   2. 实验 B（status_bar_suite）：同一催办任务跑两遍（无/有状态栏），
#      有栏组每轮删旧状态、status_bar() 生成最新 <agent_status> 借 user
#      槽位贴末尾（每轮替换式）；计数由 Counter 代码维护，LLM 只读不数
# 验证内容：
#   - 实验 A 轨迹角色 system → user → assistant → tool → user → assistant
#     （Skill 正文注入位置一目了然；若忘 bind_tools，模型会在文本里
#     "假装调工具"——首次运行实测踩过）
#   - 实验 B 电话计数每家 ≤ 2（本组两均达标；原书 0.6B 无栏时打到第 4 次——
#     轨迹越长/模型越小，状态栏价值越大）
