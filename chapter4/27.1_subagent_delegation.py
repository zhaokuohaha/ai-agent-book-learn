"""chapter4/27.1_subagent_delegation.py — 子 Agent 委托：四要素岗位书与上下文交接

任务超出单个 Agent 的能力边界时，协作工具让它把子任务委托出去、再整合结果。
为什么不造一个「全能 Agent」？专业化分工——类比后端微服务：单体膨胀后按业务域
拆分，每个服务独立迭代独立部署；一组各自专精的子 Agent（独立提示词/工具集/
知识库）比一个巨型单体好维护。

本实验看委托的「接口」——交出去时给子 Agent 两样东西：

① 岗位书 = 提示词四要素（make_persona 里逐条对应）：
   角色清晰   开门见山「你是专门负责订单数据分析的助手 Agent」
   来源标注   [FROM_MAIN_AGENT] / [FROM_USER] / [TOOL_RESULT]——标注出处，
              防止混淆来源引发提示注入（D12 信任分层在协作里的落点）
   边界明确   什么归自己管、什么要转交上报
   输出标准化 约定 JSON——降低主 Agent 解析负担，错误处理更可靠

② 交接上下文（本实验的实证变量，作业①）：
   最小化传递：主 Agent 手挑「必要」信息——但「必要」靠主 Agent 判断，容易漏
   LLM 生成交接：主 Agent 先把任务+背景压缩成交接文档（约束带来源标注）再传
   对照现场：用户的「只要 P0 级别」约束（[FROM_USER]）在 A 组漏传——
   同一个子 Agent、同一份数据，口径却全错。交接质量决定委托质量。

运行：uv run python -X utf8 chapter4/27.1_subagent_delegation.py（需 .env，约 3 次调用）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL

# 主 Agent 手里的原始素材
USER_ASK = "统计昨日订单异常率，只要 P0 级别。"  # [FROM_USER]——约束在这里
STATS = "昨日订单总量 1100 单；异常 177 单，其中 P0 级 12 单、P1 级 45 单、P2 级 120 单。"  # [TOOL_RESULT]
TASK = "统计昨日订单异常率"


def make_persona(task: str, ctx: str) -> str:
    """子 Agent 的岗位书——四要素按序落进提示词"""
    return (
        "你是专门负责订单数据分析的助手 Agent。\n"  # ① 角色清晰
        f"[FROM_MAIN_AGENT] {task}\n"  # ② 来源标注
        f"[CONTEXT] {ctx}\n"
        "只处理数据分析范围内的事；超出边界的事项转交上报。\n"  # ③ 边界明确
        '只输出 JSON：{"rate": "异常率", "basis": "口径说明"}'  # ④ 输出标准化
    )


def delegate(model: ChatOpenAI, task: str, ctx: str) -> str:
    """一次委托 = 岗位书 + 上下文 → 子 Agent 一次推理"""
    return str(model.invoke([HumanMessage(make_persona(task, ctx))]).content)


HANDOFF_PROMPT = (
    "你是主 Agent。把下面的用户原话和背景数据压缩成给子 Agent 的交接文档："
    "保留用户的目标和所有约束，标注信息来源（[FROM_USER] / [TOOL_RESULT]），简洁不丢要点。\n"
    f"[FROM_USER] {USER_ASK}\n[TOOL_RESULT] {STATS}"
)


if __name__ == "__main__":
    model = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0)

    print("=== 岗位书预览（四要素 = 委托接口）===")
    print(make_persona("[任务]", "[上下文]"))

    print("\n=== A 组：最小化传递（主 Agent 只塞了任务和原始数据）===")
    print(delegate(model, TASK, STATS))

    print("\n=== B 组：LLM 生成交接文档（主 Agent 先压缩再传）===")
    handoff = str(model.invoke([HumanMessage(HANDOFF_PROMPT)]).content)
    print(f"交接文档：{handoff}\n")
    print(delegate(model, TASK, handoff))

    print("\n看对照：同一个子 Agent、同一份数据——口径差异全部来自交接（不在子 Agent 本身）")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（约 3 次真实调用）：
#   1. 打印岗位书——四要素逐条落在提示词里
#   2. A 组：只传「任务+原始数据」→ 子 Agent 输出
#   3. 主 Agent 生成交接文档（1 次调用）→ B 组子 Agent 输出
# 验证内容（模型输出动态，验收看口径差异不赌固定文案）：
#   - 岗位书四要素齐全（角色/来源/边界/输出格式）
#   - A 组口径 = 全部异常（177/1100 ≈ 16.1%）——「只要 P0」约束丢失
#   - B 组交接文档保留 [FROM_USER] 约束 → 输出 P0 口径（12/1100 ≈ 1.1%）
#   - 若 B 组交接也丢约束（概率小）：重跑或加强交接提示词（22.3 方法论）
#   - 结论：子 Agent 再专精，喂错交接也白搭——委托的接口设计比子 Agent 能力
#     更决定成败；「来源标注」顺带是注入防线（[TOOL_RESULT] 里的指令 ≠ 用户要求）
