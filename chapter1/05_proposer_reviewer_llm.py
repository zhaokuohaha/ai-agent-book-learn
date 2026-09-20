"""chapter1/05_proposer_reviewer_llm.py — 提议者-审核者：实际 LLM 版本（LangChain）

本文件是 04 的实际 LLM 落地：
    - 提议者和审核者是两个独立 LLM 实例（不共享上下文）
    - 完整流程：任务输入 → 提议者产出 → 审核者评判 → 通过/驳回 → 落地
    - 审核者只看产物本身，不看提议者的推理过程（隔离性）
    - 轨迹 append-only：每一步追加进 AuditLog，可重放可审计

与 04 的区别：
    - 04 用规则函数演示"只看产物"的隔离性
    - 05 用真实 LLM 演示两个独立实例扮演不同角色（提议者 vs 审核者）

使用 LangChain 生态：ChatOpenAI 统一接口，便于切换模型。

运行：uv run chapter1/05_proposer_reviewer_llm.py（需先填好 .env）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 定位项目根目录

from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from common.config import API_KEY, BASE_URL, MODEL

# ---------- LLM 初始化（LangChain 统一接口） ----------

llm = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL)

# ---------- 只增不改（Append-only）：可缓存、可重放、可审计 ----------


@dataclass
class Event:
    kind: str      # propose / approve / reject
    payload: str   # 产物内容
    prev: str      # 前一个事件的指纹


class AuditLog:
    """只增不改：只有 append，没有 update / delete"""
    def __init__(self):
        self.events, self.last = [], ""

    def append(self, kind: str, payload: str) -> Event:
        e = Event(kind, payload, self.last)
        self.events.append(e)
        self.last = hash((e.kind, e.payload, e.prev))
        return e


# ---------- 提议者-审核者（实际 LLM，LangChain） ----------


def proposer_llm(task: str, feedback: str = "") -> str:
    """提议者 LLM：只管产出，system prompt 强调'你是提议者'"""
    system = "你是 SQL 提议者。根据用户任务生成 SQL 查询。只输出 SQL，不解释。"
    if feedback:
        system += f"\n上次提议被驳回，原因：{feedback}。请改进。"
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=f"任务：{task}")])
    return resp.content.strip()


def reviewer_llm(artifact: str) -> tuple[str, str]:
    """审核者 LLM：只看产物本身，system prompt 强调'你是审核者，只看 SQL'"""
    system = "你是 SQL 审核者。只根据 SQL 本身判断是否安全（无注入风险、语法正确）。输出 'approve' 或 'reject: 原因'。"
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=f"SQL：{artifact}")])
    output = resp.content.strip()
    if output.lower().startswith("approve"):
        return "approve", "安全"
    # "reject: 原因"
    reason = output.split(":", 1)[1] if ":" in output else output
    return "reject", reason


# ---------- 完整流程：提议 → 审核 → 驳回 → 重新提议 → 通过 → 落地 ----------


def run_with_review(task: str, max_rounds: int = 3):
    """提议者-审核者循环（实际 LLM）"""
    log = AuditLog()
    feedback = ""
    for round_num in range(1, max_rounds + 1):
        draft = proposer_llm(task, feedback)
        log.append("propose", draft)
        print(f"  [round {round_num}] propose: {draft}")

        verdict, reason = reviewer_llm(draft)
        log.append(verdict, f"{draft} | {reason}")
        print(f"  [round {round_num}] {verdict}: {reason}")

        if verdict == "approve":
            return f"落地：{draft}", log
        feedback = reason
    return "达到最大轮数，未通过", log


if __name__ == "__main__":
    # 正常任务：提议者生成 SQL，审核者判断安全，一轮通过
    result, log = run_with_review("查询用户 tom 的信息")
    print(f"\n最终结果：{result}")
    print(f"轨迹长度：{len(log.events)}")
