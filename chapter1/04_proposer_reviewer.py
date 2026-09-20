"""chapter1/04_proposer_reviewer.py — 贯穿全书的设计模式（上）：提议者-审核者 + 只增不改

Day 4 核心概念（原书第 1 章）：

五个设计模式（全书公共词汇表，后续 7 章反复出现）：
    1. 提议者-审核者   产出与评判由两个不共享上下文的角色分别承担
    2. 渐进式披露     先给目录，按需加载细节（优化上下文预算 + 选择精度）
    3. 只增不改       状态以追加方式演进，写下的不再回头修改
    4. 边界集+保留集  修改同时在两批样本上验证（应改变的 + 不应影响的）
    5. 最小 diff+可回滚  每次修改尽量小、带来源、可单独回滚

本文件聚焦模式 1 + 3：

提议者-审核者（Proposer-Reviewer）：
    成立前提：自审不可靠——同一上下文里的模型很难发现自己的认知盲区，也很难
    察觉自己已被注入。审核者看到的是产物本身（渲染结果、测试输出、调用参数），
    而不是产出方的推理过程。
    后端类比：没有人允许自己 approve 自己的 PR；生产变更的 maker-checker 双人复核。
    书中落点：第 3 章知识更新、第 4 章工具调用的事前审批与事后验证、第 7 章评估。

只增不改（Append-only）：
    状态以追加方式演进，写下的内容不再回头修改。换来三个性质：
    可缓存、可重放、可审计。
    后端类比：事件溯源、Kafka 日志、MySQL binlog——全是 append-only，所以能
    重放恢复、能审计追责。
    关键落点：KV Cache 前缀稳定（改动越靠前，作废的缓存越多）；新工具 schema
    追加到轨迹末尾，而不是插回前缀。

代码重点：
    - AuditLog 只有 append，没有 update/delete——轨迹可重放可审计
    - 审核者只看产物本身，不看提议者的推理过程——隔离性
    - 演示完整循环：提议 → 审核 → 驳回 → 重新提议 → 通过 → 落地
    - 轨迹链式指纹（Event.prev）：每个事件指向前一个，形成审计链

运行：uv run chapter1/04_proposer_reviewer.py（纯规则模拟，无需 API key）
"""

from dataclasses import dataclass

# ---------- 只增不改（Append-only）：可缓存、可重放、可审计 ----------


@dataclass
class Event:
    kind: str      # propose / approve / reject
    payload: str   # 产物内容
    prev: str      # 前一个事件的指纹（形成链式审计）


class AuditLog:
    """只增不改：只有 append，没有 update / delete"""
    def __init__(self):
        self.events, self.last = [], ""

    def append(self, kind: str, payload: str) -> Event:
        e = Event(kind, payload, self.last)
        self.events.append(e)
        self.last = hash((e.kind, e.payload, e.prev))  # 链式指纹：把 prev 也哈希进来，任一事件被篡改后续指纹全变
        return e


# ---------- 提议者-审核者（Proposer-Reviewer） ----------


def proposer(task: str, feedback: str = "") -> str:
    """提议者：根据任务和反馈产出 SQL（演示场景，会产出带注入风险的 SQL）"""
    if feedback:  # 被驳回后，简化处理：返回安全 SQL
        return "SELECT id, name FROM users WHERE name = 'safe_user'"
    return f"SELECT * FROM users WHERE name = '{task}'"


def reviewer(artifact: str) -> tuple[str, str]:
    """审核者：只看产物本身，返回 (verdict, reason)"""
    if ";" in artifact or "--" in artifact:
        return "reject", "SQL 包含多语句分隔符或注释，疑似注入"
    return "approve", "SQL 安全"


# ---------- 完整流程：提议 → 审核 → 驳回 → 重新提议 → 通过 → 落地 ----------


def run_with_review(task: str, max_rounds: int = 3):
    """提议者-审核者循环：最多 max_rounds 轮，通过则落地"""
    log = AuditLog()
    feedback = ""
    for round_num in range(1, max_rounds + 1):
        draft = proposer(task, feedback)
        log.append("propose", draft)
        print(f"  [round {round_num}] propose: {draft}")

        verdict, reason = reviewer(draft)
        log.append(verdict, f"{draft} | {reason}")
        print(f"  [round {round_num}] {verdict}: {reason}")

        if verdict == "approve":
            return f"落地：{draft}", log
        feedback = reason  # 把驳回原因传给提议者
    return "达到最大轮数，未通过", log


if __name__ == "__main__":
    # 危险任务：第一次提议带注入，被审核者拦截；第二次重新提议安全 SQL，通过
    result, log = run_with_review("tom'; DROP TABLE users; --")
    print(f"\n最终结果：{result}")
    print(f"轨迹长度：{len(log.events)}（可重放可审计）")
