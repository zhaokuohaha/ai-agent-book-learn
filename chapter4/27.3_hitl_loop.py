"""chapter4/27.3_hitl_loop.py — HITL：人工介入的超时降级与反馈闭环

前面的闸门是机器审（26.3 Sidecar、26.4 循环拦截）；机器拿不准或风险太高时，
最后一级闸门是「人」——HITL（Human-in-the-loop）。但人工介入不是一发一收就完，
要设计成学习闭环（三件套）：

    超时降级   请求可能没人理——设超时阈值 + 默认保守行为（「N 分钟无响应就
               拒绝高危操作」）。保守的含义：高危默认拒绝、低危可默认放行
    优先级队列 人的注意力是稀缺资源——紧急请求多渠道通知（电话+群+红灯），
               普通请求只发邮件
    反馈沉淀   人类的批准/拒绝及理由不是一次性消费——可归纳的原则进知识库或
               Skill（D13），隐式偏好形成后训练数据。沉淀让第 N 次同类决策
               更快、更一致

本实验四幕剧本（模拟人工响应器；生产里换成 Slack/webhook 等待 + 超时定时器）：
    幕1 批准：高危删除 → 多渠道通知 → 批准（理由进沉淀账本）→ 执行
    幕2 拒绝：另一高危操作 → 拒绝（理由进账本）→ 回执回主 Agent
    幕3 超时：无人响应 → 超时降级为保守拒绝——「超时」也是一种拒绝来源
    幕4 回放：同类请求再现 → 先检索沉淀账本命中历史 → 人工复用理由快速批准

设计要点：审批申请结构化（操作/风险/理由）；回执一律「为什么+出路」——与
26.4 的拒绝回执同构：拒绝是信息，不是死路。

运行：uv run python -X utf8 chapter4/27.3_hitl_loop.py（纯本地，0 次 API 调用）
"""

GATE_TIMEOUT = 2.0  # 教学版压缩；生产里是分钟级
LEDGER: list[dict] = []  # 沉淀账本：批准/拒绝 + 理由——供后续同类请求检索

REQUESTS = [
    ("删除 /data/old_snapshots（例行清理）", "high"),
    ("清空审计日志表 audit_log", "high"),
    ("删除 /data/orders 旧分区", "high"),
    ("再次删除 /data/old_snapshots（例行清理）", "high"),
]


def notify(op: str, risk: str) -> None:
    """优先级队列：紧急走多渠道，普通只发邮件——注意力按风险花"""
    channels = "电话 + 群消息 + 控制台红灯" if risk == "high" else "邮件"
    print(f"  [通知] {channels} ← 待审：{op}")


def scripted_human(op: str):
    """剧本化「人工」：批准 / 拒绝 / 不在（None）——生产里换成 Slack/webhook 等待"""
    if "audit_log" in op:
        return "rejected", "审计日志有合规保留要求，且无备份"
    if "old_snapshots" in op:
        return "approved", "例行清理，已确认冷备"
    return None  # /data/orders 无人过问 → 超时降级


def target_of(op: str) -> str:
    for key in ("old_snapshots", "audit_log", "orders"):
        if key in op:
            return key
    return op


def lookup_ledger(op: str) -> list:
    """反馈沉淀的用法：同类请求先查历史——决策更快、更一致"""
    return [e for e in LEDGER if target_of(e["op"]) == target_of(op)]


def request_approval(op: str, risk: str) -> tuple:
    """HITL 闸门：通知 → 等待窗口 → 三态（批准 / 拒绝 / 超时降级）"""
    notify(op, risk)
    resp = scripted_human(op)
    if resp is None:
        print(f"  [超时] 等待窗口 {GATE_TIMEOUT:.0f}s 内无人响应 → 降级保守行动：高危默认拒绝")
        return "rejected", f"超时无人响应（阈值 {GATE_TIMEOUT:.0f}s），保守默认拒绝"
    return resp


def settle(op: str, decision: str, reason: str) -> None:
    """反馈沉淀：理由进账本——批准/拒绝都不是一次性消费"""
    LEDGER.append({"op": op, "decision": decision, "reason": reason})
    print(f"  [沉淀] 账本 +1：{decision}（{reason}）")


def receipt(op: str, decision: str, reason: str) -> str:
    """回执同 26.4：拒绝也要写「为什么+出路」——模型拿到它才能重规划"""
    if decision == "approved":
        return f"已执行：{op}"
    return f"[HITL 拒绝] {op} 未执行：{reason}。请换更安全方案或补充材料后请求人工复核。"


if __name__ == "__main__":
    print(f"=== HITL 闭环：审批三态 + 反馈沉淀（模拟人工；超时阈值 {GATE_TIMEOUT:.0f}s）===")
    for i, (op, risk) in enumerate(REQUESTS, 1):
        print(f"\n--- 幕{i}：{op} ---")
        hits = lookup_ledger(op)
        if hits:
            print(f"  [检索] 沉淀命中 {len(hits)} 条：")
            for h in hits:
                print(f"         · {h['decision']}（{h['reason']}）")
        decision, reason = request_approval(op, risk)
        print(f"  [回执→主 Agent] {receipt(op, decision, reason)}")
        settle(op, decision, reason)

    print(f"\n看沉淀账本：{len(REQUESTS)} 次请求、3 种结果（批准/拒绝/超时降级）。幕3 的拒绝")
    print("理由是「超时」——无人值守也兜得住；幕4 因命中幕1 的理由而快速批准——反馈")
    print("沉淀让第 N 次决策更快更一致（生产里账本换知识库/Skill，跨会话复用）")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（纯本地，0 次 API 调用；「人工」为剧本化模拟）：
#   幕1 批准：高危 → 多渠道通知 → 批准（理由：已确认冷备）→ 执行 + 沉淀
#   幕2 拒绝：audit_log → 拒绝（理由：合规保留）→ 回执（为什么+出路）→ 沉淀
#   幕3 超时：orders → 无人响应 → 降级保守拒绝（超时也是一种拒绝来源）→ 沉淀
#   幕4 回放：再删 old_snapshots → 先命中幕1 记录 → 人工复用理由快速批准
# 验证内容：
#   - 通知渠道按 risk 分发（本剧本全 high；普通只发邮件的分支在 notify 里）
#   - 三种结果都有回执；拒绝回执含「为什么+出路」——与 26.4 同构，可直接回填
#     主循环驱动重规划
#   - 幕3 的拒绝理由是「超时」而非人工拒绝——降级机制在工作（保守默认）
#   - 幕4 先打印「沉淀命中 1 条」再审批——学习闭环让决策更快、更一致
#   - 真实接线：scripted_human 换 Slack/webhook 等待 + 超时定时器；理由归档进
#     知识库/Skill（D13）；隐式偏好（总批准某类/总拒绝某类）作后训练数据
