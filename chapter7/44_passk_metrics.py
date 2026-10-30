"""chapter7/44_passk_metrics.py — Pass@k 与 Pass^k:能力上限 vs 业务可靠性

本实验新增的主概念是 **两个评估指标的对偶**:
- Pass@k = 1-(1-p)^k:同一任务独立跑 k 次,至少一次过就算过(连续得分取最好
  叫 Best@k)。回答「这件事原则上做得到吗」——能力上限。后端类比:消息队列
  at-least-once,重试足够多次至少成功一次。科研发现/漏洞挖掘/开放创作值钱的
  就是它(人可以从 k 条候选里挑最好那条);Anthropic 长时运行 Agent、Manus
  虚拟电脑、早期活人感,社交网络传播的都是那条最好的轨迹。
- Pass^k = p^k:同一任务连续跑 k 次,每次都过、且都不触发一票否决项(安全/
  合规/幻觉)。回答「能否稳定可靠交付」——业务可靠性。后端类比:SLA 承诺
  「连续 N 天零故障」,任何一次事故都违约。支付/退款/权限变更/生产部署看它。

对偶:p 固定、k 增大 → Pass@k 单调趋近 1(多试总会中),Pass^k 单调趋近 0
(次数越多越难全对)。「多跑几次」对两类指标改善方向完全相反——这正是它们
各自服务不同目标的原因。

指标没有绝对好坏,只有场景匹配:昨天 τ²-bench 五过四(0.8)本身说明不了问题
——退款客服 0.8 = 每 5 个用户 1 个拿不到退款(不可接受),挖漏洞 Agent 0.8 =
5 次里命中 4 次(很可观)。定义指标第一步不是选公式,是问「我的场景对成功
率的要求是什么」。

数字概念——公式即本体(38.1 时间轴同款:不真跑评估,直接算 + 蒙特卡洛眼见
为实)。两个工程坑在文末注释。

运行:uv run python -X utf8 chapter7/44_passk_metrics.py(零 API,瞬时)
"""

import random


def pass_at_k(p: float, k: int) -> float:
    """能力上限:k 次独立采样至少一次成功(at-least-once 语义)"""
    return 1 - (1 - p) ** k


def pass_hat_k(p: float, k: int, veto: float = 0.0) -> float:
    """业务可靠性:k 次每次都成功且不触发一票否决(SLA 连续零故障语义)
    veto = 单次触发安全/合规/幻觉的概率——触发即该次判失败,与功能成功与否无关"""
    return (p * (1 - veto)) ** k


def simulate(p: float, k: int, n: int = 20000, veto: float = 0.0, seed: int = 42):
    """蒙特卡洛:模拟 n 组、每组 k 次独立运行,数出经验频率(眼见为实,对照公式)"""
    rnd = random.Random(seed)
    at_least = all_clean = 0
    for _ in range(n):
        func = [rnd.random() < p for _ in range(k)]            # 每次功能是否成功
        clean = [f and rnd.random() >= veto for f in func]     # 成功且不触发一票否决
        at_least += any(func)      # Pass@k:至少一次功能成功(OR)
        all_clean += all(clean)    # Pass^k:每次都干净通过(AND)
    return at_least / n, all_clean / n


def demo(p: float, k: int, veto: float = 0.0):
    """一组对照:公式值 vs 蒙特卡洛经验值——两者应吻合(公式不是拍脑袋)"""
    sa, sp = simulate(p, k, veto=veto)
    print(f"  p={p} k={k}" + (f" veto={veto}" if veto else ""))
    print(f"    Pass@k 公式 {pass_at_k(p, k):9.4%}  ← 模拟 {sa:9.4%}   能力上限:至少中一次(OR)")
    print(f"    Pass^k 公式 {pass_hat_k(p, k, veto):9.4%}  ← 模拟 {sp:9.4%}   业务可靠:每次都中(AND)")


if __name__ == "__main__":
    print("=== 幕一 · 场景绑定:成功率 0.8 本身没有意义(呼应昨天 τ²-bench 5 过 4) ===")
    p = 0.8
    print(f"  退款客服(交付型):单次失败率 {1 - p:.0%} → 每 5 个用户就有 1 个拿不到退款,不可接受;")
    print(f"                    它要的是 Pass^k——连续 5 单全对仅 {pass_hat_k(p, 5):.1%}。")
    print(f"  挖漏洞 Agent(探索型):Pass@5 = {pass_at_k(p, 5):.2%} → 5 次里几乎必命中一次,已很可观。")
    print("  同一个 0.8,业务含义天差地别——指标第一步不是选公式,是问场景要什么。\n")

    print("=== 幕二 · 记忆锚点:p=0.6, k=5(差 12 倍以上) ===")
    demo(0.6, 5)
    print(f"  鸿沟:Pass@5={pass_at_k(0.6, 5):.1%}(几乎总能成一次) vs Pass^5={pass_hat_k(0.6, 5):.1%}(连续全对很难),")
    print(f"        倍数差 ≈ {pass_at_k(0.6, 5) / pass_hat_k(0.6, 5):.0f} 倍——「演示很惊艳」与「能上生产」之间隔着的工程鸿沟。\n")

    print("=== 幕三 · 对偶:p=0.6 固定,k 从 1→10 扫描(多跑几次,方向相反) ===")
    print("    k │ Pass@k(↑趋近1) │ Pass^k(↓趋近0)")
    for k in range(1, 11):
        print(f"   {k:>2} │    {pass_at_k(0.6, k):8.3%}    │    {pass_hat_k(0.6, k):8.3%}")
    print("  【停一停】为什么「多跑几次」对 Pass@k 是好事、对 Pass^k 是坏事?")
    print("    → Pass@k 是 OR(至少中一次):多跑一次多一次机会,单调↑趋近 1;")
    print("      Pass^k 是 AND(每次都中):多跑一次多一道关卡,单调↓趋近 0——对偶的根源。\n")

    print("=== 幕四 · 作业③ p=0.9,k=10:高单次成功率下依然悬殊 + 一票否决被 k 放大 ===")
    demo(0.9, 10)
    print(f"  单次 90% 已很高,但 Pass^10={pass_hat_k(0.9, 10):.1%}(仅约 1/3)——连续 10 次全对依然难;")
    print(f"  Pass@10={pass_at_k(0.9, 10):.4%} 近乎必然。这就是高 p 下两者的悬殊。")
    print("  叠加一票否决(每次 5% 概率幻觉/违规,触发即该次失败—— Pass^k 的 AND 会放大它):")
    demo(0.9, 10, veto=0.05)
    print(f"    → Pass^10 从 {pass_hat_k(0.9, 10):.1%} 掉到 {pass_hat_k(0.9, 10, 0.05):.1%};而 Pass@k 不看 veto(只问能不能中一次)仍 {pass_at_k(0.9, 10):.2%}。")

# ---- 执行逻辑与验证(看完代码再回头看这里) ----
# 执行逻辑(零 API 瞬时,纯数值):
#   1. pass_at_k / pass_hat_k 直接算解析公式;simulate 蒙特卡洛采样经验频率
#   2. 四幕:场景绑定(0.8 两种命运)→ 锚点(0.6/5)→ 对偶扫描(k=1..10)→
#      作业③(0.9/10)+ veto 放大;demo() 把公式值与模拟值并排对照
# 验证内容(数字确定性,可手算复核):
#   - 公式 vs 模拟吻合:0.6/5 → Pass@5=98.976%、Pass^5=7.776%(模拟 ±0.5%);
#     0.9/10 → Pass@10≈99.99999999%、Pass^10≈34.87%
#   - 对偶单调:幕三表 Pass@k 随 k 递增趋 1、Pass^k 递减趋 0(方向相反)
#   - veto 放大:0.9/10/0.05 → Pass^10 = (0.9×0.95)^10 ≈ 20.9% < 34.87%,
#     而 Pass@10 不受 veto 影响(能力上限只问能否中一次)
#   - 简化注:模拟用伯努利独立采样近似真实评估(真评估每次是一个完整 Agent
#     轨迹);veto 建模为独立概率,真实一票否决是规则/裁判判定的确定性事件
# 两个工程坑:
#   ① 口径必须写清:「k 次」是同一任务 k 次独立采样(本实验),还是流水线连续
#     k 个不同任务?混淆会让指标失去意义。
#   ② 副作用操作不能盲目重试:退款/部署这类真实影响,「重试直到成功」会把
#     at-least-once 的失败转嫁给用户;正解是沙箱/可回滚环境采样,每次失败如实
#     计入可靠性——不能因「最后成功了」抹掉中间的失败记录。
