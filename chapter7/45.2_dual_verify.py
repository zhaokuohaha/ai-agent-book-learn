"""chapter7/45.2_dual_verify.py — 验证器深度:双重验证,别信 Agent 的自我报告

本实验新增的主概念是 **验证器必须核实机器可独立复核的事实,而非 Agent 的
自我报告**。Agent 很容易洋洋洒洒写「全部完成」,实际没完成。

典范是 SWE-bench Verified 的双重验证:
  FAIL_TO_PASS  修复前失败、修复后通过——证明问题真解决了(新测试红变绿)
  PASS_TO_PASS  修复前后均通过——证明没引入新缺陷(老测试不能绿变红)
只查前者:Agent 可以「敷衍性修复」蒙混——压制症状(让新测试过)却不解决根因;
只查后者:等于没查(bug 压根没碰)。这正是回归测试的铁律。

场景:discount(price, vip) 有 bug(vip 该打 8 折却返回原价)。三种 Agent 提交
——真修复 / 无条件打折(蒙混)/ 没改,自我报告都说「修好了」。看双重验证如何
拦截那个「看起来修好了」的蒙混版。类比:别只看接口返回 200,要查副作用是否
落库(read your writes);Terminal-Bench 更狠——要求从源码编译内核并在 QEMU
启动日志出现自定义消息,Agent 无法伪造。确定性,零 API。

运行:uv run python -X utf8 chapter7/45.2_dual_verify.py
"""


# ---- 被测源码的三个版本(模拟 Agent 对代码库的提交;改的是源码不是测试)----
def discount_buggy(price, vip):
    """修复前(基线):vip 也应打折,但代码返回原价——这是 bug"""
    return price


def discount_good(price, vip):
    """真修复:区分 vip——解决根因"""
    return price * 0.8 if vip else price


def discount_cheat(price, vip):
    """敷衍性修复:无条件打折——让 vip 测试过(压制症状),却把普通用户也打了折(没解决根因)"""
    return price * 0.8


# ---- 评估方持有的固定测试集(Agent 动不了它——删改仓库测试也没用)----
FAIL_TO_PASS = [("vip 应打 8 折: discount(100,True)==80", lambda d: d(100, True) == 80)]
PASS_TO_PASS = [("普通用户原价: discount(100,False)==100", lambda d: d(100, False) == 100),
                ("零价保持零: discount(0,True)==0", lambda d: d(0, True) == 0)]


def run_suite(version, suite):
    """跑一组测试,返回逐项 (名称, 通过?)"""
    return [(name, bool(fn(version))) for name, fn in suite]


def dual_verify(candidate, baseline=discount_buggy):
    """双重验证:FAIL_TO_PASS 要「基线红→候选绿」,PASS_TO_PASS 要「基线绿→候选仍绿」"""
    f2p, p2p = run_suite(candidate, FAIL_TO_PASS), run_suite(candidate, PASS_TO_PASS)
    f2p_ok = not all(ok for _, ok in run_suite(baseline, FAIL_TO_PASS)) and all(ok for _, ok in f2p)
    p2p_ok = all(ok for _, ok in run_suite(baseline, PASS_TO_PASS)) and all(ok for _, ok in p2p)
    return f2p, p2p, f2p_ok, p2p_ok


if __name__ == "__main__":
    candidates = [
        ("真修复", discount_good, "我修好了:vip 现在正确打 8 折"),
        ("敷衍蒙混", discount_cheat, "我修好了:vip 折扣测试已通过!"),   # 自我报告一样自信
        ("啥也没改", discount_buggy, "问题已解决,请验收"),
    ]
    for label, version, report in candidates:
        f2p, p2p, f2p_ok, p2p_ok = dual_verify(version)
        print(f"\n=== {label} · Agent 自我报告:「{report}」===")
        print("  FAIL_TO_PASS(修复前红 → 修复后须绿:证明问题真解决):")
        for name, ok in f2p:
            print(f"    {'✓' if ok else '✗'} {name}")
        print("  PASS_TO_PASS(修复前后都须绿:证明没引入新缺陷):")
        for name, ok in p2p:
            print(f"    {'✓' if ok else '✗'} {name}")
        print(f"  → 只查 FAIL_TO_PASS:{'放行' if f2p_ok else '拦截'}"
              f"｜双重验证(F2P ∧ P2P):{'放行' if (f2p_ok and p2p_ok) else '拦截'}")

    print("\n" + "=" * 60)
    print("三个版本自我报告都说「修好了」,验证器核实机器可复核的事实后:")
    print("  · 只查 FAIL_TO_PASS:敷衍蒙混版(无条件打折)会放行——vip 测试确实绿了(被骗)")
    print("  · 双重验证:PASS_TO_PASS 抓到它把普通用户也打了折(绿变红)→ 拦截")
    print("  · 啥也没改版:FAIL_TO_PASS 仍红 → 两法都拦截(等于没查的只 P2P 才会漏放)")
    print("铁律:新测试红变绿(FAIL_TO_PASS)+ 老测试不绿变红(PASS_TO_PASS)——")
    print("     别信自我报告,核实副作用是否真的落地(read your writes)。")

# ---- 执行逻辑与验证(看完代码再回头看这里)----
# 执行逻辑(零 API 确定性,纯函数 + 断言):
#   1. discount 三版本 = 三种 Agent 提交;FAIL_TO_PASS/PASS_TO_PASS 是评估方
#      固定测试集(Agent 改源码不改测试——删改仓库断言在这里无效)
#   2. dual_verify:对每版跑两组测试,F2P 要「基线红→候选绿」、P2P 要「基线绿→
#      候选仍绿」;主逻辑打印逐项 ✓/✗ + 只查 F2P vs 双重验证的放行/拦截对照
# 验证内容(可手算复核):
#   - 真修复:F2P 80==80 ✓、P2P 100==100 ✓ 0==0 ✓ → 双重放行
#   - 敷衍蒙混:F2P cheat(100,True)=80 ✓(只查 F2P 会放行!)、P2P cheat(100,
#     False)=80≠100 ✗ → 双重拦截(PASS_TO_PASS 抓到误伤普通用户)
#   - 啥也没改:F2P buggy(100,True)=100≠80 ✗ → 两法都拦截
#   - 简化注:用函数字典模拟测试套件(生产是 pytest 跑真实测试文件);三版本
#     手写(生产是 Agent 生成的 diff);基线 red/green 校验确保题目本身有效
