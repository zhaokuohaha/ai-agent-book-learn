"""chapter5/29.3_failure_recovery.py — 故障恢复：分级升级、熔断与死亡螺旋

本实验新增的主概念是「API 层故障恢复的三级升级」——按对用户透明度分级，
能用低级别解决就不升级：
    一级 静默重试：可重试错误（限流/网络抖动）→ 指数退避 + 随机抖动。
         抖动不是花活：大量客户端同时退避会同步醒来形成二次拥塞
    二级 降级接续：截断续写/换备用模型——本实验不展开（1、3 两级已能看清
         「分级升级，逐级透明」的骨架；完整版是原书实验 5-1/5-2 的重活）
    三级 暴露用户：所有自动手段用尽才报错，并附上已尝试的恢复动作——绝不裸报错

伴生概念两个：
    熔断上限——每条恢复路径都要有上限。Claude Code 压缩重试上限 3 次来自真实
    事故统计：曾有会话在这条路径连续失败三千余次，仅这类无效重试每天全球
    约 25 万次 API 调用的浪费——3 是「绝大多数故障已在此前恢复」与「继续
    重试基本无望」之间的经验拐点。
    死亡螺旋——错误处理路径自身又调 LLM → 再次出错 → 连锁反应。真实事故
    形态：上下文溢出 → 停止钩子想调 LLM 写提交信息 → 又溢出 → 又触发钩子。
    防护两条：错误路径禁用一切会再调 LLM 的副作用逻辑；深度计数器打断残余连锁。

工具层错误走另一条路（不终止会话、错误变输入）——见 29.2 的循环。
本实验全程用模拟客户端：故障确定性可复现，零 API 调用、无需 .env。

运行：uv run python -X utf8 chapter5/29.3_failure_recovery.py
"""

import random
import time

BASE = 0.4        # 退避基数（秒）——教学版缩短；生产常见 1s 起步
JITTER = 0.3      # 抖动上限：错开同步重试，防二次拥塞
MAX_ATTEMPTS = 3  # 熔断上限——Claude Code 压缩重试的同款数字


class RateLimit(Exception):
    """模拟 HTTP 429——与任务内容无关的基础设施噪声（可重试的典型）"""


class ContextOverflow(Exception):
    """模拟上下文溢出——死亡螺旋剧本的火种"""


def flaky_llm(n_fail: int):
    """模拟客户端：前 n_fail 次调用抛限流，之后恢复——故障可确定性复现"""
    state = {"calls": 0}

    def invoke(prompt: str) -> str:
        state["calls"] += 1
        if state["calls"] <= n_fail:
            raise RateLimit(f"HTTP 429（第 {state['calls']} 次调用被限流）")
        return f"「{prompt}」的回答"
    return invoke


def with_retry(fn, prompt: str) -> str:
    """一级恢复（退避+抖动）；重试用尽 → 熔断 → 升级三级：暴露用户（附尝试记录）"""
    tried = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn(prompt)
        except RateLimit as e:
            tried.append(f"第 {attempt} 次：{e}")
            if attempt == MAX_ATTEMPTS:  # 熔断：不再重试，把球交给「暴露用户」
                raise RuntimeError(f"恢复手段用尽（重试 {MAX_ATTEMPTS} 次均被限流）。已尝试：{'；'.join(tried)}")
            backoff = BASE * 2 ** (attempt - 1)
            wait = backoff + random.uniform(0, JITTER)
            print(f"    [静默重试] 退避 {wait:.2f}s（指数 {backoff:.1f}s + 抖动）…")
            time.sleep(wait)


def bad_stop_hook():
    """事故版停止钩子：主请求溢出 → 钩子调 LLM 写提交信息 → 再溢出 → 再触发……"""
    try:
        raise ContextOverflow("主请求上下文溢出")
    except ContextOverflow:
        return bad_stop_hook()  # 错误路径上的 LLM 调用再次失败——连锁燃料


def good_stop_hook() -> str:
    """防护版：错误路径禁用一切会再调 LLM 的副作用——宁丢一次辅助功能"""
    try:
        raise ContextOverflow("主请求上下文溢出")
    except ContextOverflow:
        return "已跳过自动提交（错误路径不调 LLM）——丢失一次辅助功能，掐断连锁燃料"


if __name__ == "__main__":
    print("=== 场景 1：可重试错误 → 一级静默重试（用户全程无感知）===")
    r = with_retry(flaky_llm(2), "分析销售数据")
    print(f"  [成功] {r}——两次限流被退避吸收，用户只看到这次回答")

    print("\n=== 场景 2：持续限流 → 熔断 → 三级暴露用户（附尝试记录）===")
    try:
        with_retry(flaky_llm(99), "分析销售数据")
    except RuntimeError as e:
        print(f"  [暴露用户] {e}")

    print("\n=== 场景 3：死亡螺旋——错误路径上的 LLM 副作用 ===")
    try:
        bad_stop_hook()
    except RecursionError:
        print("  A 事故版：连锁直到撞上 Python 递归上限（约千层）才被打断——")
        print("    真实事故形态是循环而非递归：曾有会话在压缩恢复路径连续失败 3000+ 次")
    print(f"  B 防护版：{good_stop_hook()}")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（零 API 调用，约 2 秒——退避真睡）：
#   1. 场景 1：前 2 次限流 → 两行 [静默重试]（0.4-0.7 / 0.8-1.1s 递增）→ 第 3 次成功
#   2. 场景 2：持续限流 → 3 次重试 → 熔断 → RuntimeError 附完整尝试记录
#   3. 场景 3A：bad_stop_hook 无限连锁 → RecursionError（连锁爆炸的活证据）
#      3B：good_stop_hook 一行放弃副作用——一行之差，连锁燃料 vs 干净失败
# 验证内容：
#   - 场景 1 的退避等待逐次翻倍（0.4→0.8 基数）且每次带随机抖动（两次运行数字不同）
#   - 场景 2 的报错文案含 3 次尝试记录——「附上已尝试的动作」，不是裸报错
#   - 场景 3A 以 RecursionError 收场；3B 正常返回——错误路径禁副作用即防螺旋
#   - 对照 29.2：工具层错误喂回模型（不终止）；API 层错误重试/熔断（不暴露中间态）
#     ——两层各走各路，合起来才是完整的「把错误变成输入」
