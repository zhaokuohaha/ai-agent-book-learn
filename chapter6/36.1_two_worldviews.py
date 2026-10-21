"""chapter6/36.1_two_worldviews.py — 回合制 vs 事件驱动：两种世界观的最小现场

本实验新增的主概念是「回合制是交互约定，不是环境性质」——我们默认
「用户说完 → Agent 想一段 → 答一句」，期间世界静止；这个前提自然到
很少被当成假设写出来，但它只是早期工具调用接口的组织方式。真实世界
不等人：邮件在思考时到达、CI 在动作中途红了、用户会插话——环境是
连续的、并发的，回合制只是给接口降了速。

坐标系定位：观察空间 × 动作空间是 Agent 的能力坐标系——前五章（上下文
工程/记忆/RAG/工具/代码）扩的是两空间的「内容」，本章沿「模态」（形式）
与「时机」（节奏）两条轴扩展；时机轴先走：纯架构问题不涉及新 I/O，是
迈出回合制的最小一步（D37 异步的模态完全没变，还是文本）。

两种世界观的可运行对照（零 API——概念日演示时序本身，sleep 模拟思考；
确定性事件脚本保证可验收）：

    A 回合制（D30 主循环的世界观）：观察只发生在回合开始（q.get），
      思考+动作期间世界被默认静止（新事件在队列里堆积），CI 失败要等
      当前回合结束才被看到——响应延迟是结构性的
    B 事件驱动（本章的世界观）：队列唤醒；处理中的任务分块执行、
      高紧急事件可抢占（课件机器人档「动作分块·可被抢占」的最小形态）；
      动作跨回合——任务发起后不阻塞，靠收尾事件 close（任务句柄的伏笔）

运行：uv run python -X utf8 chapter6/36.1_two_worldviews.py（无需 .env，约 4 秒）
"""

import queue
import threading
import time

# 世界的事件脚本：确定性时序（同一脚本喂两套循环，差别只在世界观）
SCRIPT = [
    (0.0, "task", "每日摘要任务", "low"),     # 跨回合任务：发起后靠收尾事件
    (0.2, "mail", "新邮件：老板催周报", "mid"),  # 同步轻任务：分块处理可被打断
    (0.5, "ci", "CI 失败：主分支红了", "high"),  # 高紧急：必须抢占
    (1.2, "task_done", "每日摘要完成", "low"),   # 收尾事件：跨回合动作的另一半
]


def run_world(q: queue.Queue) -> None:
    """事件源：世界主动推送——不问 Agent 有没有空（时延按脚本走）"""
    t0 = time.time()
    for at, kind, body, urg in SCRIPT:
        time.sleep(max(0.0, at - (time.time() - t0)))
        q.put({"kind": kind, "body": body, "urgency": urg, "t": round(time.time() - t0, 2)})


def turn_based(q: queue.Queue) -> tuple[float, list]:
    """A 回合制：回合开始才观察，思考期间世界静止（事件堆积），动作本回合做完"""
    ci_seen, backlog = None, []
    t0 = time.time()
    while True:
        ev = q.get()                      # 观察：只发生在回合开始
        backlog.append(ev)
        if ev["kind"] == "task_done":
            break
        print(f"    [A {time.time() - t0:4.1f}s] 回合开始处理 {ev['body']}"
              f"（思考+动作 0.5s，期间世界被假设静止）")
        if ev["kind"] == "ci":
            ci_seen = round(time.time() - t0 - ev["t"], 1)  # 延迟=开始处理时刻-到达时刻
        time.sleep(0.5)                   # 本回合必须做完
    return ci_seen, backlog


def event_driven(q: queue.Queue) -> tuple[float, list]:
    """B 事件驱动：队列唤醒 + 高紧急抢占分块任务 + 动作跨回合靠收尾事件"""
    ci_seen, seen = None, []
    t0 = time.time()
    handles = set()                       # 任务句柄（D37 伏笔：跨回合动作的追踪）
    while True:
        ev = q.get()                      # 世界推送唤醒（阻塞等待，不忙轮询）
        seen.append(ev)
        if ev["kind"] == "task_done":
            print(f"    [B {time.time() - t0:4.1f}s] 收尾事件：每日摘要任务关闭（动作跨回合）")
            break
        if ev["kind"] == "task":
            handles.add("每日摘要")
            print(f"    [B {time.time() - t0:4.1f}s] 发起「{ev['body']}」→ 挂句柄不阻塞（跨回合）")
            continue
        if ev["kind"] == "ci":
            ci_seen = round(time.time() - t0 - ev["t"], 1)
            print(f"    [B {time.time() - t0:4.1f}s] 高紧急到达 → 立即处理 CI 失败")
            continue
        for step in range(3):             # 轻任务分块执行：每块之间让世界有机会插话
            time.sleep(0.1)
            if not q.empty() and q.queue[0]["urgency"] == "high":
                print(f"    [B {time.time() - t0:4.1f}s] 第 {step + 1}/3 块后被抢占——"
                      f"邮件处理让位（可被打断）")
                break
        else:
            print(f"    [B {time.time() - t0:4.1f}s] 「{ev['body']}」处理完成")
    return ci_seen, seen


if __name__ == "__main__":
    for name, runner in (("A 回合制", turn_based), ("B 事件驱动", event_driven)):
        print(f"=== {name}：同一事件流（0.0 任务 / 0.2 邮件 / 0.5 CI 失败 / 1.2 收尾） ===")
        q: queue.Queue = queue.Queue()
        world = threading.Thread(target=run_world, args=(q,), daemon=True)
        world.start()
        t0 = time.time()
        ci_seen, seen = runner(q)
        world.join()
        print(f"  → CI 失败响应延迟 {ci_seen}s｜事件 {len(seen)} 条全部处理｜"
              f"总时长 {time.time() - t0:4.1f}s\n")

    print("=== 对照结论 ===")
    print("  同一个世界：A 把 0.5s 到达的 CI 压到当前回合结束才看见（结构性延迟），")
    print("  思考期间到达的邮件全部堆积；B 靠队列唤醒即时看到、高紧急抢占分块中的任务、")
    print("  摘要任务发起即挂句柄等收尾事件——回合制只是约定，世界本来就不等人。")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（零 API，约 4 秒，确定性时序）：
#   1. run_world 线程按脚本时序投递 4 个事件（0.0/0.2/0.5/1.2s）
#   2. A turn_based：逐事件 q.get → sleep 0.5 处理（回合制）→ CI 排到
#      0.2 邮件回合之后才开始被处理
#   3. B event_driven：任务发起即挂句柄不阻塞；邮件分 3 块处理（0.1s/块），
#      第 3 块时 CI 到达（0.5s）触发抢占演示；1.2s 收尾事件关闭句柄
# 验证内容（时序确定性，输出数值可复现）：
#   - A 的 CI 响应延迟 ≈ 0.5s（邮件回合 0.2~0.7s 把 0.5s 到达的 CI 压到
#     1.0s 才开始处理）；B 的 CI 延迟 ≈ 0.0~0.1s（到达即处理/抢占中让位）
#     ——同一世界两种命运
#   - A 输出可见「思考+动作 0.5s，期间世界被假设静止」；邮件在 CI 回合前
#     被串行排队——堆积效应的直接证据
#   - B 输出可见「挂句柄不阻塞」「第 3/3 块后被抢占」「收尾事件关闭」——
#     动作跨回合 + 可抢占的最小现场（D37 任务句柄的伏笔）
#   - 简化注：抢占用「块间查队列」模拟（sleep 本身不可中断）——生产用
#     asyncio.cancel/线程中断；事件源生产上是消息队列/webhook，非脚本线程
