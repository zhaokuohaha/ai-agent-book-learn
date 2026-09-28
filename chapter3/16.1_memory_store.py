"""chapter3/16.1_memory_store.py — 用户记忆系统（下）：三分类、仅追加与跨会话

核心概念（原书第 3 章 · W3 Day 16）：

认知三分类（存什么）
    三套记忆体系各答一个问题：层次答"存在哪里"、格式答"怎么存"、三分类答"存什么"：
    情景 episodic = 操作日志（某次订了东京航班）、语义 semantic = 配置表（素食者）、
    程序 procedural = SOP 脚本（搜直飞→确认座位→用常旅客号）。三分类与存储格式
    正交（语义可用 Simple Notes 存，程序可用 JSON Cards 存）；与 15 的 category
    也正交——category 答"业务上是什么"，mtype 答"认知上是什么"（走哪条检索路由）。

User as Code + 仅追加（怎么管）
    四种格式全是文本，聚合与冲突检测全靠 LLM"心算"不可靠——User as Code 把记忆
    改成带类型对象 + 普通函数规则（active 同 key 取最新：能确定性计算的，别让
    模型算）。Mem0 v2 写入时消歧（ADD/UPDATE/DELETE/NOOP 四选一，误删不可逆）
    → v3 仅追加 + 检索时按时间取当前（LoCoMo 71.4→92.5）。实现见
    common/memory_store.py：平铺事件流，与 15 的"当前值+history"嵌套构成同一
    问题的两种解。Agent 接入见 16.2。

作业落点：跨会话存取（load/save 走 JSON 文件，session1/session2 两个真实进程
    重启验证）+ 每条记忆带时间戳与类型 + 旧偏好被新偏好覆盖且保留历史（靠窗→过道）。

运行：uv run python -X utf8 chapter3/16.1_memory_store.py session1   # 先跑：入库落盘
      uv run python -X utf8 chapter3/16.1_memory_store.py session2   # 后跑：重启召回+覆盖
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.memory_store import MemoryStore

STORE_PATH = Path(__file__).parent / "memory_store.json"  # 记忆文件：跨进程存活的物证
TS_1 = "2026-09-24"  # 会话 1：与 15 同日，衔接订票剧本；写死保证实验可复现
TS_2 = "2026-09-28"  # 会话 2：四天后"重启"（即今天）


def session1():
    """会话 1：三类记忆入库并落盘。条目为人工预置——模拟 15 实验 LLM 提取（ETL）
    的产物；本实验聚焦存取层，不重复提取部分"""
    STORE_PATH.unlink(missing_ok=True)  # 每次演示从零开始
    store = MemoryStore(STORE_PATH)

    print(f"=== 会话 1（{TS_1}，首次对话）：会话结束 → 提取入库 → save ===")
    facts = [  # 三分类各就各位：语义=配置表，情景=操作日志，程序=SOP
        ("diet", "素食者", "semantic"),
        ("seat_preference", "靠窗座位", "semantic"),
        ("trip", "2026-09-24 预订 NH960 上海→东京（靠窗）", "episodic"),
        ("booking_flow", "订机票流程：先搜直飞 → 确认座位偏好 → 素食餐备注 → 里程累计 MileagePlus 12345678", "procedural"),
    ]
    for key, content, mtype in facts:
        store.remember(key, content, mtype, TS_1)
        print(f"  [入库] {mtype:<11} {key} = {content}")

    print(f"\n[save] {len(store.entries)} 条记忆落盘：{STORE_PATH.name}——进程即将退出，")
    print("记忆在盘上不在内存里。接着换个进程验证重启召回：")
    print("  uv run python -X utf8 chapter3/16.1_memory_store.py session2")


def session2():
    """会话 2：全新进程 load 召回（作业 1 验收）+ 跨会话的偏好覆盖（作业 2）"""
    print(f"=== 会话 2（{TS_2}，新进程重启）：load() 跨会话召回 ===")
    store = MemoryStore(STORE_PATH)  # 上一进程的内存早已销毁，全靠文件恢复
    if not store.entries:
        raise SystemExit("记忆文件为空或不存在——请先运行 session1 写入记忆")
    print(f"[load] 从 {STORE_PATH.name} 恢复 {len(store.entries)} 条记忆：")
    for e in store.entries:
        print(f"  [{e.mtype}] {e.key} = {e.content}（{e.ts}）")

    print("\n[检索路由] 三分类各答一类问题（by_type 分流，参考架构图 3-4）：")
    for mtype, question in [
        ("semantic", '"用户是谁、偏好什么？"'),
        ("episodic", '"上次做了什么？"'),
        ("procedural", '"这类任务该怎么做？"'),
    ]:
        items = ", ".join(f"{k}={e.content}" for k, e in sorted(store.by_type(mtype).items()))
        print(f"  {question} → {mtype}：{items}")

    print(f"\n[作业 2] 偏好覆盖（{TS_1} 靠窗 → {TS_2} 改口过道），仅追加不删旧：")
    print("  用户：最近晕机，坐靠窗反而难受，还是过道吧。")
    store.remember("seat_preference", "过道座位", "semantic", TS_2)
    print(f"  当前事实（同 key 取 ts 最新）：{store.active()['seat_preference'].content}（{TS_2}）")
    print("  完整历史（时间推理与审计的原料）：")
    for e in store.history("seat_preference"):
        print(f"    {e.ts}  {e.content}")
    print("  → '我以前是不是喜欢靠窗？'可答：是，09-24 前后靠窗，09-28 起改为过道")
    print("  （重复跑 session2 会堆叠同值条目——仅追加日志的脏数据，生产靠压缩治理）")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "session1":
        session1()
    elif mode == "session2":
        session2()
    else:
        print("用法：先 session1（入库落盘），再 session2（重启召回 + 覆盖演示）")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（纯 Python，零 LLM 调用；两个模式 = 两个真实进程）：
#   1. session1：删旧文件从零演示 → 三类记忆各入库（semantic×2 / episodic×1 /
#      procedural×1，15 订票对话的"ETL 产物"）→ 每条 remember 即 save → 进程退出
#   2. session2：新进程构造 MemoryStore 即 load → 打印召回全景 → by_type 演示
#      检索路由 → 覆盖演示（靠窗→过道：只追加新条目）→ active 取最新 +
#      history 完整时间线
# 验证内容：
#   - session2 的 [load] 能列出 session1 写入的 4 条——进程重启后召回（作业 1 ✓）
#   - 路由演示：semantic=饮食/座位、episodic=行程、procedural=订票流程
#   - 覆盖后 active=过道（09-28）、history=[靠窗(09-24), 过道(09-28)]——
#     当前值取最新、历史全保留（作业 2 ✓）
