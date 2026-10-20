"""chapter5/35.2_trace_replay.py — 轨迹回放与体检：调试第一性原理的落地

D35 项目检查点（本日为调试日，不新增章节概念）：调试 Agent 的第一性
原理——模型只对它看到的上下文负责。任何没进入上下文（和轨迹）的事实，
对模型（和你）都不存在；定位行为异常的第一步不是改提示词，是把当时的
完整上下文打印出来看。后端的 trace（跨服务/耗时/在哪抛错）在 Agent 世界
的对应物 = 轨迹回放。

三件事：
    ① 回放器：读 35.1 跑出的完整轨迹（trace.jsonl 事件流），提取关键
       事件（错误/拒绝/验收）定位到轮，回放末轮「模型看到了什么」
    ② 定位一个此前没查明的 bug：D34 的简版 WAL 截断 300 字、只存工具
       调用漏 assistant 文本——当时轨迹「存在但回放不了」（看不到模型
       说了什么/看到了什么）——可观测性缺陷，35.1 升级完整快照修复
    ③ 五类高频 bug 体检（D35 清单逐条）+ 200 字使用体验（基于 D34/D35
       两轮真实运行：惊喜点=能力上界，失望点=失败模式，最痛缺失=backlog）

运行：先跑 35.1（生成 .agent_ws8/trace.jsonl），再：
uv run python -X utf8 chapter5/35.2_trace_replay.py（零 API——纯读轨迹）
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRACE = Path(__file__).parent / ".agent_ws8" / "trace.jsonl"


def replay(path: Path) -> None:
    """回放器：关键事件定位 + 末轮上下文回放——「模型看到的」逐轮可查"""
    events = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
    turns = {e["turn"] for e in events}
    print(f"=== 回放：{len(events)} 条事件 / {len(turns)} 轮 ===")
    keys = ("[错误]", "[拒绝]", "[超时]", "AssertionError", "验收未通过", "SyntaxError")
    print("--- 关键事件定位（问题出在第几轮，一眼可见） ---")
    hits = 0
    for e in events:
        if e.get("kind") == "tool" and any(k in e["content"] for k in keys):
            hits += 1
            print(f"  [轮 {e['turn']}] {e['name']}：{e['content'][:110]}")
    if not hits:
        print("  （无错误类事件——本轮运行一次通过，轨迹干净）")
    print("--- 末轮上下文回放（模型汇报前最后看到了什么） ---")
    last = max(turns)
    for e in [e for e in events if e["turn"] == last][-3:]:
        head = e.get("name", e["kind"])
        print(f"  [轮 {e['turn']}] {head}：{e['content'][:110]}")


if __name__ == "__main__":
    if not TRACE.exists():
        sys.exit(f"[提示] 先跑 35.1 生成轨迹：{TRACE}")
    replay(TRACE)

    print("\n=== 五类高频 bug 体检（D35 清单逐条） ===")
    print("  1 主循环无终止条件    ✓ 验收判停 + 轮数 12 兜底 + 指纹熔断（D34 接线）")
    print("  2 工具结果没喂回      ✓ append-only 成对追加（AIMessage 与 ToolMessage 成对）")
    print("  3 old_string 匹配失败 ✓ 透明报错 + 提示先 read_file 核对（D31 设计特性，自纠路径）")
    print("  4 超时静默            ✓ 结构化错误带回死前输出（D32：UNBUFFERED+taskkill /T）")
    print("  5 自写工具没校验      △ 自举过沙箱用例验证，但未走内置工具同款审查管道——")
    print("    注册的 extract_json 不经语义审查门（judge 只挂 bash）——35.1 已知最薄弱一环")
    print("  → 命中 0/5 完全缺陷、1/5 部分差距；四类全过靠的是 D30-D33 每天补的课")

    print("\n=== 200 字使用体验（基于 D34/D35 两轮真实轨迹，逐条可回放验证） ===")
    print("  惊喜：没人教它先跑测试——D34 SYSTEM 一句「先跑它看失败」，轮 1 就 bash 跑")
    print("  测试定位；D35 更甚，轮 1 先 glob+read 验收测试再动手，write_file 一次写")
    print("  对，4 轮 9 秒交活——验收驱动的收敛速度肉眼可见。")
    print("  失望：D34 轮 3 同轮两处 edit 看着利落，轮 4 复跑仍挂——第一遍没修干净，")
    print("  多花 3 轮才收敛；D35「先写 2-3 行设计」半生效：设计被写进 textstat.py")
    print("  的 docstring 而不是对话，上下文里没有对账锚点（D29 设计介入点的反面观察）。")
    print("  最痛缺失：上下文无压缩——两轮任务短没炸，任务一长错误信息全量堆叠就会")
    print("  翻旧账（D14/D25 还没接进主循环）。下一迭代 backlog。")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（零 API，纯读 35.1 的轨迹文件）：
#   1. replay：读 trace.jsonl 全部事件——关键事件（[错误]/[拒绝]/Assertion/
#      验收未通过）按轮定位打印；末轮最后 3 条事件回放「模型汇报前看到了什么」
#   2. 五类体检：逐条打印 ✓/△ 与证据出处（哪天补的课/已知差距）
#   3. 体验：三段式打印（惊喜/失望/最痛缺失——W7 定量评估的肉眼版）
# 验证内容：
#   - 轨迹事件数/轮数与 35.1 汇报一致；assistant 事件含文本 content（D34
#     漏存的——本回放器正是靠它工作，即「定位 D34 未查明 bug」的证据）
#   - 关键事件若为空：说明 35.1 一次通过（干净轨迹也是合法回放结果）；
#     若有：按轮定位——「第 N 轮报错 → 后续轮自纠」在事件流里一目了然
#   - 末轮回放能看到模型最终汇报前收到的最后上下文（验收结果/工具输出）
#   - 若 35.1 未跑：友好退出提示先跑 35.1
