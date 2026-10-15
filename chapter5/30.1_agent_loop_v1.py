"""chapter5/30.1_agent_loop_v1.py — Agent 主循环 v1：骨架、规约与三道保险丝

本实验新增的主概念是「主循环」——Agent 运行时 = 事件循环：LLM 是调度器
（每轮决定下一个动作）、工具调用是 syscall（进「内核态」干活再返回）、
最终文本是进程退出。模型本身无状态（D09：API 像 HTTP 一样无状态），
让它「持续工作」的正是这个外层循环。Claude Code / Codex 的内核都是它，
差异只在循环内外的工程质量——今天把工程件装上。

与 D09 / 29.2 的分工：
    D09 手拼 messages 看清「全量重发」；29.2 用循环跑测试闭环（聚焦完成
    标准）；本实验聚焦循环自身的工程规约——循环从「能用」升级到
    「有界、可审计」：
    ① 工具环规约：assistant(tool_calls) 与 tool 结果成对追加、append-only
       ——messages 是 WAL 预写日志（D16 回响）：只追加不改历史，崩溃后读
       log 即可恢复现场；改一个字节 = 前缀缓存全灭（D10 回响）
    ② 三道保险丝（循环必须有界）：轮数上限 / token 预算（usage_metadata
       累计）/ 重复指纹（「工具名+参数」连续 3 次相同 → 无进展熔断——
       29.3 的熔断对象从下游故障换成模型行为）
    ③ 轨迹持久化：每条消息产生即 append 到 trace.jsonl（WAL 正统：先记
       日志再继续，崩溃后 log 尾部就是断点）——中立格式只存 role/content/
       工具名+参数，不存厂商私有字段（评估重放与跨厂商接管的原料，原书
       实验 5-1 的伏笔）

作业落点（今日作业三条全在本实验）：
    ① while + 成对追加 + 无工具调用退出 → run_agent 骨架
    ② 至少两个兜底 → 三道保险丝（指纹自检段不烧 API 确定性验证）
    ③ 观察 messages 逐轮增长 → 每轮打印角色序列 + token（D09 roles() 承接）

运行：uv run python -X utf8 chapter5/30.1_agent_loop_v1.py（需 .env，约 4-5 次调用）
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from common.coding_tools import make_tools
from common.config import API_KEY, BASE_URL, MODEL

WS = Path(__file__).parent / ".agent_ws"

SYSTEM = "你是工作区里的 Coding Agent。总结类结论必须来自你真正读过的文件内容，不凭空猜测。"

TASK = ("了解工作区里这个项目：先看有哪些文件，读项目文档，"
        "最后用两三句话总结项目是做什么的、进展到哪了。")

NOTES = '''# 夜灯项目笔记

夜灯是一个桌面小工具：到点提醒休息（番茄钟）+ 屏幕色温调节。

## 进展
- v0.1 番茄钟已可用（25 分钟专注 / 5 分钟休息）
- v0.2 开发中：色温调节，卡在 Windows API 权限问题
- 待办：导出每日专注报表

## 数据
- 日志落在 focus_log.txt（每天一行：日期,专注分钟数）
'''

SMALL = "备用笔记：周末要买滤纸和咖啡豆。\n"

ROLE = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}


def roles(messages) -> str:
    """角色序列：D09 的作业①模式——每轮看 messages 长什么样（作业③观察点）"""
    return " → ".join(ROLE[m.type] for m in messages)


def make_dup_guard(limit: int = 3):
    """重复指纹保险丝：同一「工具名+参数」连续 limit 次触发熔断（无进展循环）"""
    state = {"fp": None, "n": 0}

    def bump(fp: str) -> int:
        state["n"] = state["n"] + 1 if fp == state["fp"] else 1  # 换指纹即重置——连续语义
        state["fp"] = fp
        return state["n"]
    return bump


def record(msg) -> dict:
    """中立轨迹记录：只留 role/content/工具名+参数，不带厂商私有字段——
    评估重放、跨厂商接管（原书实验 5-1）的原料"""
    if msg.type == "ai":
        return {"role": "assistant", "content": msg.content,
                "tool_calls": [{"name": c["name"], "args": c["args"]} for c in msg.tool_calls]}
    r = {"role": ROLE[msg.type], "content": msg.content}
    if msg.type == "tool":
        r["tool_call_id"] = msg.tool_call_id
    return r


def run_agent(model, task, ws: Path, max_turns: int = 8, token_budget: int = 30000):
    """主循环 v1：成对追加骨架 + 三道保险丝 + JSONL 轨迹"""
    by_name = {t.name: t for t in make_tools(ws)}
    llm = model.bind_tools(list(by_name.values()))
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=task)]
    trace, guard, spent = ws / "trace.jsonl", make_dup_guard(), 0

    def log_trace(msgs) -> None:
        """产生即落盘：决策/回执一旦 append 就写——崩溃后 log 尾部即断点"""
        with trace.open("a", encoding="utf-8") as f:
            for m in msgs:
                f.write(json.dumps(record(m), ensure_ascii=False) + "\n")

    for turn in range(1, max_turns + 1):            # 保险丝①：轮数上限
        resp = llm.invoke(messages)
        messages.append(resp)                        # 成对追加的前一半：决策记账
        log_trace([resp])                            # WAL：决策产生即落盘
        spent += (resp.usage_metadata or {}).get("total_tokens", 0)
        print(f"[轮 {turn}] {roles(messages)}｜本轮 {len(resp.tool_calls)} 个调用｜累计 {spent} token")
        if not resp.tool_calls:                      # 正常终止：模型决定收工
            return "完成", str(resp.content), messages
        if spent > token_budget:                     # 保险丝②：token 预算
            return "ABORT", f"token 预算用尽（{spent} > {token_budget}）", messages
        for call in resp.tool_calls:
            name, args = call["name"], call["args"]
            print(f"  [调用] {name}({str(args)[:60]})")
            if name not in by_name:                  # 幻觉调用：错误变输入（29.2 规约保留）
                result = f"[错误] 工具不存在：{name}——可用：{list(by_name)}"
            else:
                try:
                    result = str(by_name[name].invoke(args))
                except Exception as e:
                    result = f"[错误] {type(e).__name__}: {e}"
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))  # 后一半
            log_trace([messages[-1]])                # WAL：回执产生即落盘
            print(f"  [返回] {result[:70]}")
            fp = name + ":" + json.dumps(args, ensure_ascii=False, sort_keys=True)
            if guard(fp) >= 3:                       # 保险丝③：重复指纹
                return "ABORT", f"重复调用无进展：{name}（连续 3 次同参数）", messages
    return "ABORT", f"达到最大轮数 {max_turns}", messages


if __name__ == "__main__":
    shutil.rmtree(WS, ignore_errors=True)  # 重建工作区：收敛初始状态
    WS.mkdir(parents=True)
    (WS / "notes.md").write_text(NOTES, encoding="utf-8")
    (WS / "small.txt").write_text(SMALL, encoding="utf-8")
    model = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=60,
                       max_retries=3)  # 29.3 伏笔回收：SDK 内置指数退避——一级静默重试接进循环（首跑遇 503 过载实证）

    print("=== 保险丝③自检：重复指纹（模拟序列，不烧 API）===")
    check = make_dup_guard()
    for fp in ["read_file:a", "read_file:a", "read_file:a", "grep:x", "read_file:a", "read_file:a"]:
        n = check(fp)
        print(f"  {fp:<13} 第 {n} 次连续" + ("，熔断" if n >= 3 else ""))

    print(f"\n=== 真实任务：{TASK}")
    status, text, messages = run_agent(model, TASK, WS)
    print(f"\n[{status}] {text[:200]}")

    print("\n=== 轨迹持久化：trace.jsonl（中立格式，重放/接管的原料）===")
    lines = (WS / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    print(f"  共 {len(lines)} 行 = {len(messages) - 2} 条轮内新增消息（system/user 初始两条不重复落盘）")
    print(f"  首行样例：{json.dumps(json.loads(lines[0]), ensure_ascii=False)[:110]}")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（约 4-5 次真实调用）：
#   1. 重建工作区（notes.md 项目笔记 + small.txt 备忘）
#   2. 指纹自检：a,a,a → 第 3 次熔断；x 重置；a,a 只到 2 不熔断（连续语义证据）
#   3. 真实任务：预期轨迹 glob → read_file(notes.md)（可能还读 small.txt）→ 总结
#   4. 读回 trace.jsonl 展示中立轨迹
# 验证内容（模型路径可能不同，验收看闭环与产物）：
#   - 每轮打印的角色序列严格 append-only：只变长、前缀不变（D09 全量重发的活证据）
#   - assistant(tool_calls) 与 tool 结果在序列里成对出现（配对链完整）
#   - 总结内容来自 notes.md 真实文本（番茄钟/色温/v0.2 卡权限——不凭空猜测）
#   - trace.jsonl 行数 = 轮内新增消息数（含最终回答——首跑曾漏：轮末统一写入
#     被提前 return 跳过，改为 WAL 产生即落盘后修复）；首行是 assistant 决策
#   - 三道保险丝中轮数/预算真实任务不会触发（上限宽松）——它们的触发逻辑与
#     指纹同一模式，指纹自检通过即三者可信
#   - 若模型总结与文件不符（编造）：SYSTEM 已要求基于所读内容，重跑可复现
