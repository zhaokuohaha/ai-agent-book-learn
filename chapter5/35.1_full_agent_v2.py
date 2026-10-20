"""chapter5/35.1_full_agent_v2.py — mini Coding Agent 完整版（v2）：从需求到交付

D34 的 final_agent 还不算完整的 Coding Agent：任务是「修 2 个预置 bug」
而非「从需求写代码」；轨迹是简版 WAL（截断 300 字、只存工具调用漏
assistant 文本）——出问题回放不了。本实验按 D35 项目检查点升级：

    ① 任务升级：一句自然语言需求 → 从零实现 textstat 模块（word_freq +
       top_k）→ 验收测试通过 → 汇报——「理解→设计→实现→自测→汇报」的
       工程化六阶段微缩走全（SYSTEM 引导先出 2-3 行设计再实现——设计
       文本留在上下文里自锚定，D29「设计文档是人类最佳介入点」的最小形态）
    ② 轨迹升级：完整事件流落盘（ts/turn/kind/name/content，不截断）——
       assistant 的推理文本也存。D35 第一性原理：模型只对它看到的上下文
       负责，任何没进入上下文（和轨迹）的事实对模型（和你）都不存在；
       调试 Agent 第一步不是改提示词，是把当时的完整上下文打印出来
    ③ 五类体检（35.2 系统输出）：v2 对照 D35 清单逐条过检

五环节（D34 接线全保留）：验收判停（Harness 持有测试资产防篡改）/
稳定前缀+成对追加/结构化错误/bash 语义审查门（fail-closed）/启动自举
extract_json 注册。

运行：uv run python -X utf8 chapter5/35.1_full_agent_v2.py（需 .env，10-20 次真实调用，1-3 分钟）
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI

from common.coding_tools import make_tools
from common.config import API_KEY, BASE_URL, MODEL

WS = Path(__file__).parent / ".agent_ws8"

# ---- 验收资产（Harness 持有，每次重写防篡改——D34 生产细节延续） ----
TEST_SRC = """from textstat import word_freq, top_k

counts = word_freq("The quick brown fox jumps over the lazy dog the FOX",
                   stopwords={"the"})
assert counts == {"quick": 1, "brown": 1, "fox": 2, "jumps": 1, "over": 1,
                  "lazy": 1, "dog": 1}, f"word_freq 错：{counts}"
assert top_k(counts, 1) == [("fox", 2)], f"top_k 错：{top_k(counts, 1)}"
print("验收通过：2/2")
"""

TASK = ("实现 textstat.py：word_freq(text, stopwords) 返回词频 dict（词统一小写、"
        "按空白切分、滤掉 stopwords 里的词）；top_k(counts, k) 返回前 k 个"
        "(词, 次数) 元组列表，按次数降序。先用 2-3 行说明你的设计，再实现，"
        "直到验收测试 test_stats.py 全部通过。")

SYSTEM = """你是工作区内的 Coding Agent。约定：新文件用 write_file、改动用 edit_file
（old_string 须与原文逐字一致且唯一）；运行 python 用 bash；先简述设计（2-3 行）
再动手——设计留在对话里作为实现的对账基准；完成标准 = test_stats.py 全部通过
（实现完先跑它，失败就修，修完再跑确认）。"""

JUDGE = ("你是命令安全审查器，只输出 JSON："
         '{"verdict": "allow"或"deny", "reason": "一句话"}。命令：')

BOOT_SPEC = {
    "name": "extract_json",
    "sig": "extract_json(text: str) -> dict",
    "desc": "从杂乱文本提取第一个 JSON 对象——LLM 输出常带 ```json 包裹或前后杂文",
    "tests": [("纯 JSON", '{"a": 1}', {"a": 1}),
              ("markdown 包裹", '```json\\n{"v": 2}\\n```', {"v": 2}),
              ("前后杂文", '答 {"x": 1} 完', {"x": 1})],
}


def boot_run_tests(code: str) -> list[dict]:
    cases = "\n".join(
        f'''try:
    results.append({{"case": {n!r}, "ok": extract_json({i!r}) == {e!r}}})
except Exception as ex:
    results.append({{"case": {n!r}, "ok": False, "error": repr(ex)}})'''
        for n, i, e in BOOT_SPEC["tests"])
    (WS / "_boot_test.py").write_text(
        f"{code}\nimport json\nresults = []\n{cases}\n"
        'print("__R__" + json.dumps(results, ensure_ascii=False))', encoding="utf-8")
    r = subprocess.run([sys.executable, "-X", "utf8", WS / "_boot_test.py"],
                       capture_output=True, text=True, timeout=15, encoding="utf-8",
                       errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    for ln in r.stdout.splitlines():
        if ln.startswith("__R__"):
            return json.loads(ln[5:])
    return [{"case": "崩溃", "ok": False, "error": r.stderr[-200:]}]


def bootstrap(model) -> BaseTool:
    """元能力：生成→沙箱验证→（失败重写一次）→wrap 注册（D33 闭环）"""
    report = ""
    for attempt in (1, 2):
        prompt = (f"写纯 Python 函数 {BOOT_SPEC['sig']}，只依赖标准库 re/json。"
                  f"{BOOT_SPEC['desc']}。只输出函数代码，无 markdown 包裹无解释。"
                  + (f"\n上一版失败报告：{report}\n请修复后重写。" if attempt == 2 else ""))
        code = model.invoke([HumanMessage(content=prompt)]).content
        code = code.removeprefix("```python").removeprefix("```").removesuffix("```").strip()
        results = boot_run_tests(code)
        if all(r["ok"] for r in results):
            ns: dict = {}
            exec(compile(code, "<self_written>", "exec"), ns)  # noqa: S102 教学简化：生产进程级沙箱
            print(f"  自举第 {attempt} 版 3 用例全过 → extract_json 注册")
            fn = ns[BOOT_SPEC["name"]]

            def wrapper(text: str) -> dict:
                return fn(text)

            return StructuredTool.from_function(wrapper, name=BOOT_SPEC["name"],
                                                description=BOOT_SPEC["desc"])
        report = json.dumps(results, ensure_ascii=False)
    raise RuntimeError(f"自举两版均未通过：{report}")


def judge(cmd: str, model, extract: BaseTool) -> bool:
    """bash 前置语义审查门（32.3），fail-closed：解析失败当拒绝"""
    raw = model.invoke([HumanMessage(content=JUDGE + cmd)]).content
    try:
        v = extract.invoke({"text": raw})
        return str(v.get("verdict", "")).lower() == "allow"
    except Exception:
        return False


def acceptance(ws: Path) -> tuple[bool, str]:
    """客观验收信号——Harness 持有测试资产：每次重写再跑，防篡改"""
    (ws / "test_stats.py").write_text(TEST_SRC, encoding="utf-8")
    r = subprocess.run([sys.executable, "-X", "utf8", "test_stats.py"], cwd=ws,
                       capture_output=True, text=True, timeout=15, encoding="utf-8",
                       errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    tail = (r.stdout + r.stderr).strip().splitlines()
    return r.returncode == 0, tail[-1] if tail else "（无输出）"


def make_dup_guard(limit: int = 3):
    last, n = None, 0

    def hit(sigs):
        nonlocal last, n
        cur = "|".join(sigs)
        n = n + 1 if cur == last else 1
        last = cur
        return n >= limit
    return hit


def run_task(model, tools, ws: Path) -> tuple[int, str]:
    by = {t.name: t for t in tools}
    trace = ws / "trace.jsonl"
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=TASK)]
    guard = make_dup_guard(3)

    def wal(events: list[dict]):
        """完整事件流落盘（D35 升级）：不截断、assistant 文本也存——
        回放时看得到「模型当时看到了什么、说了什么」"""
        with trace.open("a", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps({**ev, "ts": round(time.time(), 2)}, ensure_ascii=False) + "\n")

    for turn in range(1, 13):
        resp = model.bind_tools(tools).invoke(messages)
        messages.append(resp)
        wal([{"turn": turn, "kind": "assistant", "content": resp.content,
              "tool_calls": [{"name": tc["name"], "args": tc["args"]} for tc in resp.tool_calls]}])
        if not resp.tool_calls:
            ok, report = acceptance(ws)
            if ok:
                return turn, report
            messages.append(HumanMessage(content=f"自检不能代替验收：{report}。"
                                                  "完成标准是测试全部通过——继续修复。"))
            wal([{"turn": turn, "kind": "human", "content": f"验收未通过：{report}"}])
            continue
        events = []
        for tc in resp.tool_calls:
            args, name = tc["args"], tc["name"]
            if name == "bash" and not judge(args.get("command", ""), model, by["extract_json"]):
                out = "[拒绝] 命令未过语义审查门——检查意图后换写法"
            else:
                out = str(by[name].invoke(args))
            messages.append(ToolMessage(content=out[:2000], tool_call_id=tc["id"]))
            events.append({"turn": turn, "kind": "tool", "name": name,
                           "content": json.dumps(args, ensure_ascii=False) + " → " + out[:400]})
            print(f"    [轮 {turn}] {name}({json.dumps(args, ensure_ascii=False)[:60]})")
        wal(events)
        if guard([f"{tc['name']}:{tc['args']}" for tc in resp.tool_calls]):
            return turn, "熔断：重复指纹（连续 3 次同样调用——原地打转）"
    return 12, "熔断：轮数上限"


if __name__ == "__main__":
    import shutil
    shutil.rmtree(WS, ignore_errors=True)
    WS.mkdir(parents=True)
    model = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL,
                       timeout=60, max_retries=3)
    ok0, rep0 = acceptance(WS)
    print(f"=== 任务：{TASK[:50]}…（起始验收：{rep0}） ===")
    print("\n--- 启动自举（元能力）---")
    extract = bootstrap(model)
    print("\n--- 主循环：理解→设计→实现→自测 ---")
    tools = make_tools(WS) + [extract]
    t0 = time.time()
    turn, report = run_task(model, tools, WS)
    n = len((WS / "trace.jsonl").read_text(encoding="utf-8").splitlines())
    print(f"\n=== 汇报 ===\n  验收：{report}｜轮数：{turn}｜轨迹事件：{n} 条｜耗时 {time.time() - t0:.0f}s")
    design = next((json.loads(ln)["content"] for ln in (WS / "trace.jsonl").read_text(
        encoding="utf-8").splitlines()
        if json.loads(ln).get("kind") == "assistant" and json.loads(ln)["content"].strip()), "")
    # 真实观察：SYSTEM 引导「先 2-3 行设计」，模型却把设计写进了产物的
    # docstring——对话里只有开场白。设计没成为上下文锚点（D29 设计介入点的
    # 反面观察，诚实记录在 README 体验段；本行验证的是 assistant 文本已入轨迹）
    print(f"  轨迹含 assistant 文本（D34 简版 WAL 漏存的——轮 1 开场）：{design[:120]}")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（10-20 次真实 LLM 调用）：
#   1. 起始 acceptance 失败（textstat 不存在——ImportError，任务动机）
#   2. bootstrap 自举 extract_json → 注册
#   3. run_task：预期轨迹——轮 1 assistant 输出 2-3 行设计（六阶段微缩的
#      设计阶段，SYSTEM 引导）→ write_file 实现 → bash 跑测试 → （若挂）
#      read/edit 修复 → 声明完成 → acceptance 强制验收
#   4. 汇报 + 从轨迹提取轮 1 设计文本（设计自锚定的证据）
# 验证内容（模型输出动态，验收看确定性环节）：
#   - 起始验收必失败（模块不存在）；终态 report 含「验收通过：2/2」
#   - test_stats.py 由 acceptance 每次重写——篡改路径已断（D34 延续）
#   - trace.jsonl 每轮含 assistant 事件（content=模型文本+tool_calls）——
#     D34 简版 WAL 漏的就是这块：回放看得到模型说了什么（35.2 用它回放）
#   - 轮 1 assistant 文本非空且入轨迹（D34 漏存项修复的证据）；真实观察：
#     SYSTEM 引导「先 2-3 行设计」，模型把设计写进了产物的 docstring 而
#     非对话——设计未成上下文锚点（D29 设计介入点的反面观察，README 记录）
#   - 保险丝兜底：12 轮内应完成；指纹熔断不触发（触发即异常轨迹）
