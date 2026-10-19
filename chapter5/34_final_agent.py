"""chapter5/34_final_agent.py — mini Coding Agent 雏形：五天积木的合体

W5 收官（复盘主线：一棵树五根枝）——D29 骨架（工程化六阶段+四层故障+
分级恢复）、D30 心脏（主循环+保险丝+WAL）、D31/D32 手脚与免疫
（old→new 唯一匹配+持久 shell / 沙箱三道防线+LLM 语义审查）、D33 肌肉
（元能力六层+自举+三重保障）。本实验不新增概念——把五天组件接线成一台
端到端的机器：一句自然语言任务走完「理解→改代码→执行→验证→汇报」。

课件五环节自查清单的逐项接线：
    主循环   终止判定绑定验收信号：模型说「完成」后强制跑 test_stats.py
            （客观信号），不过则失败报告喂回继续修——不靠模型自觉
    上下文   稳定前缀（system 不动，保 KV 缓存）+ append-only 成对追加
    工具环   coding_tools 的结构化错误（edit_file 0/多命中、超时带死前输出）
    安全     bash 前置语义审查门（32.3 生产路线；fail-closed：解析失败当拒绝）
    元能力   启动自举 extract_json 注册进工具表（33.1），judge 当场用它解析

两个生产细节（本实验的真正新增量）：
    ① 验收资产由 Harness 持有——acceptance 每次重写 test_stats.py 再跑，
      模型篡改测试作弊的路径被物理切断（D31「守门员建在模型够不着的
      数据上」的延伸：验收信号同样不可被 Agent 污染）
    ② 修复循环 = 课件 run_task 伪代码形态：solve → 验收不过 → 失败报告
      喂回 → 再修（保险丝兜底：轮数 12 / 重复指纹连续 3 次）

长度说明：收官合体实验，用户授权侧重生产性——超出教学红线（~180 行），
README 记忆卡注明。

运行：uv run python -X utf8 chapter5/34_final_agent.py（需 .env，10-20 次真实调用，1-3 分钟）
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

WS = Path(__file__).parent / ".agent_ws7"

# ---- 任务资产：带 2 个 bug 的模块（模型要修的对象） ----
STATS_BUGGY = '''"""数据统计模块（现有 2 处 bug，验收测试不过）"""


def average(nums):
    return sum(nums) / (len(nums) + 1)      # bug1：分母多了 +1


def variance(nums):
    m = average(nums)
    return sum((x - m) ** 2 for x in nums)  # bug2：忘除以 n
'''

# ---- 验收资产（Harness 持有，不在模型手里——每次重写防篡改） ----
TEST_SRC = """from stats import average, variance

assert average([2, 4]) == 3.0, f"average 分母错：{average([2, 4])}"
assert variance([1, 2, 3]) == 2 / 3, f"variance 忘除 n：{variance([1, 2, 3])}"
assert average([5]) == 5.0
print("验收通过：3/3")
"""

TASK = "修复 stats.py，直到 test_stats.py 全部通过。"

SYSTEM = """你是工作区内的 Coding Agent。约定：改动用 edit_file（old_string 须与
原文逐字一致且唯一，read_file 看到的行号前缀不算原文）；运行 python 用 bash；
完成标准 = test_stats.py 全部通过（先跑它看失败，修完再跑确认）。"""

JUDGE = ("你是命令安全审查器，只输出 JSON："
         '{"verdict": "allow"或"deny", "reason": "一句话"}。命令：')

# ---- 元能力（33.1 复用）：启动自举 extract_json ----
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
    """自举：生成→沙箱验证→（失败重写一次）→wrap 注册（33.1 两轮循环）"""
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
            print(f"  自举第 {attempt} 版 3 用例全过 → extract_json 注册进工具表")
            fn = ns[BOOT_SPEC["name"]]

            def wrapper(text: str) -> dict:
                return fn(text)

            return StructuredTool.from_function(wrapper, name=BOOT_SPEC["name"],
                                                description=BOOT_SPEC["desc"])
        report = json.dumps(results, ensure_ascii=False)
    raise RuntimeError(f"自举两版均未通过：{report}")


# ---- 安全（32.3 接线）：bash 前置语义审查门，用自举的 extract_json 解析 ----
def judge(cmd: str, model, extract: BaseTool) -> bool:
    raw = model.invoke([HumanMessage(content=JUDGE + cmd)]).content
    try:
        v = extract.invoke({"text": raw})
        return str(v.get("verdict", "")).lower() == "allow"
    except Exception:
        return False  # fail-closed：解析失败当拒绝（生产原则）


# ---- 主循环（30.1 复用 + D34 核心：验收判停）----
def acceptance(ws: Path) -> tuple[bool, str]:
    """客观验收信号——Harness 持有测试资产：每次重写再跑，防模型篡改作弊"""
    (ws / "test_stats.py").write_text(TEST_SRC, encoding="utf-8")
    r = subprocess.run([sys.executable, "-X", "utf8", "test_stats.py"], cwd=ws,
                       capture_output=True, text=True, timeout=15, encoding="utf-8",
                       errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    tail = (r.stdout + r.stderr).strip().splitlines()
    return r.returncode == 0, tail[-1] if tail else "（无输出）"


def make_dup_guard(limit: int = 3):
    """重复指纹保险丝：连续 limit 次同样的工具签名 = 原地打转，熔断"""
    last, n = None, 0

    def hit(sigs):
        nonlocal last, n
        cur = "|".join(sigs)
        n = n + 1 if cur == last else 1
        last = cur
        return n >= limit
    return hit


def run_task(model, tools, ws: Path) -> tuple[int, str]:
    """课件 run_task 伪代码的落地：solve → 验收 → 不过喂回再修（保险丝兜底）"""
    by = {t.name: t for t in tools}
    trace = ws / "trace.jsonl"
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=TASK)]
    guard = make_dup_guard(3)
    verdict = "熔断：轮数上限"

    def wal(role: str, content: str):
        with trace.open("a", encoding="utf-8") as f:  # WAL：产生即落盘（30.1 教训）
            f.write(json.dumps({"role": role, "content": content[:300]}, ensure_ascii=False) + "\n")

    for turn in range(1, 13):  # 轮数保险丝
        resp = model.bind_tools(tools).invoke(messages)
        messages.append(resp)
        if not resp.tool_calls:
            ok, report = acceptance(ws)  # 模型说「完成」≠完成——强制验收
            if ok:
                return turn, report
            messages.append(HumanMessage(content=f"自检不能代替验收：{report}。"
                                                  "完成标准是测试全部通过——继续修复。"))
            wal("human", f"验收未通过：{report}")
            continue
        for tc in resp.tool_calls:
            args, name = tc["args"], tc["name"]
            if name == "bash" and not judge(args.get("command", ""), model, by["extract_json"]):
                out = "[拒绝] 命令未过语义审查门（32.3）——检查意图后换写法"
            else:
                out = str(by[name].invoke(args))
            messages.append(ToolMessage(content=out[:2000], tool_call_id=tc["id"]))
            wal(name, json.dumps(args, ensure_ascii=False))
            print(f"    [轮 {turn}] {name}({json.dumps(args, ensure_ascii=False)[:70]})")
        if guard([f"{tc['name']}:{tc['args']}" for tc in resp.tool_calls]):
            return turn, "熔断：重复指纹（连续 3 次同样调用——原地打转）"
    return 12, verdict


if __name__ == "__main__":
    import shutil
    shutil.rmtree(WS, ignore_errors=True)
    WS.mkdir(parents=True)
    (WS / "stats.py").write_text(STATS_BUGGY, encoding="utf-8")
    model = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL,
                       timeout=60, max_retries=3)

    ok0, rep0 = acceptance(WS)
    print(f"=== 任务：{TASK}（起始验收：{rep0}） ===")

    print("\n--- 启动自举（元能力环节）---")
    extract = bootstrap(model)

    print("\n--- 主循环：理解→改代码→执行→验证 ---")
    tools = make_tools(WS) + [extract]
    t0 = time.time()
    turn, report = run_task(model, tools, WS)
    n_trace = len((WS / "trace.jsonl").read_text(encoding="utf-8").splitlines())

    print(f"\n=== 汇报 ===")
    print(f"  验收：{report}｜轮数：{turn}｜轨迹：{n_trace} 条（WAL）｜耗时 {time.time() - t0:.0f}s")

    print("\n=== 五环节自查（课件清单逐项对照） ===")
    print("  [x] 主循环   终止判定绑定验收信号：模型说完成后强制跑 test_stats.py，不过则喂回再修")
    print("  [x] 上下文   稳定前缀（system 不动）+ append-only 成对追加 + WAL 落盘")
    print("  [x] 工具环   结构化错误：edit_file 0/多命中报错、超时带回死前输出（D32 升级）")
    print("  [x] 安全     bash 白名单 + 语义审查门（fail-closed）+ 路径锁 + 超时绳 + taskkill /T")
    print("  [x] 元能力   启动自举 extract_json 注册，judge 当场用它解析（能力自举服务安全）")
    print("  最薄弱一环（作业②，诚实记录）：审查门只挂 bash——py_exec 的代码、write_file")
    print("  /edit_file 的落盘内容同样可藏恶意动作；生产应全执行路径过审 + 源码只读挂载补丁流")

    print("\n=== 思考题 1（作业③）：安全与能力如何平衡 ===")
    print("  分场景配置而非全局开关：日常代码任务默认断网+源码只读已足够；需要联网装")
    print("  依赖时走出口白名单按需放行（32.1 的 pypi.org 清单）；涉及凭证的操作永远")
    print("  不进沙盒（挂载层即信任边界）。平衡点 = 按任务动态收缩的权限边界。")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（10-20 次真实 LLM 调用：自举 1-2 + 主循环每轮 1 + bash 每次审查 1）：
#   1. 预置 stats.py（2 bug）→ 起始 acceptance 打出失败（assert 报错行——任务动机）
#   2. bootstrap：自举 extract_json（3 用例沙箱验证 + 注册）
#   3. run_task：主循环修 bug——预期轨迹 read_file/grep → bash 跑测试看失败 →
#      edit_file ×2 → bash 复跑 → 声明完成 → acceptance 强制验收
#   4. 验收不过时失败报告作为 HumanMessage 喂回（repair 循环，课件伪代码形态）
#   5. 汇报 + 五环节自查 + 思考题答案
# 验证内容（模型输出动态，验收看确定性环节）：
#   - 起始验收必失败（bug 真实存在：average([2,4])==1.333≠3）
#   - 终态：report 含「验收通过：3/3」且 returncode==0——测试是客观信号，
#     模型自检不可替代；test_stats.py 由 acceptance 每次重写，篡改路径已断
#   - WAL trace.jsonl 行数 ≥ 轮数+1（产生即落盘的完整性证据）
#   - 若模型试图跑危险命令：审查门拒绝并把理由喂回（轨迹里可见 [拒绝]）
#   - 保险丝兜底：正常修复应在 12 轮内完成，指纹熔断不该触发（触发即异常轨迹）
