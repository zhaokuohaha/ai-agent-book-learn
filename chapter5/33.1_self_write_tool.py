"""chapter5/33.1_self_write_tool.py — 自写工具并注册：生成→验证→注册→使用

本实验新增的主概念是「元能力」（meta-capability）——能创造其他能力的能力：
Agent 用代码当场写出新工具，不必事先预制所有能力。后端类比：普通能力像
已部署的微服务接口，元能力像平台的动态插件机制——需求来了现场编译加载
新 handler，不等下个版本排期。这是元能力六层同心圆的最外层：Agent 自举。

自举目标选 extract_json（从杂乱文本剥出 JSON）——不是随便挑的：32.3 的
llm_scan 刚手写 re.search 剥 deepseek 输出的 ```json 包裹，这是主循环/
审查器真实反复需要的工具。Agent 为自己的下一环造工具，使用环直接拿它
解析一次真实 LLM 输出——闭环自指。

闭环四步（课件作业骨架的落地）：
    ① 生成   LLM 按 SPEC 写函数（签名 + 用例要求）
    ② 验证   沙箱 subprocess 跑 3 用例（隔离 + 超时绳 + UNBUFFERED，
             D32 的绳；失败报告结构化：用例名/期望/实际/报错）
    ③ 重写   失败带报错喂回 LLM 再来一次（D29.2 测试-修复循环复用，
             课件骨架就是一次重试）
    ④ 注册   通过后 wrap 成 StructuredTool 入表——注册的不是裸代码而是
             受限执行体：能力来自自举，边界仍归 Harness（教学版 = exec
             隔离命名空间；生产 = 进程级沙箱 wrap（D32 三道防线），注明差距）

运行：uv run python -X utf8 chapter5/33.1_self_write_tool.py（需 .env，2-3 次真实调用）
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL

WS = Path(__file__).parent / ".agent_ws6"

SPEC = {  # 工具规格：签名、描述、测试用例（教学版预置，保证确定性验收）
    "name": "extract_json",
    "sig": "extract_json(text: str) -> dict",
    "desc": "从杂乱文本中提取第一个 JSON 对象并解析为 dict——常见于 LLM 输出带 ```json 包裹或前后杂文",
    "tests": [
        ("纯 JSON", '{"a": 1}', {"a": 1}),
        ("markdown 包裹", '结果：\n```json\n{"verdict": "deny"}\n```\n完毕', {"verdict": "deny"}),
        ("前后杂文", '答案是 {"x": [1, 2], "y": "z"} 谢谢', {"x": [1, 2], "y": "z"}),
    ],
}

GEN_PROMPT = """写一个纯 Python 函数 {sig}。要求：只依赖标准库 re/json；
{desc}。只输出函数代码本身——不要 markdown 代码块包裹、不要解释、不要调用示例。"""


def run_tests(code: str) -> list[dict]:
    """沙箱验证：生成代码 + 用例写临时文件，subprocess 独立进程跑（隔离+绳）"""
    cases = "\n".join(
        f'''try:
    ok = extract_json({inp!r}) == {exp!r}
    results.append({{"case": {name!r}, "ok": ok, "got": repr(extract_json({inp!r}))}})
except Exception as e:
    results.append({{"case": {name!r}, "ok": False, "error": repr(e)}})'''
        for name, inp, exp in SPEC["tests"])
    test_py = (f"{code}\n\nimport json\nresults = []\n{cases}\n"
               'print("__REPORT__" + json.dumps(results, ensure_ascii=False))')
    (WS / "self_tool_test.py").write_text(test_py, encoding="utf-8")
    r = subprocess.run([sys.executable, "-X", "utf8", WS / "self_tool_test.py"],
                       capture_output=True, text=True, timeout=15, encoding="utf-8",
                       errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    for ln in r.stdout.splitlines():
        if ln.startswith("__REPORT__"):
            return json.loads(ln[len("__REPORT__"):])
    return [{"case": "进程级失败", "ok": False,
             "error": (r.stderr.strip().splitlines() or ["无输出崩溃"])[-1]}]


def self_write_tool(model) -> BaseTool:
    """自举主流程：生成 → 沙箱验证 → 失败带报错重写一次 → wrap 注册"""
    report = ""
    for attempt in (1, 2):
        if attempt == 1:
            prompt = GEN_PROMPT.format(sig=SPEC["sig"], desc=SPEC["desc"])
        else:
            prompt = (f"你写的函数未通过测试。失败报告：\n{report}\n"
                      f"请修复后重写整个函数（{SPEC['sig']}），仍只输出函数代码。")
        code = model.invoke([HumanMessage(content=prompt)]).content
        code = code.removeprefix("```python").removeprefix("```").removesuffix("```").strip()
        results = run_tests(code)
        print(f"  第 {attempt} 版沙箱验证：" + "；".join(
            f"{r['case']}→{'过' if r['ok'] else '败(' + str(r.get('got') or r.get('error')) + ')'}"
            for r in results))
        if all(r["ok"] for r in results):
            ns: dict = {}  # wrap：隔离命名空间 exec——注册受限执行体而非裸代码
            exec(compile(code, "<self_written>", "exec"), ns)  # noqa: S102 教学简化：生产用进程级沙箱
            fn = ns[SPEC["name"]]

            def wrapper(text: str) -> dict:  # @tool 需完整类型标注（schema 依赖）
                return fn(text)

            print(f"  注册：{SPEC['name']} → 工具表（wrap 受限执行体）")
            return StructuredTool.from_function(wrapper, name=SPEC["name"], description=SPEC["desc"])
        report = json.dumps(results, ensure_ascii=False)
    raise RuntimeError(f"两版均未通过：{report}")


if __name__ == "__main__":
    WS.mkdir(parents=True, exist_ok=True)
    model = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL,
                       timeout=60, max_retries=3)
    print("=== ①②③ 生成 → 沙箱验证 →（失败重写）→ 注册 ===")
    tool = self_write_tool(model)

    print("\n=== ④ 使用：解析一次真实 LLM 输出（闭环自指） ===")
    judge = ("你是命令安全审查器，只输出 JSON："
             '{"verdict": "allow"或"deny", "reason": "一句话"}。命令：find /logs -exec rm {} \\;')
    raw = model.invoke([HumanMessage(content=judge)]).content
    print(f"  LLM 原始输出（注意包裹形态）：{raw[:80]}…")
    parsed = tool.invoke({"text": raw})
    print(f"  extract_json 解析 → verdict={parsed['verdict']!r}——主循环/审查器的下一环，"
          f"从此不用手写剥壳")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（2-3 次真实 LLM 调用）：
#   1. self_write_tool：第 1 版生成（按 SPEC 签名+描述）→ run_tests 沙箱跑 3 用例
#   2. 全过 → exec 隔离命名空间取出函数 → StructuredTool 注册；有败 → 失败
#      报告（用例名/got/error）喂回 LLM 重写一次（D29.2 循环复用）
#   3. 使用环：JUDGE prompt 拿一次真实 LLM 输出（大概率 ```json 包裹）→
#      用刚注册的 extract_json 解析
# 验证内容（模型输出动态，验收看闭环轨迹与确定性环节）：
#   - 沙箱验证环节是确定性的：3 用例逐条打印 过/败（含 got/error）
#   - 注册后 tool.invoke 是确定性的：parse 结果即函数行为
#   - 使用环：LLM 原始输出无论什么包裹形态，extract_json 都应解析出
#     verdict 字段——「Agent 为自己造的工具，下一环立刻用上」的闭环证据
#   - 与生产的差距（注明）：wrap 用进程内 exec 隔离命名空间——生产是进程
#     级/容器沙箱执行体（D32 三道防线 + 32.3 LLM 审查生成的代码本身）；
#     重试次数 1 次（课件骨架）；SPEC 预置而非模型自主提出（下一台阶）
