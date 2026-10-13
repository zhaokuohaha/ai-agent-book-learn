"""chapter4/28_final_agent.py — W4 收官：玩具 Agent 合体（≥3 个真实工具）

第 4 章的知识在这里合体——一个玩具 Agent 接入三个真实工具（一读一写一执行）
跑完整任务。知识点地图（每行对应代码落点）：
    D09 心跳    手写 while + bind_tools + ToolMessage 回填——循环骨架
    22.1 定位   感知（read_file）/ 执行（write_file）/ 通用执行器（run_command）
    22.2 元能力 run_command 不写专用工具，让模型写脚本——代码编排的雏形
    22.3 描述   三个 docstring = W4 自查清单模板（何时用/边界/例子/返回值/调用）
    24.x 披露   工具仅 3 个→全量注入即正解、顺序稳定不重排（KV 保命）；
                三层按需披露与主动发现是 100+ 工具时的解法，这里不做
    25.x 截断   read_file 显式截断 + offset 续读——绝不静默、只给把手
    26.x 安全   内嵌工具实现而非循环：run_command = 26.1 拆链白名单 +
                26.2 圈绳 + UTF-8 强制；read/write = 路径锁
    27.x 协作   单任务单 Agent 足够——spawn/cancel 留给多任务/长任务场景
    感知闭环    任务要求「写完读回核对」——执行后再感知，自我校验

任务链（中间计算留在执行环境，只有结论回上下文）：
    读 sales.txt → 写 calc.py → 跑脚本 → 写 report.md → 读回核对

运行：uv run python -X utf8 chapter4/28_final_agent.py（需 .env，约 6 次调用）
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL

WS = Path(__file__).parent / ".agent_ws"  # 工作区 = 工具的合法活动半径
LIMIT = 4000  # read_file 单次返回上限——超出走显式截断（D25 三板斧之三）
ALLOWED = {"python", "dir"}  # run_command 白名单：只给任务真正需要的能力
SEGMENT = re.compile(r"&&?|\|\|?")

SALES = """2025-01-05 华东 128000
2025-01-12 华南 96500
2025-01-19 华北 154300
2025-01-26 西南 78200
2025-02-02 华东 142100
2025-02-09 华南 112800
2025-02-16 华北 98700
2025-02-23 西南 135500
"""

TASK = ("分析工作区里 sales.txt 的销售记录：计算销售总额、平均值、最大单笔，"
        "把结论（含一个 markdown 表格）写入 report.md，最后读回 report.md 核对三项指标齐全。")

LOG = "".join(f"2025-10-13 12:{i // 60:02d}:{i % 60:02d} [INFO] 心跳正常 seq={i:03d}\n" for i in range(150))


def _resolve(path: str):
    """路径锁：解析后必须落在工作区内，否则返回 None"""
    p = (WS / path).resolve()
    return p if WS.resolve() in p.parents else None


@tool
def read_file(path: str, offset: int = 0) -> str:
    """当需要查看工作区文件内容时使用，例如读取数据文件做分析。
    只能读取工作区内的文本文件；单次最多返回 4000 字符——更长的从 offset 处
    续读，返回值末尾会写明「完整多少字符、下次 offset 用多少」，绝不静默截断。
    返回值：内容片段；文件不存在时报 [错误]（不抛异常）。
    例如：read_file("sales.txt") / read_file("system.log", offset=4000)"""
    p = _resolve(path)
    if p is None:
        return "[错误] 路径必须在工作区内"
    if not p.is_file():
        return f"[错误] 文件不存在：{path}"
    text = p.read_text(encoding="utf-8")
    part = text[offset:]
    if len(part) <= LIMIT:
        return part
    return part[:LIMIT] + f'\n...[已截断：完整 {len(text)} 字符，续读 read_file("{path}", offset={offset + LIMIT})]'


@tool
def write_file(path: str, content: str) -> str:
    """当需要保存结论或脚本到文件时使用，例如保存分析报告、保存计算脚本。
    只能在工作区内创建/覆盖文本文件；不追加、不移动——覆盖即替换全文。
    返回值：成功返回「已写入 路径（N 字节）」；路径越界时返回 [错误]。
    例如：write_file("calc.py", "print(1 + 2)")"""
    p = _resolve(path)
    if p is None:
        return "[错误] 路径必须在工作区内"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"已写入 {path}（{len(content.encode('utf-8'))} 字节）"


@tool
def run_command(command: str) -> str:
    """当需要执行工作区内的命令时使用，典型场景：运行你写好的计算脚本。
    只执行白名单命令（python / dir），链式命令逐段校验；cwd 锁定工作区、
    超时 15 秒——要跑计算请先用 write_file 写脚本，再 python 脚本名 运行。
    返回值：命令输出（最多 2000 字符，含错误输出）；被拒或超时返回原因。
    例如：run_command("python calc.py")"""
    for seg in SEGMENT.split(command):
        parts = seg.strip().split()
        if parts and parts[0].lower() not in ALLOWED:
            return f"[拒绝] 链中命令不在白名单：{parts[0]!r}"
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True,
                           cwd=WS, timeout=15, encoding="utf-8", errors="replace",
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    except subprocess.TimeoutExpired:
        return "[超时] 命令超过 15 秒，已终止"
    out = (r.stdout + r.stderr).strip()
    return out[:2000] if out else "（无输出）"


TOOLS = [read_file, write_file, run_command]
BY_NAME = {t.name: t for t in TOOLS}


def run_agent(model: ChatOpenAI, max_turns: int = 8) -> None:
    """合体循环：保持干净——安全的闸门在工具实现里，不在这里"""
    llm = model.bind_tools(TOOLS)
    messages = [HumanMessage(content=TASK)]
    for _ in range(max_turns):
        resp = llm.invoke(messages)
        messages.append(resp)
        if not resp.tool_calls:
            print(f"\n[完成] {str(resp.content)[:200]}")
            return
        for call in resp.tool_calls:
            print(f"  [调用] {call['name']}({json.dumps(call['args'], ensure_ascii=False)[:80]})")
            result = str(BY_NAME[call["name"]].invoke(call["args"]))
            print(f"  [返回] {result[:90]}")
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    print("  [warn] 轮数用尽")


if __name__ == "__main__":
    model = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0)

    shutil.rmtree(WS, ignore_errors=True)  # 重建工作区：收敛初始状态（22.4 首跑教训）
    WS.mkdir(parents=True)
    (WS / "sales.txt").write_text(SALES, encoding="utf-8")
    (WS / "system.log").write_text(LOG, encoding="utf-8")

    print("=== 工具登记：描述按 W4 自查清单写满（对照每项的落点）===")
    for t in TOOLS:
        desc = t.description.strip().replace("\n", "\n      ")
        print(f"\n· {t.name}\n      {desc}")

    print("\n=== 工具抽查：大文件读取——显式截断 + offset 续读（D25）===")
    head = read_file.invoke({"path": "system.log"})
    print(f"  头部：{head.splitlines()[0]}")
    print(f"  （尾部）…{head[-90:]}")
    tail2 = read_file.invoke({"path": "system.log", "offset": LIMIT})
    print(f"  续读 offset={LIMIT}：{tail2[:36]}…（到尾不再截断）")

    print(f"\n=== 任务：{TASK}")
    run_agent(model)

    print("\n=== 产物（真实落盘）===")
    report = WS / "report.md"
    print(report.read_text(encoding="utf-8") if report.exists() else "[report.md 未生成]")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（约 6 次真实调用）：
#   1. 重建工作区（sales.txt + system.log）；打印三工具描述（自查清单五点对照）
#   2. 工具抽查：read_file("system.log") 显式截断 → offset=4000 续读到尾
#   3. Agent 轨迹预期：read_file(sales.txt) → write_file(calc.py) →
#      run_command("python calc.py") → write_file(report.md) → read_file(report.md)
#      读回核对 → 自然语言汇报
#   4. 读回 report.md 展示——产物真实落盘
# 验证内容（模型路径可能不同，验收看结构与产物）：
#   - 抽查段：头部内容 + 尾部「已截断：完整 N 字符，续读 offset=…」；续读到尾不再截断
#   - 三个工具都被调用；轨迹含「写报告后再 read_file 核对」的感知闭环
#   - report.md 真实生成，含表格：总额 946100、均值 118262.5、最大 154300
#   - run_command 实际执行真 python：失败时看 [返回] 报错（编码/引号已兜底）
#   - 若模型心算不写脚本：任务已明示「用 python 计算」，重跑可复现（22.3 方法论）
#   - 彩蛋：这条链就是第 5 章「代码编排」的雏形——中间计算留在执行环境，
#     只有最终结论回到上下文
