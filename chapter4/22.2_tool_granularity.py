"""chapter4/22.2_tool_granularity.py — ACI 原则一：通用执行器优先，专用工具有理由再上

本实验新增的主概念是「工具的粒度与形态决策」。承接 22.1 的五类地图：地图告诉你
有哪几类工具；本实验回答「每个工具该做多大」——ACI（Agent-Computer Interface，
给 Agent 用的接口设计，对应给人用的 HCI）三原则的第一条。

直觉：工具不是「一个 API 端点包一个工具」地照搬——那是早期形态，粒度过细，
Agent 要协调好几个工具才能干一件事；成熟思路是让工具对应 Agent 的目标。

原则一的三个层次（本文件三个演示的落点）：
    1. 默认上通用执行器：LLM 本身会写代码，沙箱 Python 解释器是「元能力」——
       一个 code_interpreter 顶几十个专用工具，还能处理没预料到的边缘场景
    2. 确需专用时粒度整合：extract_pdf_text / extract_docx_content /
       extract_pptx_content 合并成 read_document(file_type=...)——模型的
       认知负担从「3 个名字里挑 1 个」降为「就 1 个工具」；扩展新格式
       只需加一个选项，不是加一个新工具
    3. 四维决定何时退回专用：安全与权限 / 参数复杂度 / 变更频率 / 模型能力

注意：形态（做成工具还是 Skill）与披露（一次让模型看见几条）是两个独立
决策，互不绑定。Skill 明天 D23 展开，今天只留钩子。

运行：uv run python -X utf8 chapter4/22.2_tool_granularity.py（纯本地，不烧 API）
"""

import subprocess
import sys

# ---- 对照 A：粒度整合——同一能力，两种切法 ----

V1_SPECIFIC = ["extract_pdf_text", "extract_docx_content", "extract_pptx_content"]
V2_INTEGRATED = "read_document(file_type='pdf'|'docx'|'pptx')"


def show_granularity():
    print("对照 A：粒度整合（同一能力，两种切法）")
    print(f"  v1 一端点一工具：{', '.join(V1_SPECIFIC)}")
    print("     模型想读 docx，得先在 3 个名字里挑对那 1 个——认知负担，还容易挑错")
    print(f"  v2 整合为一把：{V2_INTEGRATED}")
    print("     一个名字一份 schema；明天要支持 xlsx？file_type 加个选项，注册表不膨胀")


# ---- 对照 B：通用执行器——一个 Python 沙箱顶 N 个专用工具 ----

UNEXPECTED_TASKS = [  # 三个「没预料到的任务」，谁也不会提前为它们造专用工具
    ("2026-10-01 到 10-07 假期共几天？",
     "from datetime import date; print((date(2026,10,7) - date(2026,10,1)).days + 1)"),
    ("[88, 42, 95, 60] 的中位数是多少？",
     "import statistics; print(statistics.median([88, 42, 95, 60]))"),
    ("38.5°C 换算成华氏度是多少？",
     "print(round(38.5 * 9 / 5 + 32, 1))"),
]


def code_interpreter(code: str) -> str:
    """通用执行器：模型写代码，工具只管执行并交回 stdout——LLM 的「元能力」。
    生产形状是容器级隔离（gVisor/Firecracker）+ 资源限额，教学演示到子进程一层"""
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=10,
    )
    return proc.stdout.strip() or proc.stderr.strip()[:120]


def show_generic_executor():
    print("对照 B：通用执行器——三个『没预料到的任务』，同一个工具全接住")
    for task, code in UNEXPECTED_TASKS:
        print(f"  任务：{task}")
        print(f"    → code_interpreter 执行模型写的代码 → {code_interpreter(code)}")
    print("  若各造专用工具：假期天数计算器 / 中位数器 / 温度换算器……注册表无限膨胀；")
    print("  通用执行器一个顶 N 个，还能接住未来第 N+1 个没预料到的任务")


# ---- 四维决策表：何时退回专用工具（或改用 Skill） ----

FOUR_DIMENSIONS = [
    ("安全与权限", "生产 DB 写操作：要精细授权 + 审计留痕", "只读、低危操作"),
    ("参数复杂度", "嵌套对象、多字段联合校验：schema 引导更稳", "简单标量参数"),
    ("变更频率", "稳定的核心能力", "常变的能力（Skill 改文字即可，D23 预告）"),
    ("模型能力", "弱模型：需要 schema 手把手引导", "强模型：Skill + 通用执行器表达力更强"),
]


def show_dimensions():
    print("四维决策：何时退回专用工具（或改用 Skill）")
    for dim, specific, generic in FOUR_DIMENSIONS:
        print(f"  {dim}")
        print(f"    偏专用 ← {specific}")
        print(f"    偏通用/Skill ← {generic}")
    print("  另注意：形态（工具 vs Skill）与披露（一次让模型看见几条）是两个独立决策")


if __name__ == "__main__":
    show_granularity()
    print()
    show_generic_executor()
    print("\n停一停：为什么说 code_interpreter 是「元能力」？")
    print("  它不是某个具体功能，而是「生成能力的能力」——LLM 会写代码，而代码")
    print("  能表达几乎任意计算，所以一个解释器 ≈ 任意专用工具的并集。")
    print()
    show_dimensions()

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（纯本地，不烧 API）：
#   1. show_granularity：粒度对照——3 个专用提取工具 vs 1 个 read_document
#   2. show_generic_executor：三个没预料到的任务（假期天数/中位数/温度换算）
#      全部由同一个 code_interpreter 真实执行（子进程跑 Python 拿 stdout）
#   3. show_dimensions：四维决策表（安全/复杂度/频率/模型能力）+ 形态≠披露
# 验证内容：
#   - 对照 A：v1 要 3 选 1、v2 一个名字一份 schema，扩展成本一目了然
#   - 对照 B：三次执行返回 7（含首尾）/ 74.0 / 101.3——同一工具接住三类
#     没预料到的计算，这就是「元能力」的实证
#   - 检查点：元能力 = 生成能力的能力（LLM 写代码 + 代码表达任意计算）
