"""chapter5/29.1_seven_tools.py — Coding Agent 七件套：通用 Agent 的全套手脚

第 5 章开章：通用 Agent 的核心是一个 Coding Agent + 文件系统（从 Manus 到
OpenClaw 都是这一范式）——因为几乎所有高效的内容生成都落到代码上：PPT/Word
本质是 OOXML、PDF 可由 Markdown/LaTeX 生成、数据分析交给 pandas。代码是效率
最高、成本最低、可复用性最强的能力基座（元能力：能当场造出新工具的能力）。

本实验新增的主概念是「七件套」——D28 的 3 工具（read/write/bash）配齐到
Coding Agent 的标准配置 7 件：
    感知三件（只读）：read_file（行号标注）/ glob_search（按文件名找）/ grep_search（按内容找）
    执行四件（改世界）：write_file / edit_file / bash / py_exec（沙盒解释器）
理论上一个 bash 就够（OpenAI Codex 只给 bash 一个工具），拆成七件是为了调用
更精准——模型说「找文件」就直接 glob，不用 bash dir 再解析输出（D24 回响：
七个工具全量注入远够不到规模问题门槛，描述清楚比数量少更重要）。

伴生概念（五个实现技巧今日落三，其余两个是 D30 主菜）：
    ① 行号标注读——read 返回「4→code」，模型能精确引用第几行，edit 更可靠
    ② old→new 精确编辑——Claude Code/Codex/Cursor 现行方案：old_string 与文件
       逐字符一致且唯一才改，不唯一/找不到就报错——绝不模糊匹配（22.4 回响）
    ③ 写后即时语法检查——write/edit 一完成立即 py_compile，语法错误当刻可见
       （IDE 红线思维），不用等跑测试才发现
    未落：并行调用+流式执行（D30 工具环）、状态栏动态注入+持久终端（D30 上下文管理）

七件套实现沉淀在 common/coding_tools.py（先例：common/memory_store.py），
29.2 的 agent 循环与 D30 主循环 v1 都从这里 import。本实验给七件套做体检：
感知三件先看清楚，执行四件再动手——正是一个 Coding Agent 微缩工作日。

运行：uv run python -X utf8 chapter5/29.1_seven_tools.py（无需 .env——只看工具行为）
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.coding_tools import make_tools

WS = Path(__file__).parent / ".agent_ws"  # 工作区：七件套的合法活动半径

APP = '''def average(nums):
    return sum(nums) / (len(nums) + 1)  # TODO: 返回值总偏小——检查分母


def handle(request):
    return f"ok: {request}"
'''

UTILS = '''def clamp(v, lo, hi):
    return max(lo, min(v, hi))
'''

BAD_GEN = 'print("TODO 清单已生成"'  # 缺右括号——写完瞬间就该被语法检查抓住


def show(call_desc: str, result: str, head: int = 12) -> None:
    """统一打印：调用 → 返回（截长保可读）"""
    lines = result.splitlines()
    body = "\n    ".join(lines[:head])
    more = f"\n    …（共 {len(lines)} 行）" if len(lines) > head else ""
    print(f"→ {call_desc}\n    {body}{more}")


if __name__ == "__main__":
    shutil.rmtree(WS, ignore_errors=True)  # 重建工作区：收敛初始状态（22.4 首跑教训）
    WS.mkdir(parents=True)
    (WS / "app.py").write_text(APP, encoding="utf-8")
    (WS / "utils.py").write_text(UTILS, encoding="utf-8")

    by = {t.name: t for t in make_tools(WS)}
    print(f"=== 七件套登记：{'、'.join(by)} ===")

    print("\n--- 感知三件（只读先行：先看清楚，再动手）---")
    show('glob_search("**/*.py")', by["glob_search"].invoke({"pattern": "**/*.py"}))
    show('grep_search("TODO", glob="**/*.py")', by["grep_search"].invoke({"pattern": "TODO", "glob": "**/*.py"}))
    show('read_file("app.py")', by["read_file"].invoke({"path": "app.py"}))

    print("\n--- 执行四件（改世界）---")
    show('edit_file：修 average 分母 bug（old→new 唯一匹配）', by["edit_file"].invoke({
        "path": "app.py",
        "old_string": "    return sum(nums) / (len(nums) + 1)  # TODO: 返回值总偏小——检查分母",
        "new_string": "    return sum(nums) / len(nums)"}))
    show('edit_file：old_string="return"（出现 2 次）', by["edit_file"].invoke({
        "path": "app.py", "old_string": "return", "new_string": "result"}))
    show('write_file("gen.py")：故意缺右括号 → 即时反馈', by["write_file"].invoke({
        "path": "gen.py", "content": BAD_GEN}))
    show('edit_file：补上右括号 → 语法检查通过', by["edit_file"].invoke({
        "path": "gen.py",
        "old_string": 'print("TODO 清单已生成"',
        "new_string": 'print("TODO 清单已生成")'}))
    show('bash("python gen.py")', by["bash"].invoke({"command": "python gen.py"}))
    show('py_exec：验证修复（不落盘，元能力随手用）', by["py_exec"].invoke({
        "code": "from app import average; print('average([2, 4]) =', average([2, 4]))"}))

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（零 API 调用，纯工具行为）：
#   1. 重建工作区（app.py 带 TODO 标注的分母 bug + utils.py 干净文件）
#   2. 感知三件：glob 列出 *.py → grep 定位 TODO 在 app.py:2 → read 带行号读全文
#   3. 执行四件：edit 修 bug（语法检查通过）→ edit 消歧失败（"return" 2 次）
#      → write 缺括号的 gen.py（语法检查当场报错）→ edit 补括号 → bash 真跑
#      → py_exec 验证 average([2, 4]) == 3.0（修复的铁证）
# 验证内容：
#   - read 返回每行带「行号→」前缀；grep 返回「文件:行号:内容」——行号可互相对上
#   - edit 成功案例返回「已修改 + 语法检查：通过」；失败案例返回 [错误] 且不改文件
#   - write 带语法错误的 .py：返回值里直接出现 SyntaxError（含行号）——即时反馈
#   - bash 输出「TODO 清单已生成」；py_exec 输出 average([2, 4]) = 3.0
#   - 对照彩蛋：py_exec 不落盘就验证了 edit 的效果——中间验证留在执行环境，
#     这正是 28 彩蛋「代码编排」的又一次预演
