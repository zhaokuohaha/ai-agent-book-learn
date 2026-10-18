"""chapter5/31.2_line_number_trap.py — 行号漂移陷阱：编辑为什么收敛到内容匹配

本实验新增的主概念是「文件编辑方案的收敛规律」——五种方案（diff+Apply
Model / old→new / 行号 / 类 Vim / 首尾匹配）里，业界正从「行号、Vim 这类
抽象定位」收敛到「基于内容匹配」：内容就在上下文里，模型看得见、对得上；
行号是外部状态，模型算不准漂移后的坐标。

本实验给这条收敛规律做实证（作业①的反面教材）：同一份文件、同一组
「两处同型修改」，两套编辑协议走一遍——
    A 行号方案（问题演示侧玩具）：模型基于阅读时的行号快照提交两处
    编辑，框架逐个应用——第一处插入 1 行后，后面所有行号集体漂移，
    第二处按旧行号定位，静默删错内容
    B old→new 方案（29.1 的 edit_file，生产主流同款）：每处编辑独立
    匹配原文内容——天然不存在漂移问题

行号陷阱最阴的地方：错位是静默的——py_compile 抓不住（语法依然合法），
只有运行才炸（NameError）。「compile 过了」不等于「改对了」，这正是
D25「不知道自己不知道」在编辑侧的回响，也是完成标准必须是「测试通过」
（29.2）而非「代码写完」的又一实证。

B 方案顺带覆盖 D31 课件概念五（即时语法反馈）：edit_file 内嵌 py_compile，
成功返回值自带语法检查结果。

运行：uv run python -X utf8 chapter5/31.2_line_number_trap.py（无需 .env——编辑计划是模拟的，确定性复现）
"""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.coding_tools import make_tools

WS = Path(__file__).parent / ".agent_ws4"

PRESET = '''def log(msg):
    print(msg)


def clean(x):
    return x


def enrich(x):
    return x


def process_a(data):
    step1 = clean(data)
    return step1


def process_b(data):
    step2 = enrich(data)
    return step2
'''

# 模型的编辑计划（基于阅读时的行号快照——模型看不见「应用后的现在」）：
PLAN = [
    (15, 15, "    log('a done')\n    return step1"),   # 快照第 15 行：process_a 的 return 前加日志
    (20, 20, "    log('b done')\n    return step2"),   # 快照第 20 行：process_b 的 return 前加日志（应用编辑 1 后已漂移到 21）
]


def apply_line_edits(path: Path, plan) -> None:
    """行号编辑（问题侧玩具）：逐个应用「编辑前快照」的行号——第二个起全部错位"""
    lines = path.read_text(encoding="utf-8").splitlines()
    for start, end, new in plan:
        lines = lines[:start - 1] + new.splitlines() + lines[end:]  # 按旧行号删插
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def show(title: str, path: Path) -> None:
    print(f"  {title}（read_file 视角）：")
    for ln in path.read_text(encoding="utf-8").splitlines()[-6:]:
        print(f"    {ln}")


def run_check(path: Path) -> str:
    """运行验证：process_b() 调一下——错位版会在这里炸"""
    r = subprocess.run([sys.executable, "-c", "import t; t.process_b(1)"], cwd=WS,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return "运行通过" if r.returncode == 0 else f"运行报错：{r.stderr.strip().splitlines()[-1]}"


if __name__ == "__main__":
    shutil.rmtree(WS, ignore_errors=True)
    WS.mkdir(parents=True)
    (WS / "t.py").write_text(PRESET, encoding="utf-8")

    by = {t.name: t for t in make_tools(WS)}

    print("=== A 行号方案：两处编辑基于同一份行号快照，逐个应用 ===")
    print(f"  编辑计划：{[(s, e) for s, e, _ in PLAN]}（第二处的行号在第一处应用后已漂移 +1）")
    apply_line_edits(WS / "t.py", PLAN)
    show("编辑后 t.py 尾部", WS / "t.py")
    print(f"  py_compile → {'通过（静默！语法依然合法）' if subprocess.run([sys.executable, '-m', 'py_compile', str(WS / 't.py')]).returncode == 0 else '未通过'}")
    print(f"  {run_check(WS / 't.py')}——enrich 调用被静默删掉，step2 成了无源之水")

    print("\n=== B old→new 方案（29.1 edit_file）：每处独立匹配原文 ===")
    (WS / "t.py").write_text(PRESET, encoding="utf-8")  # 重置现场（对照实验铁律：A/B 同一初始状态）
    for old, new in [
        ("    step1 = clean(data)\n    return step1",
         "    step1 = clean(data)\n    log('a done')\n    return step1"),
        ("    step2 = enrich(data)\n    return step2",
         "    step2 = enrich(data)\n    log('b done')\n    return step2"),
    ]:
        r = by["edit_file"].invoke({"path": "t.py", "old_string": old, "new_string": new})
        print(f"  {str(r).splitlines()[0]}｜{str(r).splitlines()[-1]}")  # 修改结果 + 即时语法检查
    show("编辑后 t.py 尾部", WS / "t.py")
    print(f"  {run_check(WS / 't.py')}——两处日志都就位")

    print("\n结论：行号是外部状态（模型看不见应用后的坐标漂移），内容就在上下文里"
          "\n（看得见、对得上）——业界从抽象定位收敛到内容匹配，不是品味，是数学。")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（零 API 调用，编辑计划是模拟的——确定性复现行号陷阱）：
#   1. 预置 t.py（两行 stub + 两个同型函数，行号见 PLAN）
#   2. A 版：apply_line_edits 逐个应用快照行号 → 第二处按旧 20 行实际删到
#      enrich 调用行 → read 展示坏尾 → py_compile（过！）→ 运行（NameError 炸）
#   3. B 版：重置 t.py → 两次 edit_file（old/new 独立匹配）→ 运行通过
# 验证内容：
#   - A 版尾部：process_b 里 enrich 消失、step2 无定义；compile 通过 + 运行
#     报 NameError——「compile 过了 ≠ 改对了」三层证据链（read 看得见 /
#     compile 不报 / run 才炸）
#   - B 版两次 edit_file 返回「已修改 + 语法检查：通过」（即时反馈内嵌）；
#     运行通过——两处日志就位
#   - 对照铁律：B 版前重置 t.py（22.4 首跑教训——A 的破坏不能带进 B 的现场）
#   - B 版 old_string 含两行上下文：内容匹配方案的「代价」——删大段时要
#     完整输出原文；这是首尾匹配方案（方案五）要优化的点，见 README 对照表
