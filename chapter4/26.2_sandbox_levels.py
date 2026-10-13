"""chapter4/26.2_sandbox_levels.py — 沙盒隔离谱系：venv 不是沙盒

本实验新增的主概念是「执行环境的隔离强度谱系」。承接 26.1：闸门（输入
验证）通过后，命令在什么「房间」里跑？课件先澄清一个高频误区——
venv 不是沙盒：它只隔离包依赖（site-packages 那个抽屉），代码的腿照样
伸出去删文件、读数据、连网络。换抽屉 ≠ 上锁房。

真隔离按强度递进三层（了解级，对照表与 Docker 示范见 README）：
    进程级（本地直接跑，如 Claude Code）：共享系统一切——本实验实测
    它的全部家当只有「一个圈 + 一根绳」：
        圈 = cwd 限定工作目录：写入默认落在圈内（真测试），但圈只圈
        「默认落脚点」不圈腿——绝对路径照样伸到圈外（真读到系统文件）
        绳 = timeout 到点绞杀：长命令被终止——但绳是止损不是防害，
        命令已经跑起来了，只是到点被杀
    容器（Docker）：独立文件系统 / 网络栈，但共享内核——逃逸需内核漏洞
    microVM（Firecracker）：独立内核，最强隔离——跑完全不可信代码
容器/microVM 还要配资源限额（--memory/--cpus，防把宿主拖垮）。

运行：uv run python -X utf8 chapter4/26.2_sandbox_levels.py（纯本地，不烧 API）
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

WS = Path(tempfile.mkdtemp(prefix="sandbox_ws_"))  # 沙盒工作目录（系统临时区）


def run_in_sandbox(cmd: str, timeout: float) -> str:
    """进程级沙盒的全部家当：圈（cwd）+ 绳（timeout）"""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           cwd=WS, timeout=timeout)
        return (r.stdout + r.stderr).strip() or "（无输出）"
    except subprocess.TimeoutExpired:
        return f"超时 {timeout}s，进程已终止"


if __name__ == "__main__":
    print("=== 误区澄清：venv 不是沙盒（此刻我们就跑在 venv 里） ===")
    print(f"  sys.prefix = {sys.prefix}")
    victim = Path(tempfile.mkdtemp(prefix="outside_")) / "重要文件.txt"
    victim.write_text("假装是圈外的重要数据", encoding="utf-8")  # 故意放在工作目录之外
    os.remove(victim)
    print(f"  venv 里的这段代码刚删了圈外文件 {victim.name}——文件还在吗：{victim.exists()}")
    print("  → venv 只隔离「装包的抽屉」（包依赖）；删文件、读数据、连网络")
    print("    的腿全在抽屉外——换抽屉 ≠ 上锁房")

    print("\n=== 进程级沙盒实测：一个圈 + 一根绳 ===")
    print(f"  圈（cwd = {WS}）：")
    run_in_sandbox("echo hello > out.txt", timeout=5)
    landed = (WS / "out.txt").read_text(encoding="utf-8").strip()
    print(f"    [圈内] echo hello > out.txt → 文件落在工作目录，内容 {landed!r}")
    winini = run_in_sandbox("type C:\\Windows\\win.ini", timeout=5).splitlines()[0]
    print(f"    [圈外] type C:\\Windows\\win.ini → 读到系统文件首行：{winini}")
    print("    → 圈只圈「默认落脚点」不圈腿：绝对路径照样伸到圈外")
    print("  绳（timeout = 1s）：")
    print(f"    [绳] ping -n 3 127.0.0.1（约 2 秒）→ {run_in_sandbox('ping -n 3 127.0.0.1', timeout=1)}")
    print("    → 绳是止损不是防害：命令已经跑起来了，只是到点被绞杀")

    print("\n=== 隔离强度谱系（了解级；Docker 实操示范见 README） ===")
    for line in [
        "进程级（刚才的圈+绳）：共享系统一切，逃逸零门槛——本地开发够用",
        "容器（Docker）：独立文件系统/网络栈，共享内核——逃逸需内核漏洞",
        "microVM（Firecracker）：独立内核，最强隔离——跑完全不可信代码",
    ]:
        print(f"  {line}")
    print("  → 容器/microVM 还要资源限额（--memory/--cpus）防拖垮宿主；")
    print("    本地开发进程级够用，生产或不可信输入必须往上走")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（纯本地，真子进程 + 真删文件）：
#   1. venv 误区：打印 sys.prefix（身处 venv 的证据）→ 在圈外临时目录建
#      文件 → 就地在 venv 的 Python 进程里 os.remove → 检查文件是否还在
#   2. 圈：cwd=工作目录跑 echo hello > out.txt（落点在圈内）；再跑
#      type C:\Windows\win.ini（绝对路径伸到圈外）
#   3. 绳：timeout=1s 跑约 2 秒的 ping → TimeoutExpired，进程终止
#   4. 谱系了解级：进程/容器/microVM 三行 + 资源限额一句话
# 验证内容：
#   - sys.prefix 显示 .venv，而删除发生在系统临时区——「venv = 沙盒」
#     不成立：它隔离装包，不隔离系统
#   - out.txt 落在工作目录、内容 hello；win.ini 首行成功读出——圈只管
#     默认落脚点，绝对路径管不住
#   - 超时提示出现且脚本继续跑——绳止损不防害
#   - 删的是自建临时文件、读的是公开系统示例文件——无副作用
