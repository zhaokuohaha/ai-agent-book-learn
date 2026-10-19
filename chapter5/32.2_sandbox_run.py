"""chapter5/32.2_sandbox_run.py — 沙箱执行：限额、结构化超时与死循环演练

本实验新增的主概念是「资源限额与结构化超时反馈」（沙箱第三道防线）——
CPU/内存/磁盘配额 + 超时，防御死循环、fork 炸弹、无限写盘。

关键工程细节（与「杀掉就完事」的区别）：超时后要向 Agent 返回结构化错误
——「执行超过 N 秒被终止，最后输出如下……」——而不是静默杀进程。模型
下一轮看得到它被杀前干了什么（打印到第几次心跳、走到哪个文件），才能
针对性修正策略；静默失败 = 模型只能瞎猜（D25「不知道自己不知道」、D29
错误变输入在超时场景的合流）。common/coding_tools.py 已同步升级：bash/
py_exec 的超时从「[超时] 已终止」一句话升级为带回死前输出的结构化反馈。

限额的跨平台形态（本实验为注释级——Windows 无 resource.setrlimit）：
    Unix：preexec_fn 里 resource.setrlimit(RLIMIT_CPU/RLIMIT_AS)（课件示例）
    Windows：Job Object（生产形态）；容器：--memory/--cpus（26.2 有实测示范：
    docker run --memory=64m 下 200MB 分配被 OOM Killed）
本实验可实测的是「绳」（超时）与结构化反馈路径——死循环真实跑、真实被杀、
死前输出真实回来。

运行：uv run python -X utf8 chapter5/32.2_sandbox_run.py（无需 .env；段 2 真等 15 秒）
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.coding_tools import make_tools

WS = Path(__file__).parent / ".agent_ws5"

SPIN = '''import time
n = 0
while True:
    print(f"心跳 {n}")
    n += 1
    time.sleep(0.5)
'''


def run_limited(code: str, timeout: int = 3) -> str:
    """段 1 原理演示：短超时 + 手工提取 TimeoutExpired 的死前输出
    （PYTHONUNBUFFERED 必须：管道下 stdout 块缓冲，不强制无缓冲则死前
    输出根本进不了管道——首跑翻车实证）"""
    try:
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=timeout, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8",
                            "PYTHONUNBUFFERED": "1"})
        return "正常返回（不该到这——死循环不会自己停）"
    except subprocess.TimeoutExpired as e:
        tail = (e.stdout or "") if isinstance(e.stdout, str) else ""
        tail = tail.strip().splitlines()[-1] if tail.strip() else "（无输出）"
        return f"[超时] 超过 {timeout}s 被终止。最后输出：{tail}"


if __name__ == "__main__":
    WS.mkdir(parents=True, exist_ok=True)
    (WS / "spin.py").write_text(SPIN, encoding="utf-8")

    print("=== 段 1：原理——死循环被杀，死前输出带得回来 ===")
    print(f"  {run_limited(SPIN)}")

    print("\n=== 段 2：集成——common 升级后的 bash/py_exec（真等 15 秒） ===")
    by = {t.name: t for t in make_tools(WS)}
    r = str(by["bash"].invoke({"command": "python spin.py"}))
    print(f"  bash('python spin.py') →")
    for ln in r.splitlines():
        print(f"    {ln}")
    print("  → 反馈是结构化的：超时事实 + 死前最后输出——模型下一轮能据此修正")
    print("    （比如「循环打印太频繁/该加终止条件」），而不是面对静默失败瞎猜")

    print("\n=== 段 3（注释级）：CPU/内存限额的跨平台形态 ===")
    print("  Unix   preexec_fn: resource.setrlimit(RLIMIT_CPU, (10,10)) + RLIMIT_AS 512MB")
    print("  Win    Job Object（win32job）；容器 --memory=64m/--cpus=0.5（26.2 实测 OOM Killed）")
    print("  → 死循环吃的是 CPU（本实验绳先杀）；内存炸弹/fork 炸弹/无限写盘各自对应限额")

# ---- 执行逻辑与验证（看完代码再回头这里） ----
# 执行逻辑（零 API 调用；段 2 真等 15 秒）：
#   1. 段 1：3 秒短超时跑内联死循环——TimeoutExpired.stdout 提取「心跳 N」末行
#   2. 段 2：spin.py（0.5s 间隔心跳）经 common 的 bash 跑——15s 绳生效，
#      返回升级后的结构化反馈（超时 + 最后输出≈「心跳 28」量级）
#   3. 段 3：限额形态打印（Windows 无 resource，注释级——26.2 Docker 示范承接）
# 验证内容：
#   - 段 1 的最后输出是「心跳 N」且 N>0——死前输出真实回来了
#   - 段 2 的返回含「[超时] 超过 15s 被终止」与「最后输出：心跳 xx」两段
#     ——对照升级前的「[超时] 命令超过 15 秒，已终止」一句话：多出来的
#     就是模型修正策略所需的唯一新信息
#   - 死循环的 CPU 占用在绳生效前真实存在（任务管理器可观察）——限额缺位
#     的现状即段 3 的动机
#   - 首跑翻车记（两层，都修在 common）：① 死前输出空——管道下子进程 stdout
#     块缓冲，几条心跳填不满缓冲区被杀时没 flush；修复：env 加
#     PYTHONUNBUFFERED=1（框架层强制，不能指望子进程自觉 -u）② bash 挂死
#     ——subprocess.run(shell=True) 超时只杀 cmd 外壳，孙进程成孤儿持有
#     管道写端，收尾读永久阻塞；修复：taskkill /F /T 级联杀进程树后再读
