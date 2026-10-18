"""chapter5/31.1_persistent_shell.py — 持久化终端会话：cd 为什么会丢

本实验新增的主概念是「命令执行环境的状态持久化」——很多命令依赖环境
状态（cd 切目录、设环境变量、起后台服务）；每条命令一个新 shell，状态
全部蒸发：刚 cd 进子目录，下一条命令又回到原地。

后端类比：连接池复用 vs 每次查询重建连接——session 内状态（事务、临时
变量）随连接断开全丢；持久化终端 = 长连接，每条命令 = 一次查询。D30 的
bash 工具每次 subprocess.run 都是「重建连接」（反例现状）；本实验给它
看正解：常驻 cmd 进程，stdin 喂命令、stdout 收结果，cwd/env 跨命令保留。

常驻进程的工程难点（本实验的代码落点）：
    ① 怎么知道命令跑完了？——哨兵标记：命令后跟 echo <唯一标记>，
       读到标记即结束（生产系统同款思路，Claude Code 的 shell 也是
       分隔符定界）
    ② Windows 管道没有 select——读 stdout 会阻塞，超时只能开线程
       看门（真实系统：超时杀进程重建会话）
    ③ 长输出截断：头尾保留、中间省略——编译/测试输出动辄几百行，
       全进上下文就是 D25 的爆炸现场

教学简化注：cmd 内建命令输出走 GBK（代码页），编码按 GBK 解——跑
python 命令（PYTHONIOENCODING=utf-8）会混编码乱码，本实验演示命令只用
cmd 内建（cd/set/echo）；真实产品的持久 shell 按平台选 powershell/bash
并统一编码层。

运行：uv run python -X utf8 chapter5/31.1_persistent_shell.py（无需 .env——纯工具行为）
"""

import re
import shutil
import subprocess
import sys
import threading
from itertools import count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WS = Path(__file__).parent / ".agent_ws3"
HEAD, TAIL = 15, 5  # 长输出截断：保留头 15 行 + 尾 5 行
PROMPT_RE = re.compile(r"^[A-Za-z]:\\[^>]*>")  # 提示符形状：盘符路径+>（reader 提前停会残留到下一条开头）
_seq = count()


class PersistentShell:
    """常驻 cmd 会话：cwd/env 跨命令保留（长连接）；哨兵定界 + 线程读超时"""

    def __init__(self, ws: Path):
        self.proc = subprocess.Popen(
            ["cmd", "/q", "/k"], cwd=ws,  # /q 关回显，/k 保持会话不退出
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def run(self, command: str, timeout: int = 15) -> str:
        tag = f"___EOT_{next(_seq)}___"  # 唯一哨兵：本轮命令的结束标记
        # 哨兵独立成行：& 连接会被 for 吃进循环体（每轮输出 tag）；括号分组
        # 又会隔离 cd（块结束目录被还原）；\r\n 的 \r 还会粘进命令尾部（cd sub\r
        # 找不到目录）——纯 \n 换行分隔是 cmd 管道唯一稳妥形式
        payload = f"{command}\n echo {tag}\n".encode("gbk")
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()
        buf = bytearray()

        def reader():  # Windows 管道无 select：阻塞读只能放线程，主线程管超时
            while True:
                b = self.proc.stdout.read(1)
                if not b:
                    break
                buf.extend(b)
                if tag.encode() in buf:
                    break

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            return f"[超时] {timeout}s 未见哨兵（真实系统：杀进程重建会话，此处仅返回）"
        lines = [PROMPT_RE.sub("", ln, count=1)  # 剥掉残留提示符（上轮 reader 读到哨兵即停，提示符晚到残留）
                 for ln in bytes(buf).decode("gbk", errors="replace").splitlines()
                 if tag not in ln and ln.strip()]
        if len(lines) > HEAD + TAIL + 3:  # 长输出截断：头尾保留中间省略（D25 回响）
            skipped = len(lines) - HEAD - TAIL
            lines = lines[:HEAD] + [f"...[中间省略 {skipped} 行]..."] + lines[-TAIL:]
        return "\n".join(lines) or "（无输出）"

    def close(self):
        self.proc.stdin.close()
        self.proc.kill()
        self.proc.wait()


if __name__ == "__main__":
    shutil.rmtree(WS, ignore_errors=True)
    (WS / "sub").mkdir(parents=True)
    (WS / "sub" / "note.txt").write_text("子目录内容\n", encoding="utf-8")

    print("=== A 无状态：每条命令一个新 shell（连接池每次重建）===")
    subprocess.run("cd sub", shell=True, cwd=WS, capture_output=True)  # 成功，但状态随进程蒸发
    r = subprocess.run("cd", shell=True, cwd=WS, capture_output=True, text=True)  # 无参 cd 显示当前目录
    print(f"  第一条：cd sub（成功）\n  第二条：cd → {r.stdout.strip()}")
    print("  → 回到了起点：cd 白做了——30.1 的 bash 工具正是这个行为")

    print("\n=== B 持久会话：常驻 cmd，状态跨命令保留（长连接）===")
    shell = PersistentShell(WS)
    print(f"  cd sub → {shell.run('cd sub')}（cd 成功不吭声，状态留在会话里）\n  cd → {shell.run('cd')}")
    print(f"  dir /b → {shell.run('dir /b')}")  # 列出的是 sub 的内容——cwd 真的跟着走了
    shell.run("set NOTE=hello")  # cmd 语法：= 前不能有空格
    print(f"  set NOTE=hello →（写入会话变量）\n  echo %NOTE% → {shell.run('echo %NOTE%')}")
    print("\n  长输出截断（头 15 行 + 尾 5 行，中间省略）：")
    out = shell.run("for /L %i in (1,1,40) do echo line %i")
    print(f"  {out}")
    shell.close()

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（零 API 调用）：
#   1. 预置 sub 子目录；A 版两条独立 subprocess.run 模拟「每次新 shell」
#   2. B 版常驻 cmd：cd sub → cd（显示 ws\sub）→ dir /b（列出 note.txt）
#      → set NOTE → echo %NOTE%（hello）→ 40 行循环输出验证截断
# 验证内容：
#   - A 版第二条 cd 输出仍是 WS 根目录——状态蒸发的活证据
#   - B 版 cd 显示 ...\sub、dir /b 列出 note.txt——cwd 跨命令保留
#   - echo %NOTE% 输出 hello——会话变量保留（env 同理）
#   - 40 行输出被截成头 15 + 省略提示 + 尾 5——长输出不灌爆上下文
#   - 哨兵机制：每条命令的输出干净截取。首跑翻车记（cmd 分隔符两大坑）：
#     ① & 连接哨兵会被 for 吃进循环体——每轮都输出 tag，reader 第一轮就停；
#     ② 括号分组隔离 cd——(cd sub) 块结束目录被还原（set 不受影响）——
#     终解：哨兵独立成行（换行分隔），命令与哨兵各自独立执行
#   - 简化注：超时只返回不重建（真实系统杀进程重建）；编码按 GBK——
#     cmd 内建命令自洽，python 命令混编码（PYTHONIOENCODING=utf-8）勿在此混用
