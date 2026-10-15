"""common/coding_tools.py — Coding Agent 七件套（第 5 章沉淀）

设计取舍（教学版七工具，29.2 测试闭环与后续主循环实验直接 import 复用）：
    七件套覆盖五分类里的感知（read/glob/grep）与执行（write/edit/bash/py_exec）
    两轴——协作/事件触发/用户沟通不是 Coding Agent 的核心（原书第 5 章）。
    理论上一个 bash 就够（OpenAI Codex 就这么干），拆成七件让每次调用更
    精准：模型说「找文件」就直接 glob，不用 bash dir 再解析输出——工具
    语义越专，模型越不容易选错（D24 回响）。
    edit 走 old→new 字符串替换（Claude Code / Codex / Cursor 现行方案）：
    唯一即成功、不唯一即失败、找不到就报错——绝不模糊匹配（22.4 参数保真）。
    read 带行号前缀（模型能精确引用「第 42 行」，后续 edit 更可靠）；
    write/edit 一完成立即 py_compile（语法错误当刻可见，不等跑测试）。
    make_tools(ws) 工厂把工作区路径闭包进工具——路径越界一律 [错误]，
    白名单+圈+绳（cwd/timeout）承自 D26 的执行工具三道闸。
"""

import os
import re
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import BaseTool, tool

ALLOWED = {"python"}  # bash 白名单：dir 已移除——glob_search 覆盖其职责且 cmd 内建命令输出 GBK 不可控
SEGMENT = re.compile(r"&&?|\|\|?")  # 链式命令按 & | 拆段，逐段查白名单


def _run(argv, cwd):
    """子进程统一入口：UTF-8 强制 + 超时（圈+绳的「绳」）"""
    return subprocess.run(argv, capture_output=True, text=True, cwd=cwd, timeout=15,
                           encoding="utf-8", errors="replace",
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def make_tools(ws: Path) -> list[BaseTool]:
    """七件套工厂：工作区闭包进每个工具——工具的合法活动半径"""
    ws = Path(ws).resolve()

    def _resolve(path: str):
        p = (ws / path).resolve()
        return p if ws in p.parents else None

    def _syntax_check(p: Path) -> str:
        """写完立即语法检查——IDE 红线思维：错误在引入当刻就可见"""
        r = _run([sys.executable, "-m", "py_compile", str(p)], cwd=ws)
        if r.returncode == 0:
            return "语法检查：通过"
        err = [ln for ln in r.stderr.splitlines() if ln.strip()][-3:]
        return "语法检查：未通过\n" + "\n".join(err)

    @tool
    def read_file(path: str, offset: int = 0, limit: int = 60) -> str:
        """当需要查看代码/配置/文档内容时使用。返回带行号前缀的行（如「4→code」），
        可精确引用行号；大文件按行范围读（offset 起始行、limit 行数），超出部分
        在返回值末尾写明总行数与续读参数，绝不静默截断。
        路径越界或文件不存在返回 [错误]。例如：read_file("stats.py")"""
        p = _resolve(path)
        if p is None or not p.is_file():
            return f"[错误] 文件不存在或越界：{path}"
        lines = p.read_text(encoding="utf-8").splitlines()
        part = lines[offset:offset + limit]
        if not part:
            return f"[错误] 越过文件末尾（共 {len(lines)} 行）"
        body = "\n".join(f"{i + offset + 1:>4}→{ln}" for i, ln in enumerate(part))
        if offset + limit < len(lines):
            body += f'\n[共 {len(lines)} 行，续读 read_file("{path}", offset={offset + limit})]'
        return body

    @tool
    def write_file(path: str, content: str) -> str:
        """当需要创建新文件或整体重写现有文件时使用；局部修改请改用 edit_file。
        写 .py 文件立即做语法检查，错误（含行号）直接出现在返回值里。
        返回「已写入 路径」+ 语法检查结果；越界返回 [错误]。
        例如：write_file("calc.py", "print(1+2)")"""
        p = _resolve(path)
        if p is None:
            return "[错误] 路径必须在工作区内"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        msg = f"已写入 {path}（{len(content.splitlines())} 行）"
        if p.suffix == ".py":
            msg += "\n" + _syntax_check(p)
        return msg

    @tool
    def edit_file(path: str, old_string: str, new_string: str) -> str:
        """当需要对现有文件做局部修改时使用（代码维护迭代的核心操作）。
        old_string 必须与文件内容逐字符一致且唯一——不唯一时请附更长上下文消歧；
        找不到就报错，绝不模糊匹配（快速失败）。read_file 的行号前缀不属于
        文件内容，old_string 请去掉行号。返回修改结果 + 语法检查（.py）。
        例如：edit_file("stats.py", old_string="return s[len(s)//2]",
        new_string="n=len(s); return s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2")"""
        p = _resolve(path)
        if p is None or not p.is_file():
            return f"[错误] 文件不存在或越界：{path}"
        text = p.read_text(encoding="utf-8")
        n = text.count(old_string)
        if n == 0:
            return "[错误] old_string 未找到——先 read_file 核对原文（去掉行号前缀），逐字符一致才能匹配"
        if n > 1:
            return f"[错误] old_string 出现 {n} 次——请附上更多上下文使其唯一"
        p.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
        msg = f"已修改 {path}（1 处替换）"
        if p.suffix == ".py":
            msg += "\n" + _syntax_check(p)
        return msg

    @tool
    def glob_search(pattern: str) -> str:
        """当需要按文件名模式定位文件时使用——探索项目结构的第一步（比内容搜索快）。
        pattern 走 glob 语法：**/*.py 找所有 Python 文件、data/** 找子目录。
        只匹配文件名不看内容；无匹配时明说「无匹配」。
        例如：glob_search("**/*.py")"""
        hits = sorted(str(q.relative_to(ws)) for q in ws.glob(pattern) if q.is_file())
        return "\n".join(hits) if hits else "无匹配"

    @tool
    def grep_search(pattern: str, glob: str = "**/*") -> str:
        """当需要在文件内容里搜索文本/正则时使用，如找 TODO、函数定义、报错来源。
        返回「文件:行号:该行内容」，可拿行号去 read_file 深读；只搜文本不搜语义
        （不知道确切用词时先 glob/read 浏览）。pattern 走正则语法；命中过多时
        只显示前 30 处并注明总数（收窄 glob 或精确化 pattern）。
        例如：grep_search("TODO", glob="**/*.py")"""
        rx = re.compile(pattern)
        hits = []
        for q in sorted(ws.glob(glob)):
            if not q.is_file():
                continue
            for i, ln in enumerate(q.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(ln):
                    hits.append(f"{q.relative_to(ws)}:{i}:{ln.strip()}")
        if not hits:
            return "无匹配"
        if len(hits) > 30:
            return "\n".join(hits[:30]) + f"\n[共 {len(hits)} 处命中，仅显示前 30]"
        return "\n".join(hits)

    @tool
    def bash(command: str) -> str:
        """当需要在终端执行命令时使用，典型场景：运行测试脚本（python test_xxx.py）。
        只执行白名单命令（python），链式命令逐段校验；查文件清单请用 glob_search
        （更精确且不依赖终端编码）。cwd 锁定工作区、超时 15 秒。
        返回 stdout+stderr（最多 2000 字符）。例如：bash("python test_stats.py")"""
        for seg in SEGMENT.split(command):
            parts = seg.strip().split()
            if parts and parts[0].lower() not in ALLOWED:
                return f"[拒绝] 链中命令不在白名单：{parts[0]!r}"
        try:
            r = subprocess.run(command, shell=True, capture_output=True, text=True,
                               cwd=ws, timeout=15, encoding="utf-8", errors="replace",
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        except subprocess.TimeoutExpired:
            return "[超时] 命令超过 15 秒，已终止"
        out = (r.stdout + r.stderr).strip()
        return out[:2000] if out else "（无输出）"

    @tool
    def py_exec(code: str) -> str:
        """当需要直接执行一段 Python 代码时使用——沙盒解释器是元能力：临时写个
        工具算数据/验证想法，不必先落盘成文件。在隔离子进程执行（cwd 锁定
        工作区、超时 15 秒）；stdout/stderr 原样返回（最多 2000 字符）。
        例如：py_exec("print(sorted([3, 1, 2]))")"""
        try:
            r = _run([sys.executable, "-c", code], cwd=ws)
        except subprocess.TimeoutExpired:
            return "[超时] 代码超过 15 秒，已终止"
        out = (r.stdout + r.stderr).strip()
        return out[:2000] if out else "（无输出）"

    return [read_file, write_file, edit_file, glob_search, grep_search, bash, py_exec]
