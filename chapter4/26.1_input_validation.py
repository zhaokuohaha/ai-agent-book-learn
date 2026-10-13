"""chapter4/26.1_input_validation.py — 执行工具第一闸：输入验证与快速失败（作业①③）

本实验新增的主概念是「执行工具的输入验证层」。承接 22.1（执行工具 =
POST/DELETE 写世界）与 22.4（拒绝而非静默修正）：感知工具读错只是信息
不对，执行工具发错命令撤不回——所以第一设计原则是快速失败：发现异常
立即拒绝，绝不「智能修正」（那是 22.4 的平行宇宙噩梦）。

实现课件参考版 safe_shell 的闸门，两个版本同台对照：
    v1 课件原版：白名单（允许清单）+ 黑名单（危险模式兜底）——但白名单
    只看「首命令」。四行测试（课件三行 + 黑名单对照一行）：echo hello
    放行 / rm -rf / 拒 / echo x & rm -rf / 拒（黑名单抓已知串）/
    echo x & whoami —— ⚠️ 绕过！whoami 不在白名单也不在黑名单，照样
    执行了（作业③现场；Linux 课件对应 echo x; cat /etc/passwd）
    v2 作业③修复：按 & | 拆链，逐段校验首命令——同一命令从「执行了」
    变「链中命令不在白名单，拒绝」，正常命令不误伤

顺带课件参考实现第三层「执行侧截断」：与 25.1 的读取类对照——读的
截断有 offset 续读（原文还在服务端），执行的输出是一次性产物，丢了
就永远丢了——所以必须持久化落盘，截断提示里给出文件路径。

运行：uv run python -X utf8 chapter4/26.1_input_validation.py（纯本地，不烧 API）
"""

import re
import shlex
import subprocess
import tempfile
from pathlib import Path

# ---- 闸门规则：白名单为主、黑名单兜底（黑名单是平台相关的已知攻击串清单） ----

ALLOWED = {"echo", "type", "dir", "findstr", "python"}  # cmd 内建 + python
BLOCKED = re.compile(r"rm\s+-rf|mkfs|dd\s+if=|curl[^|]*\|\s*(ba)?sh|del\s+/[sq]|format\s+[a-z]:")
SEGMENT = re.compile(r"&&?|\|\|?")  # cmd 链式分隔符（Linux 课件的对应物是 ; | &&）
LIMIT = 2000  # 输出截断线
SAVE = Path(tempfile.mkdtemp(prefix="agent_ws_")) / ".last_output.txt"  # 完整输出落盘处


def first_word(segment: str) -> str:
    parts = shlex.split(segment)
    return parts[0].lower() if parts else ""


def execute(cmd: str) -> dict:
    """校验通过后的执行——带上课件第三层：执行侧截断 + 持久化。"""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    full = (r.stdout + r.stderr).strip()
    out = full[:LIMIT]
    if len(full) > LIMIT:
        SAVE.write_text(full, encoding="utf-8")
        out += f"\n...[已截断：完整输出共 {len(full)} 字符，已存 {SAVE}，可分段读取]"
    return {"ok": r.returncode == 0, "output": out}


def safe_shell(cmd: str) -> dict:
    """v1 课件原版：白名单查首命令 + 黑名单兜底——漏洞就在「首命令」三个字"""
    if first_word(cmd) not in ALLOWED:
        return {"ok": False, "error": f"首命令不在白名单: {first_word(cmd)!r}"}
    if BLOCKED.search(cmd):
        return {"ok": False, "error": "命中危险命令模式，已拒绝"}
    return execute(cmd)


def safe_shell_v2(cmd: str) -> dict:
    """v2 作业③修复：先拆链，每段首命令都过白名单——链中任何一段出圈即拒"""
    for seg in SEGMENT.split(cmd):
        word = first_word(seg)
        if word and word not in ALLOWED:
            return {"ok": False, "error": f"链中命令不在白名单: {word!r}"}
    if BLOCKED.search(cmd):
        return {"ok": False, "error": "命中危险命令模式，已拒绝"}
    return execute(cmd)


def show(label: str, cmd: str, fn) -> None:
    r = fn(cmd)
    verdict = f"放行｜{r['output']!r}" if r["ok"] else f"拒绝｜{r['error']}"
    print(f"  [{label}] {cmd!r} → {verdict}")


if __name__ == "__main__":
    print("=== 作业① 课件三行测试 + 黑名单对照（v1） ===")
    for label, cmd in [
        ("正常任务", "echo hello"),
        ("教科书攻击", "rm -rf /"),
        ("链中藏已知攻击串", "echo x & rm -rf /"),
        ("作业③绕过现场", "echo x & whoami"),
    ]:
        show(label, cmd, safe_shell)

    print("\n停一停：echo x & whoami 为什么能绕过白名单？（先想 3 秒再看）")
    print("  答：白名单只查「首命令」echo。cmd 用 & 串联整条链，第二段的")
    print("  whoami 从未被检查——字符串看的是开头，shell 执行的是全链。")
    print("  黑名单也救不了：它只认「已知」攻击串，whoami 是编外的")

    print("\n=== 作业③ 修复：v2 拆链逐段校验 ===")
    for label, cmd in [("同一绕过命令", "echo x & whoami"), ("正常任务不误伤", "echo hello")]:
        show(label, cmd, safe_shell_v2)

    print("\n=== 顺带第三层：执行侧截断 + 持久化（对照 25.1） ===")
    long_cmd = "python -c \"print('x' * 5000)\""
    r = safe_shell_v2(long_cmd)
    print(f"  {long_cmd}")
    print(f"    → ok={r['ok']}｜正文截到 {len(r['output'].splitlines()[0])} 字符（原始 5000）；尾部提示：")
    print(f"    {r['output'].splitlines()[-1]}")
    print("  → 读的截断可 offset 续读（原文还在服务端）；执行的输出丢了就没了")
    print("    ——落盘 + 给路径，模型知道「完整版在哪、怎么拿」")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（纯本地，真子进程；工作目录在系统临时区，无副作用）：
#   1. v1 四行测试：echo hello / rm -rf / / echo x & rm -rf / / echo x & whoami
#   2. 停一停：链式绕过的原因（白名单查首命令，shell 执行全链）
#   3. v2 对照：同一绕过命令被拒 + 正常命令放行（修复不误伤）
#   4. 长输出命令：截到 2000 字符 + 完整内容写入 .last_output.txt
# 验证内容：
#   - 绕过行的输出里是真实用户名——「whoami 真的执行了」不是理论是事实
#   - v2 对同一命令报「链中命令不在白名单: 'whoami'」，echo hello 照常放行
#   - 截断尾部有文件路径；打开该文件可看全 5000 字符（输出丢了就没了，
#     所以必须落盘——与 25.1「读可续读」形成对照）
#   - 彩蛋：课件的字符串黑名单挡得住 echo x & rm -rf /（已知串）却挡不住
#     whoami——黑名单是兜底不是防线，语义审查是 26.3 的课题
