"""chapter5/29.2_test_fix_loop.py — 测试-修复循环：「测试通过」才算完成

本实验新增的主概念是「测试-修复循环」——课件最小骨架印证（七件套 + while +
错误变输入）的落地。Coding Agent 最常见的偷懒是写完代码不跑测试就报告完成，
解法是把「测试通过」而非「代码写完」定义为完成标准（Harness 四件套里的
「验收基线 + 反馈信号」——测试结果就是对错判官）。

任务设计（确定性闭环）：工作区预置带 bug 的 stats.py 和必失败的 test_stats.py
——初始状态确定失败，Agent 必须「跑测试 → 看失败输出 → 读代码（行号）→
edit 修复 → 再跑 → 通过」才能交付。工程化六阶段里最核心的 实现+测试 环节
就压缩在这几轮里；需求澄清/设计文档是复杂任务的台阶，修 bug 属于「边界
清晰、直接进实现」的简单需求（原书：流程按需裁剪）。

伴生概念：错误变输入（工具层故障恢复走另一条路——不终止会话）：
    ① 测试失败的输出原样回填 ToolMessage——模型下一轮自行定位修复
    ② 幻觉调用（调不存在的工具）收到结构化错误「工具不存在 + 可用清单」
    ③ 工具异常同样转结构化错误喂回——「反馈越结构化越好」（D26.4 同构）

七件套从 common/coding_tools.py import（29.1 沉淀）；循环骨架复用 D28——
区别只在工具从 3 件换成 7 件、完成标准从「读回核对」升级为「测试通过」。

骨架忠实点：messages 首位是 system prompt——模拟 AGENTS.md 注入（项目级系统
提示词）：项目约定（怎么干活）进 system、本次指令（干什么）进 user——D11
信任分层回响；作业②「AGENTS.md 写哪五条」的代码落点即此。

运行：uv run python -X utf8 chapter5/29.2_test_fix_loop.py（需 .env，约 4-7 次调用）
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from common.coding_tools import make_tools
from common.config import API_KEY, BASE_URL, MODEL

WS = Path(__file__).parent / ".agent_ws"

STATS = '''def median(nums):
    s = sorted(nums)
    return s[len(s) // 2]
'''

TEST = '''from stats import median

assert median([3, 1, 2]) == 2, "奇数长度：中间那个数"
assert median([4, 1, 3, 2]) == 2.5, "偶数长度：中间两数的平均"
print("全部测试通过")
'''

SYSTEM = ("你是在工作区里干活的 Coding Agent。项目约定（相当于本项目的 AGENTS.md，对任何任务都成立）："
          "修改现有代码一律用 edit_file 做局部修改，不整文件重写；"
          "完成标准是测试通过——bash 跑出通过输出才算完成，代码写完不算。")

TASK = ("工作区的 stats.py 有 bug：python test_stats.py 会失败。"
        "请修复它，直到测试输出「全部测试通过」。")


def run_agent(model: ChatOpenAI, max_turns: int = 10) -> None:
    """闭环循环：干净的 while——错误变输入的三条防线都在工具调度处"""
    tools = make_tools(WS)
    by_name = {t.name: t for t in tools}
    llm = model.bind_tools(tools)
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=TASK)]  # 课件骨架：system_prompt_with_project_md
    for _ in range(max_turns):
        resp = llm.invoke(messages)
        messages.append(resp)
        if not resp.tool_calls:
            print(f"\n[完成] {str(resp.content)[:200]}")
            return
        for call in resp.tool_calls:
            name, args = call["name"], call["args"]
            print(f"  [调用] {name}({str(args)[:70]})")
            if name not in by_name:  # 幻觉调用：结构化错误喂回，不崩溃
                result = f"[错误] 工具不存在：{name}——可用：{list(by_name)}"
            else:
                try:
                    result = str(by_name[name].invoke(args))
                except Exception as e:  # 工具异常也变输入：下一轮自行纠正
                    result = f"[错误] {type(e).__name__}: {e}"
            print(f"  [返回] {result[:90]}")
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    print("  [warn] 轮数用尽")


if __name__ == "__main__":
    shutil.rmtree(WS, ignore_errors=True)  # 重建工作区：bug 必在，闭环必走修复路径
    WS.mkdir(parents=True)
    (WS / "stats.py").write_text(STATS, encoding="utf-8")
    (WS / "test_stats.py").write_text(TEST, encoding="utf-8")

    print(f"=== 任务：{TASK}")
    run_agent(ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL,
                         timeout=60, max_retries=0))  # max_retries=0：重试策略自己写（29.3 的伏笔）

    print("\n=== 落盘终检（以测试为准，不以汇报为准）===")
    r = subprocess.run([sys.executable, "test_stats.py"], cwd=WS, capture_output=True,
                       text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})  # 子进程强制 UTF-8（Windows 管道默认 GBK）
    verdict = "通过" if "全部测试通过" in r.stdout else f"仍失败：{(r.stdout + r.stderr).strip()[:200]}"
    print(f"最终测试：{verdict}")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（约 4-7 次真实调用）：
#   1. 重建工作区：stats.py（median 偶数长度 bug）+ test_stats.py（两条断言）
#   2. Agent 轨迹预期：bash 跑测试 → 失败输出（AssertionError 回填=错误变输入）
#      → read_file(stats.py)（带行号）→ edit_file 修复 → bash 再跑 → 通过 → 汇报
#   3. 脚本最后自己再跑一次测试——验收不看模型的自评，看落盘状态
# 验证内容（模型路径可能不同，验收看闭环与产物）：
#   - 轨迹含「跑测试失败 → 修复 → 再跑」的完整闭环；失败输出确实出现在 [返回] 里
#   - system 约定生效：修复动作走 edit_file（约定写在 system 而非任务文本——
#     AGENTS.md 机制微缩：项目级约定与任务级指令分层，作业②的代码落点）
#   - 最终测试：通过——median([4,1,3,2]) == 2.5（偶数取中间两数平均）
#   - 若模型用 write_file 重写也没拦（任务引导非强制）——闭环达成即算通过
#   - 若模型绕开测试直接汇报：终检会暴露「仍失败」——这正是本实验的裁判设计
#   - 首跑翻车记：终检 subprocess 曾漏设 PYTHONIOENCODING——agent 轨迹里的 bash
#     返回正常「全部测试通过」，终检却乱码误判「仍失败」（Windows 管道 GBK）——
#     子进程读中文必须带编码环境变量，与 bash 工具同款处理
