"""study_agent/agent.py — AI 学习伙伴：入口与组装层

study_agent 是贯穿性示例项目（方案见 README.md）：懂课本（knowledge）、
懂你（memory）、按计划陪你学（定时复习）、随叫随到（事件响应）。

D37 升级：主循环从临时回合制换成正式事件驱动 core.EventLoop——
D36 论断「回合制是交互约定」的反转落地：用户输入不再是主循环的
input()，而是输入线程 emit 的 Event（与定时器事件同一条队列、同一种
信封）；CLI 提示符由输入线程打印（队列阻塞时提醒仍能到达）。

组装（依赖倒置：core 不知道任何 skill，skill 向 core 注册）：
    loop.on("user", handle_user)     ← quiz 等指令分发
    loop.on("timer", print_reminder) ← daily_review 的提醒事件

指令：出题 [N] / 画像 / 题库 / 讲解 <关键词> / 周报 / 退出；quiz 中想中止直接说（语义路由，如「不答了」）；周报句柄式后台生成（37.2 幕二同构）。
运行：uv run python -X utf8 study_agent/agent.py [定时提醒周期秒数，默认 300]
（需 .env——判分用真实 LLM）
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))               # study_agent/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))         # 仓库根（common）

from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL

from core import Event, EventLoop
from knowledge import collect_questions, render_bank
from memory import LearnerProfile
from skills import daily_review
from skills.diagram_explain import explain_block
from skills.quiz import run_quiz
from skills.weekly_report import start_report

BANNER = """=== AI 学习伙伴（study_agent·事件驱动版）===
指令：出题 [N]（默认 5）｜画像｜题库｜讲解 <关键词>｜周报｜退出｜（quiz 中想中止直接说，如「不答了」）"""


def main() -> None:
    due = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0  # 定时提醒周期（秒）
    model = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL,
                       timeout=60, max_retries=3)
    profile = LearnerProfile()
    loop = EventLoop()

    def handle_user(ev: Event) -> None:
        line = ev.content
        if line.startswith("出题"):
            n = int(line.split()[1]) if len(line.split()) > 1 else 5
            run_quiz(model, profile, n)
        elif line == "画像":
            print(profile.render())
        elif line == "题库":
            bank = collect_questions()
            print(f"  共 {len(bank)} 题｜{render_bank(bank)}｜题库随章节 README 自动更新")
        elif line.startswith("讲解"):
            explain_block(model, line[2:].strip())
        elif line == "周报":
            print(f"  {start_report(model, profile, loop)}")  # 句柄式：立即返回，报告作为事件送达
        else:
            print("  指令：出题 [N] / 画像 / 题库 / 讲解 <关键词> / 周报 / 退出")

    def print_reminder(ev: Event) -> None:
        print(f"\n[定时提醒] {ev.content}\n你> ", end="", flush=True)

    def print_report(ev: Event) -> None:
        print(f"\n[周报已生成]\n{ev.content}\n你> ", end="", flush=True)

    loop.on("user", handle_user)
    loop.on("timer", print_reminder)
    loop.on("report", print_report)  # 后台生成完成的事件送达
    daily_review.start(due, loop, profile)

    def cli_input():
        """输入线程：CLI 输入 → 统一事件（与定时器同一条队列）"""
        while True:
            try:
                line = input("\n你> ").strip()
            except EOFError:
                loop.emit(Event("system", "cli", "quit"))
                return
            if not line or line in ("退出", "q"):
                loop.emit(Event("system", "cli", "quit"))
                return
            loop.emit(Event("user", "cli", line))

    print(BANNER)
    threading.Thread(target=cli_input, daemon=True).start()
    loop.run()  # 事件驱动主循环：批量消费 → 分发 handler → quit 退出


if __name__ == "__main__":
    main()
