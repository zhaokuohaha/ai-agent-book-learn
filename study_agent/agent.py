"""study_agent/agent.py — AI 学习伙伴：入口与组装层

study_agent 是贯穿性示例项目（方案见 README.md）：懂课本（knowledge）、
懂你（memory）、按计划陪你学、随叫随到。本文件是组装层——仓库里唯一
知道所有模块的地方，把 skills 装到交互循环上。

主循环当前是**临时回合制**（你问它答）——D36 课件论断「回合制是交互
约定」的活体现；D37 课件来了把它升级为事件驱动 core.py（事件队列/
任务句柄/抢占），此后 core 稳定不回改（新能力只进 skills/）。

指令：出题 [N] / 画像 / 题库 / 退出
运行：uv run python -X utf8 study_agent/agent.py（需 .env——判分用真实 LLM）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))               # study_agent/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))         # 仓库根（common）

from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL

from knowledge import collect_questions, render_bank
from memory import LearnerProfile
from skills.quiz import run_quiz

BANNER = """=== AI 学习伙伴（study_agent）===
指令：出题 [N]（默认 5）｜画像｜题库｜退出"""


def main() -> None:
    model = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL,
                       timeout=60, max_retries=3)
    profile = LearnerProfile()
    print(BANNER)
    while True:  # 临时回合制主循环（D37 升级为事件驱动 core.py）
        try:
            line = input("\n你> ").strip()
        except EOFError:
            print()  # 管道输入结束：等价退出（可无人值守验证）
            break
        if not line or line in ("退出", "q"):
            break
        if line.startswith("出题"):
            n = int(line.split()[1]) if len(line.split()) > 1 else 5
            run_quiz(model, profile, n)
        elif line == "画像":
            print(profile.render())
        elif line == "题库":
            bank = collect_questions()
            print(f"  共 {len(bank)} 题｜{render_bank(bank)}｜题库随章节 README 自动更新")
        else:
            print("  指令：出题 [N] / 画像 / 题库 / 退出")


if __name__ == "__main__":
    main()
