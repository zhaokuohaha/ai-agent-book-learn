"""study_agent/skills/daily_review.py — 定时复习提醒（set_timer 循环定时器）

37 课件三类事件触发工具之首：set_timer——处理依赖物理时间的事件。
循环定时器线程到点 emit 事件（来源=timer，不是 user——注入防线：
提醒内容不会被当成用户指令）。提醒内容自带上下文（未掌握题数与
优先重出清单），减少唤醒后的二次查询（课件设计要点）。

演示周期与真实周期：参数 due_seconds 控制——CLI 会话演示用短周期
（如 120s），真实产品是每日 09:00；到点时 Agent 正忙（如 quiz 中），
事件在队列里排队等安全点批量消费——队列式策略的日常形态。
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # study_agent/

from core import Event, EventLoop
from memory import LearnerProfile


def start(due_seconds: float, loop: EventLoop, profile: LearnerProfile) -> None:
    """启动循环定时器线程（daemon：随主进程退出）"""

    def tick():
        while True:
            time.sleep(due_seconds)
            wrong = profile.wrong_questions()
            msg = f"复习时间到：当前未掌握 {len(wrong)} 题"
            if wrong:
                msg += f"（下轮出题优先重出：{', '.join(wrong[:3])}）"
            loop.emit(Event(source="timer", channel="cron", content=msg,
                            context="间隔复习：错题重答对即出池"))

    threading.Thread(target=tick, daemon=True).start()
