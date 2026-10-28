"""study_agent/skills/weekly_report.py — 学习周报：先发起后收尾

D42 能力包（W6 收官）：「动作跨回合，先发起后收尾」主线在 study_agent 的
真实落地——指令「周报」立即返回句柄式提示（37.2 幕二 initiate 同构），
后台线程完成「确定性统计 → LLM 总结建议 → emit 事件」，报告作为事件
送达打印——生成期间用户可继续出题（后台长任务不打断前台交互，快慢分
离的最小版）。

分工同全仓约定：统计确定性（近 7 天判分聚合——能确定性计算的别让模型
算），总结与建议交给 LLM（不编造数据里没有的事实）。
"""

import threading
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage

from core import Event
from memory import LearnerProfile

ADVICE = """你是学习伙伴的周报撰写员。基于下面的确定性统计数据，写一份简短学习周报：
- 一段总结：本周学习面貌（节奏、覆盖面、正确率）
- 2-3 条具体建议：针对未掌握题目和正确率，给出下周可执行的复习动作
语气鼓励但诚实，不得编造数据里没有的事实。

统计数据：
{stats}
"""

_inflight = threading.Event()  # 防并发重复生成（生产正解：单飞标志）


def weekly_stats(profile: LearnerProfile, days: int = 7) -> dict:
    """近 N 天判分统计（确定性）：答题数/正确数/按日分布/当前未掌握"""
    since = (datetime.now() - timedelta(days=days)).isoformat()
    recent = [e for e in profile.store.entries if e.ts >= since]
    by_day: dict[str, int] = {}
    for e in recent:
        by_day[e.ts[:10]] = by_day.get(e.ts[:10], 0) + 1
    return {"answered": len(recent), "correct": sum(e.content == "correct" for e in recent),
            "by_day": by_day, "wrong_now": profile.wrong_questions()}


def render_stats(s: dict) -> str:
    days = "、".join(f"{d} {n} 题" for d, n in sorted(s["by_day"].items())) or "—"
    rate = f"{s['correct'] / s['answered']:.0%}" if s["answered"] else "无记录"
    return (f"近 7 天答题 {s['answered']} 次（{days}），正确 {s['correct']} 次"
            f"（正确率 {rate}）；当前未掌握 {len(s['wrong_now'])} 题"
            + (f"：{', '.join(s['wrong_now'][:8])}" if s["wrong_now"] else ""))


def start_report(model, profile: LearnerProfile, loop) -> str:
    """发起周报生成——立即返回（句柄式），后台完成后 emit 事件送达。
    单飞标志：生成中再喊「周报」会被告知稍候，不并发重复生成。"""
    if _inflight.is_set():
        return "周报还在生成中——完成会自动送达，稍候"
    _inflight.set()

    def work():
        try:
            stats = weekly_stats(profile)
            text = model.invoke([HumanMessage(content=ADVICE.format(
                stats=render_stats(stats)))]).content
            loop.emit(Event("report", "cron", text))
        finally:
            _inflight.clear()

    threading.Thread(target=work, daemon=True).start()
    return "周报生成中（后台进行，可继续出题）——完成后自动送达"
