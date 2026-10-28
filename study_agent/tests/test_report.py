"""study_agent/tests/test_report.py — 周报统计的确定性回归

构造带新旧时间戳的画像数据（直接 store.remember 造 8 天前的旧记录），
断言：本周过滤正确（旧数据不计入）、按日聚合正确、未掌握池正确、
渲染不炸（空数据/正常数据两路）。LLM 生成侧看轨迹不在此测（动态输出）。
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))        # study_agent/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # 仓库根

from memory import LearnerProfile
from skills.weekly_report import render_stats, weekly_stats


def make_profile() -> LearnerProfile:
    """本周 2 对 1 错 + 8 天前 1 条旧记录（不应计入周报）"""
    p = LearnerProfile(Path(tempfile.mkdtemp()) / "profile.json")
    now = datetime.now().isoformat(timespec="microseconds")
    old = (datetime.now() - timedelta(days=8)).isoformat()
    p.store.remember("chapter6:q35", "correct", "episodic", now)
    p.store.remember("chapter6:q36", "wrong", "episodic", now)
    p.store.remember("chapter5:q1", "correct", "episodic", old)  # 8 天前——不进本周
    return p


if __name__ == "__main__":
    p = make_profile()
    s = weekly_stats(p)
    assert s["answered"] == 2, f"本周应 2 次，实际 {s['answered']}"
    assert s["correct"] == 1, "正确应 1 次"
    assert len(s["by_day"]) == 1 and sum(s["by_day"].values()) == 2, "按日聚合应只含今天"
    assert s["wrong_now"] == ["chapter6:q36"], "未掌握应含 q36"
    text = render_stats(s)
    assert "2 次" in text and "50%" in text and "chapter6:q36" in text

    empty = weekly_stats(LearnerProfile(Path(tempfile.mkdtemp()) / "p.json"))
    assert empty["answered"] == 0 and "—" in render_stats(empty), "空数据渲染应含占位符"

    print("全部通过：周报统计（本周过滤/按日聚合/未掌握池/空数据渲染）正确")
