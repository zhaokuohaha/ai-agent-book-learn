"""study_agent/memory.py — 学习者画像：判分记录的 append-only 日志 + 聚合视图

「懂你」的一半。复用第 3 章沉淀的 common/memory_store（Mem0 v3 思想
教学版）：每次判分是一条 episodic 记忆（key = "章:题号"，content =
correct/wrong），仅追加不覆盖——误删不可逆的教训。

active() 的「同 key 取最新」天然实现间隔复习的核心语义：**错题重答对
即转为已掌握**（最新记录覆盖旧状态），而 history() 保留完整时间线
（这题你错了几次才过的——时间推理的原料）。画像视图 = 对 active 记录
做确定性聚合（第 3 章：能确定性计算的别让模型算）。
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.memory_store import MemoryStore

PROFILE = Path(__file__).parent / "data" / "profile.json"


class LearnerProfile:
    """学习者画像：判分日志（MemoryStore）+ 聚合视图（确定性函数）"""

    def __init__(self, path: Path = PROFILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.store = MemoryStore(path)  # 构造即加载：跨会话的入口

    def record(self, chapter: str, num: int, ok: bool) -> None:
        """一次判分 = 一条 episodic 记忆，仅追加（覆盖语义交给 active 取最新）。
        微秒时间戳：同一秒内连错带重（quiz 常见）也能分出先后——同 ts 时
        active 取不到最新，间隔复习语义会翻车"""
        self.store.remember(f"{chapter}:q{num}", "correct" if ok else "wrong",
                            "episodic",
                            datetime.now().isoformat(timespec="microseconds"))

    def mastered(self, chapter: str, num: int) -> bool | None:
        """该题当前状态：True 已掌握 / False 未掌握 / None 没答过"""
        e = self.store.active().get(f"{chapter}:q{num}")
        return None if e is None else e.content == "correct"

    def wrong_questions(self) -> list[str]:
        """当前未掌握的题（间隔复习的出题优先池）"""
        return sorted(k for k, e in self.store.active().items() if e.content == "wrong")

    def render(self) -> str:
        """画像视图：按章聚合 active 记录——注入上下文 / CLI 展示两用"""
        by_ch: dict[str, list[str]] = {}
        for k, e in self.store.active().items():
            by_ch.setdefault(k.split(":")[0], []).append(e.content)
        if not by_ch:
            return "学习者画像：（暂无记录——先做一次自测）"
        lines = [f"  {ch}：答题 {len(v)} 次，未掌握 {v.count('wrong')} 题"
                 + (f"（{', '.join(self.wrong_questions()[:3])}…）"
                    if any(x == "wrong" for x in v) else "")
                 for ch, v in sorted(by_ch.items())]
        total = sum(len(v) for v in by_ch.values())
        wrong = sum(v.count("wrong") for v in by_ch.values())
        return f"学习者画像（active 视图，共 {total} 题、未掌握 {wrong}）：\n" + "\n".join(lines)


if __name__ == "__main__":
    print(LearnerProfile().render())
