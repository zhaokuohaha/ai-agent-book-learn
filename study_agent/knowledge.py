"""study_agent/knowledge.py — 课本知识库：把各章 README 变成可出题的结构化资产

study_agent 是「AI 学习伙伴」的贯穿性示例项目（方案见 README.md）：
它懂你的课本（各章 README）、懂你（学习者画像）、按计划陪你学、随叫随到。
本模块是「懂课本」的一半——解析 chapterN/README.md 的自测题段，
把「N. 题干？（参考答案）」格式抽成 Question 对象供 quiz 出题。

设计：解析是确定性纯函数（能确定性计算的别让模型算——第 3 章约定），
题库在进程内构建、零落盘（课本本身就是 source of truth，出题时现读）。
"""

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEAD = re.compile(r"^(\d+)\.\s+(.+)")


@dataclass
class Question:
    """一道自测题：章、题号、题干、参考答案（判分时对照，不要求逐字）"""

    chapter: str   # 如 "chapter5"
    num: int       # 章内题号
    stem: str      # 题干（出题时展示）
    answer: str    # 参考答案（判分时用）


def parse_line(ln: str) -> Question | None:
    """一行 → 一道题。用「最后一个？」定位题干/答案分界——题干可含术语
    括号与多个问句、答案可含嵌套括号，两种复杂形态都能正确切分
    （非贪婪正则截术语括号、贪婪正则错切嵌套答案——都被这一个定位法取代）"""
    p = ln.rfind("？") + 1
    if p == 0:
        return None
    rest = ln[p:].strip()
    if not (rest.startswith("（") and rest.endswith("）")):
        return None
    if not (m := HEAD.match(ln[:p])):
        return None
    return Question("", int(m.group(1)), m.group(2).strip(), rest[1:-1])


def collect_questions() -> list[Question]:
    """扫全部章节 README 的自测段抽题——课本即题库（新增章节自动入库）"""
    bank: list[Question] = []
    for readme in sorted(ROOT.glob("chapter*/README.md")):
        chapter = readme.parent.name
        in_quiz = False
        for ln in readme.read_text(encoding="utf-8").splitlines():
            if ln.startswith("## "):
                in_quiz = "自测" in ln
                continue
            if in_quiz and (q := parse_line(ln.strip())):
                bank.append(Question(chapter, q.num, q.stem, q.answer))
    return bank


def render_bank(bank: list[Question]) -> str:
    """题库概览（按章计数）——「课本有多少弹药」一目了然"""
    by_ch: dict[str, int] = {}
    for q in bank:
        by_ch[q.chapter] = by_ch.get(q.chapter, 0) + 1
    return "  ".join(f"{ch}:{n} 题" for ch, n in sorted(by_ch.items()))


if __name__ == "__main__":
    bank = collect_questions()
    print(f"课本题库：共 {len(bank)} 题｜{render_bank(bank)}")
    print(f"样例：[{bank[0].chapter} q{bank[0].num}] {bank[0].stem}")
