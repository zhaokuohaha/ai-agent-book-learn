"""study_agent/tests/test_knowledge.py — 知识库与画像的确定性回归

保护对象：README 自测题解析（题库是 quiz 的地基——解析坏了出题就错）
与画像聚合（间隔复习的语义地基）。纯函数零 LLM，秒级可跑。
每章新增自测题后跑一次，确保格式兼容没被破坏。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))        # study_agent/

from knowledge import collect_questions
from memory import LearnerProfile, PROFILE

# ---- 1. 题库解析：格式正确、规模合理、章内题号唯一 ----
bank = collect_questions()
assert len(bank) >= 70, f"题库过小：{len(bank)}（各章 README 自测段解析可能失效）"
for q in bank:
    assert q.stem and q.answer, f"空题干/空答案：{q}"
    assert q.stem.endswith(("？", "?", "。")), f"题干非问句形态（答案嵌套括号错切）：{q.stem}"
    assert q.answer and not q.answer.endswith("（"), f"答案异常：{q.answer[:30]}"
seen: set[tuple[str, int]] = set()
for q in bank:
    assert (q.chapter, q.num) not in seen, f"章内题号重复：{q.chapter} q{q.num}"
    seen.add((q.chapter, q.num))
assert len({q.chapter for q in bank}) >= 5, "应覆盖至少 5 个章节"

# ---- 2. 画像语义：记录 → active 取最新 → 错题重答对即出池 ----
test_path = PROFILE.parent / "_test_profile.json"
test_path.unlink(missing_ok=True)
p = LearnerProfile(test_path)
assert p.mastered("chapter6", 99) is None            # 没答过
p.record("chapter6", 99, False)
assert p.mastered("chapter6", 99) is False           # 答错 → 未掌握
assert p.wrong_questions() == ["chapter6:q99"]       # 进入错题池
p.record("chapter6", 99, True)
assert p.mastered("chapter6", 99) is True            # 重答对 → 已掌握（间隔复习语义）
assert p.wrong_questions() == []                     # 出池
assert len(p.store.history("chapter6:q99")) == 2     # 时间线保留完整证据
test_path.unlink()
print(f"全部通过：题库 {len(bank)} 题（{len(seen)} 章内题号无重复）、画像语义（错→重答对→出池）正确")
