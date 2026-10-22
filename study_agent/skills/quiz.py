"""study_agent/skills/quiz.py — 自测问答：抽题 → 问答 → LLM 判分 → 记画像

study_agent 的第一个能力包（D36 骨架落地，全部用已学概念搭建）：
    抽题   画像加权——未掌握的错题优先重出（间隔复习雏形：active 取最新，
           答对即出池）、没答过的次之随机、已掌握不出
    判分   LLM 对照参考答案判「意思对即可」（不逐字）——结构化 JSON
           输出（33.1 自举 extract_json 的同款场景：deepseek 爱包 ```json，
           剥壳解析沿用）
    记录   每次判分进 LearnerProfile（episodic 仅追加）

判分是本 skill 唯一的 LLM 环节——抽题与记录全是确定性函数（能确定性
计算的别让模型算）。input_fn 参数化让交互可注入（可测性，非脚手架）。
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))        # study_agent/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # 仓库根（common）

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from knowledge import Question, collect_questions
from memory import LearnerProfile

JUDGE = """你是学习自测的判分员。先判断学生回答的意图：
- 若回答表达的是中止/放弃本轮自测（如「停」「不答了」「算了」「跳过」）而非
  在答题，输出 action="stop"（中止意图是开放语义——不能用关键词匹配，
  37.1 棺材钉同款问题）
- 否则 action="answer"，并判分：意思对即可，不要求逐字复述
题目：{stem}
参考答案：{answer}
学生回答：{reply}
只输出 JSON：{{"action": "answer"或"stop", "verdict": "correct"或"wrong", "explain": "一句话"}}
（action=stop 时 verdict 填 "wrong"，explain 简述其中止意图）"""


def pick(bank: list[Question], profile: LearnerProfile, n: int) -> list[Question]:
    """画像加权抽题：未掌握优先 → 没答过随机 → 已掌握不出"""
    wrong = [q for q in bank if profile.mastered(q.chapter, q.num) is False]
    fresh = [q for q in bank if profile.mastered(q.chapter, q.num) is None]
    random.shuffle(fresh)
    return (wrong + fresh)[:n]  # 错题全部优先（重出上限 = n）


def judge(model: ChatOpenAI, q: Question, reply: str) -> tuple[bool, str, bool]:
    """LLM 判分 + 中止意图路由（一次调用两用）：返回 (判对?, 讲解, 中止?)
    ——中止判定并入判分调用，成本零增加；语义路由而非关键词匹配"""
    raw = model.invoke([HumanMessage(content=JUDGE.format(stem=q.stem, answer=q.answer, reply=reply))]).content
    m = re.search(r"\{.*\}", raw, re.S)
    v = json.loads(m.group(0))
    return v["verdict"] == "correct", v.get("explain", ""), v.get("action") == "stop"


def run_quiz(model, profile: LearnerProfile, n: int, input_fn=input, print_fn=print) -> None:
    """一轮自测：抽题 → 逐题问答判分 → 记录 → 战绩。
    题间是安全点：回答表达中止意图（语义路由，并入判分调用）则中止本轮
    ——取消式打断的 CLI 诚实实现：已完成部分照常入账，不伪造未答的题"""
    bank = collect_questions()
    if not (qs := pick(bank, profile, n)):
        print_fn("  题库已全部掌握——去学新章节吧（或等定时复习提醒）")
        return
    hit = 0
    for i, q in enumerate(qs, 1):
        print_fn(f"\n[{i}/{len(qs)}]（{q.chapter} q{q.num}）{q.stem}")
        reply = input_fn("你的回答> ").strip()
        if reply:
            ok, why, stop = judge(model, q, reply)
            if stop:
                print_fn(f"  已中止（{why}；安全点：已完成 {hit}/{i - 1} 题照常入账，其余不打分）")
                return
        else:
            ok, why = False, "空答按错误计——画像里留下这个坑（下轮重出）"
        hit += ok
        profile.record(q.chapter, q.num, ok)
        print_fn(f"  {'✓ 对' if ok else '✗ 错'}——{why}")
    print_fn(f"\n本轮战绩：{hit}/{len(qs)}"
             + (f"｜未掌握共 {len(profile.wrong_questions())} 题（下轮优先重出）"
                if profile.wrong_questions() else "｜当前无未掌握题"))
