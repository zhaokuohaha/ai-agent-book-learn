"""chapter3/18.2_rerank_generate.py — RAG 基础（下）· 二：LLM 重排与"检索→注入→生成"闭环

本实验新增的主概念：重排序 rerank（伴生：注入生成——三段流程收口，作业 ②）。

为什么融合之后还要精排
    18.1 实测了 RRF 的天花板：配送（稠密2+BM25四）与退货（稠密4+BM25二）
    得分对称相等，近并列分不开——排名投票回答不了"这段话到底能不能回答
    这个问题"。精排器逐对细看，给每个 (query, doc) 对一个真分数。

Bi-Encoder vs Cross-Encoder（三层讲解）
    直觉：猎头筛简历 vs 面试官面谈——前者把人和岗位各自归档再比对（快而
    粗，海量初筛），后者把人和岗位放在一起聊（慢而准，只对小池子）。
    机制：
        Bi:    query ─编码→ 向量 ┐
                                   ├ 余弦比距离（两者永不见面）  O(1) 查询
               doc   ─编码→ 向量 ┘
        Cross: [query ⊕ doc] → 一个模型逐词交互 → 相关性分数      O(N) 推理
    代码落点：本实验用 LLM 充当逐对打分器（LLM-as-reranker，工业真实一路，
    零下载即跑）；课程原配方 CrossEncoder（如 BAAI/bge-reranker-base）是同
    思想的小模型：把 query 和 doc 拼接送入、直接输出相关性分。

注入与生成（三段流程收口：检索 → 注入 → 生成）
    检索（18.1 双路+RRF+本实验精排）→ 注入（top 片段拼进 system——知识是
    开发者侧的稳定信息，D11 的角色信任体系）→ 生成（LLM 只依据片段回答并
    标注来源，不编造——检索不到就明说）。

运行：uv run python -X utf8 chapter3/18.2_rerank_generate.py（需 .env，约 4 次真实调用）
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL
from common.retriever import BM25Index, DenseIndex, load_faq, rrf_fuse

QUESTION = "退货的话，钱多久能回来？"


def llm_rerank(model, question, candidates, chunks):
    """LLM 逐对精判：每个 (query, doc) 对单独打一次分——Cross-Encoder 思想，
    打分器换成 LLM。返回按相关性降序的 (分数, doc_id) 列表"""
    scored = []
    for i in candidates:
        resp = model.invoke([
            SystemMessage(content=(
                "你是检索结果重排器。判断【文档】能在多大程度上回答【问题】，"
                "只输出一个 0 到 10 的整数：10=直接包含答案，5=部分相关，0=无关。"
            )),
            HumanMessage(content=f"【问题】{question}\n【文档】{chunks[i]}"),
        ])
        score = int(re.findall(r"\d+", resp.content)[0])  # 模型偶尔带文案，取第一个数字
        scored.append((score, i))
    return sorted(scored, reverse=True)


if __name__ == "__main__":
    model = ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0,
    )
    chunks, topics = load_faq()
    dense, bm25 = DenseIndex(chunks), BM25Index(chunks)

    print("=== Part 0：问题先行——RRF 投票之后，顺序仍然'大概齐' ===")
    d_rank = [i for _, i in dense.search(QUESTION, 5)]
    b_rank = [i for _, i in bm25.search(QUESTION, 5)]
    fused = rrf_fuse([d_rank, b_rank])[:3]  # 融合取 top-3 作为精排候选池（流水线：粗筛快，精排小）
    print(f"问题「{QUESTION}」")
    for r, i in enumerate(fused, 1):
        print(f"  候选 {r}：[{topics[i]}] {chunks[i][:24]}…")
    print("  → 投票只看共识多少，没逐对细看'能不能回答'——谁真谁假？上精排")

    print("\n=== Part 1：LLM 重排——逐对精判（Cross-Encoder 思想） ===")
    print("  停一停：精排为什么只对 top 候选做，不对全库逐对打分？")
    print("  答：逐对打分 = 每个文档都要过一次模型（O(N) 推理），百万文档就炸了；")
    print("      双编码器比向量是 O(1) 查询——所以流水线天生'粗筛快、精排小'")
    reranked = llm_rerank(model, QUESTION, fused, chunks)
    for score, i in reranked:
        print(f"  {score:>2} 分  [{topics[i]}]")
    print(f"  → 重排后第一名：[{topics[reranked[0][1]]}]——它才真正回答'钱多久回来'")

    print("\n=== Part 2：注入生成——三段流程收口（检索 → 注入 → 生成） ===")
    top2 = reranked[:2]
    refs = "\n\n".join(f"【参考资料 {n}】\n{chunks[i]}" for n, (_, i) in enumerate(top2, 1))
    answer = model.invoke([
        SystemMessage(content=(
            "你是电商客服助理。仅依据下方参考资料回答用户问题；资料不足以回答就"
            "明说，不要编造。回答末尾用（依据：资料 N）标注来源。\n\n" + refs
        )),
        HumanMessage(content=QUESTION),
    ]).content
    print(f"  问：{QUESTION}")
    print(f"  答：{answer}")
    print("\n  → 口诀收口：检索（双路+RRF+精排）→ 注入（top 片段进 system）→ 生成")
    print("    （基于片段、标注来源、不足则明说）——完整 RAG 闭环")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（约 4 次真实 LLM 调用：重排 3 + 生成 1；嵌入与 BM25 本地）：
#   1. 双路检索 + RRF 融合取 top-3 候选池（复用 common/retriever）
#   2. llm_rerank：候选逐对送 LLM 打 0-10 分（Cross-Encoder 思想），降序
#   3. 重排 top-2 拼进 system 参考资料 → LLM 生成回答（要求标注来源）
# 验证内容：
#   - Part 0 候选池含退款块（金标）与退货块（部分相关）——RRF 分不出先后
#   - Part 1 重排后退款块应居首（直接回答"钱多久回来"）；本次实测金标块在
#     RRF 里只列第三，被重排救回第一；退货块部分相关但不含答案要点，
#     按"能否回答"的标准可能只得低分——验收看相对顺序不看具体分值
#   - Part 2 回答含"3-5 个工作日/原路退回"要点并带（依据：资料 N）标注；
#     模型输出动态——验收看要点与标注，不断言固定文案
# 双端验证（2026-09-29 用户本机重跑）：重排 10/0/0 与候选池顺序一致（确定性）；
#   生成文案两次措辞微变（"另需在…" vs "另外，订单签收后…"）但要点与来源
#   标注不变——LLM 输出验收看轨迹不看文案的活例
