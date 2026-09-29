"""chapter3/18.3_retrieval_metrics.py — RAG 基础（下）· 三：检索质量三指标

本实验新增的主概念：recall@k / MRR / nDCG——检索质量的度量衡。

三指标各答一个问题（直觉先行）
    recall@k："该找的找到了吗"——金标文档进前 k 的查询比例
    MRR："找到得够不够靠前"——第一个金标文档排名倒数的平均（第 1 名 1 分，
         第 10 名 0.1 分）
    nDCG："整个列表质量如何"——所有金标的位置都算数，排得越靠前贡献越大
         （按 1/log2(rank+1) 折损），再除以理想排序归一到 0~1
    对 RAG 最关键的是 recall@k：金标没进上下文，LLM 连用上它的机会都没有；
    进了上下文哪怕排第二，生成器照样能读（工业界还看"检索失败率"——
    正确信息未进 top-20 的查询比例）。

实验设计：人工标注的查询集（每题已知金标块）对三种检索器（稠密/BM25/混合）
    打分——把 18.1 的"混合更好"从直觉变成数字；也是未来调参（换嵌入模型/
    调 chunk 大小）的回归测试基线：改动前后各跑一遍，指标掉没掉一目了然。

运行：uv run python -X utf8 chapter3/18.3_retrieval_metrics.py（零 LLM 调用，本地嵌入）
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.retriever import BM25Index, DenseIndex, load_faq, rrf_fuse

# 标注查询集：(查询, 金标 doc_id 集合)——评测的前提是人工标注"正确答案在哪块"。
# q4 考同义改写（"包邮"→语料里只有"免运费"）；q5 金标只有退款块——退货块讲
# 流程不含时效，BM25 会被"退货"字面带偏（金标块对查询词零字面命中）
LABELED = [
    ("退款几天能到账", {0}),
    ("换货有什么限制", {1}),
    ("会员打几折", {2}),
    ("包邮有什么条件", {3}),
    ("退货的钱多久回来", {0}),
    ("怎么开发票", {4}),
]


def recall_at_k(ranking, gold, k=3):
    """该找的找到了吗：金标任一进前 k 即命中"""
    return 1.0 if gold & set(ranking[:k]) else 0.0


def mrr(ranking, gold):
    """找到得够不够靠前：第一个金标排名的倒数"""
    for r, i in enumerate(ranking, 1):
        if i in gold:
            return 1 / r
    return 0.0


def ndcg_at_k(ranking, gold, k=3):
    """整个列表质量：金标按 1/log2(rank+1) 折损累加 ÷ 理想值（金标全挤前排）"""
    dcg = sum(1 / math.log2(r + 1) for r, i in enumerate(ranking[:k], 1) if i in gold)
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return dcg / ideal


if __name__ == "__main__":
    chunks, topics = load_faq()
    dense, bm25 = DenseIndex(chunks), BM25Index(chunks)

    def dense_rank(q):
        return [i for _, i in dense.search(q, 5)]

    def bm25_rank(q):
        return [i for _, i in bm25.search(q, 5)]

    def hybrid(q):
        return rrf_fuse([dense_rank(q), bm25_rank(q)])

    print("=== 三种检索器 × 三指标（6 个标注查询的平均分） ===")
    print(f"  {'检索器':<10}{'recall@3':>9}{'MRR':>7}{'nDCG@3':>9}")
    for name, rank_fn in (("稠密", dense_rank), ("BM25", bm25_rank), ("混合 RRF", hybrid)):
        rs = [recall_at_k(rank_fn(q), g) for q, g in LABELED]
        ms = [mrr(rank_fn(q), g) for q, g in LABELED]
        ns = [ndcg_at_k(rank_fn(q), g) for q, g in LABELED]
        print(f"  {name:<10}{sum(rs) / len(rs):>9.2f}{sum(ms) / len(ms):>7.2f}{sum(ns) / len(ns):>9.2f}")

    print("\n=== 逐题看 top-1（谁在哪题翻车；括号内是该路 top-1 得分） ===")
    for q, gold in LABELED:
        d1, b1, h1 = dense_rank(q)[0], bm25_rank(q)[0], hybrid(q)[0]
        ds, bs = dense.search(q, 1)[0][0], bm25.search(q, 1)[0][0]
        mark = lambda i: "✓" if i in gold else "✗"
        print(f"  「{q}」金标={{{','.join(topics[g] for g in sorted(gold))}}}")
        print(f"    稠密→{topics[d1]}{mark(d1)}({ds:.2f})  BM25→{topics[b1]}{mark(b1)}({bs:.1f})  混合→{topics[h1]}{mark(h1)}")
    print("\n  → 读数（q5「退货的钱多久回来」是三路分野的教科书）：BM25 被'退货'字面带偏，")
    print("    金标被挤到第 5（recall=0）；稠密 top-1 命中（MRR 满分）；混合把金标拉回第 2：")
    print("    救回了 recall，但 top-1 被 BM25 的第一名票稀释（MRR 0.92→0.81）")
    print("  → 混合的价值在召回——RAG 的命门指标：金标进了上下文，LLM 才有机会用；")
    print("    代价是 top-1 精度被稀释，而这正是 18.2 重排的活——流水线各司其职：")
    print("    双路召回 → RRF 保 recall → 重排修 top-1（实测：重排把 RRF 第 3 的金标救回第 1）")
    print("  → 另两个真发现：q4「包邮」全员 top-1 翻车（同义改写盲区）；q6「开发票」")
    print("    BM25 得分 0.0 却'命中'——jieba 把'开发票'切成一个词，对不上语料的'发票'，")
    print("    全 0 瞎排靠并列顺序撞大运：中文 BM25 的命门在分词")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（零 LLM 调用，本地嵌入；复用 common/retriever 三检索器）：
#   1. 6 个标注查询（金标含同义改写题 q4、字面带偏题 q5）逐一跑三种检索器
#   2. 汇总表：recall@3 / MRR / nDCG@3 三指标平均分
#   3. 逐题行：三路 top-1 各自命中与否 + 得分（BM25 得分 0 = 瞎排，碰巧命中也算运气）
# 验证内容：
#   - q5「退货的钱多久回来」三路分野：BM25 recall@3=0（金标被"退货"字面
#     挤到第 5）；稠密 top-1 命中；混合救回 recall（金标第 2）但 top-1 被稀释
#     （MRR 0.92→0.81）——混合保召回、重排修 top-1 的分工实证
#   - q4「包邮」三路 top-1 全翻车（同义改写盲区）；q6 BM25 得分 0.0 靠并列
#     顺序撞大运（jieba 把"开发票"切一词，对不上"发票"）——分词是中文 BM25 命门
#   - 汇总表：混合 recall@3 = max(单路)，MRR 介于两路之间——指标把取舍量化了
# 双端验证（2026-09-29 用户本机重跑）：指标表与逐题行与首次运行逐位一致——
#   本实验全部为确定性计算（嵌入/BM25/RRF/指标），是三段流水线里可回归测试的部分
