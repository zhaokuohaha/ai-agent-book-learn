"""chapter3/20.2_contextual_prefix.py — 上下文感知检索：给 chunk 补前缀锚回语境

本实验新增的主概念：上下文感知检索（Anthropic 的招，作业②手写前缀）。

解决 D17 埋的伏笔（问题先行）
    "该公司 Q2 收入增长 3%"——单独拿出来，谁知道"该公司"是谁？指代断裂
    导致：稠密路向量语义失真（"该公司"的向量是噪音）、稀疏路缺关键词
    （查询"ACME"对不上任何词）。

做法（三层讲解）
    直觉：把便签从撕下来的状态贴回它原来的那一页。
    机制：索引期给每个 chunk 生成一句前缀摘要（如"[节选自 ACME 2025 Q2
          财报·关键业绩指标章]"），前缀+原文拼接后入库——一句话同时增强
          两路：稠密路获得语义背景，稀疏路多出"ACME""手机"等可精确匹配词。
    代码落点：本实验按作业②手写前缀（零 LLM）；生产用 LLM 批量生成
          （一次性索引投入，Anthropic 靠 prompt caching 压到约 $1/百万 token）。
    效果：Anthropic 数据 BM25+前缀检索失败率降 49%，再加 reranker 降 67%。

易混对照（Part 3）：本节的上下文感知检索发生在【索引期、对知识库 chunk、
    做加法（补背景）】；第 2 章的上下文感知压缩发生在【运行期、对对话
    历史、做减法（裁剪无关）】——名字像双胞胎，方向恰好相反。

运行：uv run python -X utf8 chapter3/20.2_contextual_prefix.py（零 LLM 调用，本地嵌入）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.retriever import BM25Index, DenseIndex

# 3 个指代模糊 chunk + 手写前缀（作业②）；2 个干扰项（含同域竞争信息）
# 干扰项不加前缀——真实库里"有背景的"与"没背景的"chunk 共存
AMBIGUOUS = [
    ("该公司 2025 年 Q2 收入同比增长 3%，净利润率 12%。",
     "[节选自 ACME 科技 2025 年第二季度财报·关键业绩指标章]"),
    ("该产品支持 120W 快充，续航可达 10 小时。",
     "[节选自星辰 X9 手机官方规格页·电池与充电节]"),
    ("该条款规定情节较轻的处三年以下有期徒刑。",
     "[节选自刑法第 233 条过失致人死亡罪·量刑档次说明]"),
]
DISTRACTORS = [
    "晨光文具 2025 年年报：营业收入 58 亿元，同比增长 9%。",
    "远航笔记本电脑规格：支持 65W 快充，续航长达 14 小时。",
]

QUERIES = ["ACME 科技的收入增长", "哪款手机续航 10 小时", "过失致人死亡情节较轻怎么判"]


def build(docs_pairs, with_prefix):
    """两套库：bare（原 chunk）vs prefixed（前缀+chunk）——干扰项两库相同"""
    bare = [c for c, _ in docs_pairs] + DISTRACTORS
    prefixed = [p + "\n" + c if with_prefix else c for c, p in docs_pairs] + DISTRACTORS
    return bare, prefixed


if __name__ == "__main__":
    print("=== Part 0：问题先行——D17 的伏笔 ===")
    print(f"  「{AMBIGUOUS[0][0]}」——'该公司'是谁？指代断裂：稠密路向量失真、")
    print("  BM25 缺关键词。解法：索引期补一句前缀把它锚回原文档语境\n")

    bare, prefixed = build(AMBIGUOUS, with_prefix=True)
    bare_idx, pre_idx = DenseIndex(bare), DenseIndex(prefixed)
    bare_bm, pre_bm = BM25Index(bare), BM25Index(prefixed)

    print("=== Part 1：稠密路对比（有无前缀的余弦得分，top-1） ===")
    for q in QUERIES:
        b = bare_idx.search(q, 1)[0]
        p = pre_idx.search(q, 1)[0]
        print(f"  「{q}」")
        print(f"    无前缀：{b[0]:+.2f}  {bare[b[1]][:22]}…")
        print(f"    有前缀：{p[0]:+.2f}  {prefixed[p[1]][:28]}…")

    print("\n=== Part 2：稀疏路对比（BM25 得分——前缀贡献可精确匹配的词） ===")
    for q in QUERIES:
        b = bare_bm.search(q, 1)[0]
        p = pre_bm.search(q, 1)[0]
        print(f"  「{q}」")
        print(f"    无前缀：{b[0]:.2f}  {bare[b[1]][:22]}…")
        print(f"    有前缀：{p[0]:.2f}  {prefixed[p[1]][:28]}…")

    print("\n  → 读数（以实际输出为准）：稀疏路全线翻 2-3 倍（0.70→2.63、1.15→3.93）——")
    print("    前缀贡献 ACME/手机/过失致人死亡等可精确匹配词；稠密路两升一平")
    print("    （手机查询 0.67→0.64 但命中不变）——bge 对模糊句有语义底子，")
    print("    前缀对稀疏路的拉动更陡，正是 Anthropic'BM25+前缀降 49%'的机理；")
    print("    再加 reranker 降 67%；成本靠 prompt caching 约 $1/百万 token——")
    print("    索引一次投入，回报率极高")

    print("\n=== Part 3：易混对照——两个'上下文感知' ===")
    print("    上下文感知检索（本节）：索引期 | 知识库 chunk | 加法（补背景前缀）")
    print("    上下文感知压缩（D14）：运行期 | 对话历史     | 减法（裁剪无关轮次）")
    print("    → 名字像双胞胎，时机/对象/方向都相反；一个富化静态知识，一个瘦身动态会话")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（零 LLM 调用，本地嵌入 + jieba 分词）：
#   1. 同一批 chunk 建两套库：bare（原样）与 prefixed（前缀+原文），
#      干扰项两库一致（同域竞争信息：晨光年报 vs ACME 财报）
#   2. Part 1/2：三个查询在两套库上分别跑稠密与 BM25 的 top-1 得分对比
#   3. Part 3：与 D14 上下文感知压缩的易混对照（定性）
# 验证内容（得分随嵌入版本浮动，以实际输出为准）：
#   - 查询"ACME 的收入增长"：无前缀 BM25 靠部分词命中（约 0.7）但区分不开
#     晨光干扰句；有前缀靠"ACME"精确词翻倍（实测 2.63）——专名增益最大
#   - 查询"哪款手机续航"：无前缀 BM25 无"手机"词可对仍靠"续航/10"命中；
#     有前缀叠加"手机"精确词升到 2.63；稠密路命中不变得分微降——前缀
#     对稀疏路拉动更陡，稠密路主要拿语义背景（ACME/刑法查询 +0.13/+0.17）
#   - 两路对比印证 Anthropic 机理：BM25+前缀失败率降 49%（稀疏路受益最大）
