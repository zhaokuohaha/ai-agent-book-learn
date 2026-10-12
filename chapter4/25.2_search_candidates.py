"""chapter4/25.2_search_candidates.py — 搜索类量控：结构化候选列表 + 分页游标

本实验新增的主概念是「搜索结果的按需披露」——防上下文爆炸三板斧的前两斧。
承接 24.1（入口侧：工具 schema 按需披露）与 25.1（出口侧·读取类：
offset/limit + 显式截断）——本实验补搜索类的出口形状：返回结构化候选
列表（编号+标题+摘要+指针），不拼接全文；结果多时用 cursor 分页，注明
总数与翻页方式，由 Agent 自主决定是否继续（三板斧①②）。

为什么不返回全文：一次命中 5 篇全文，Agent 往往只深读 1 篇——其余 4 篇
白烧窗口。真实网页单页 3k-30k 字符、摘要 ~160 字符，比例一到两个数量级。
候选列表就是搜索结果页：先浏览货架，再取一件。

指针是第二跳的入口：网页搜索给 URL、文档站给 read_doc(标题)——Agent
浏览候选后决定深读谁，深读动作由 25.1 的 offset/limit 完成；两斧合璧即
Cursor 的 search→read 两段式（24.1「索引先行、详情按需」的同构回响）。

运行：uv run python -X utf8 chapter4/25.2_search_candidates.py（纯本地，不烧 API）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.retriever import BM25Index

# ---- 迷你文档站：6 篇本仓库学习笔记（搜索类感知工具的语料） ----

DOCS = {
    "KV Cache：前缀复用的省钱术": """长上下文的 token 开销贵在每次请求都重算全量。KV Cache 的思路：
prompt 前缀逐字节相同，注意力的中间结果（K/V）就能跨请求复用——静态前缀
（system+tools）越稳定，命中比例越高，账单越便宜。代价是工具集一变，
前缀从变动点起全部失效重算，这就是「tools 只追加不重排」的由来：
改动点越靠前，重算越长；缓存过期同样触发重算——稳定的前缀结构是
省钱的地基。""",
    "上下文压缩：窗口快满时的抢救": """会话越聊越长逼近窗口上限，把旧消息 summarize 成摘要替换原文——
细节有损但主线保住，token 开销随之降下来。压缩时机是门艺术：太早丢
细节，后面的问题答不上；太晚窗口已满，压缩本身也要占空间。生产常在
窗口用量到 80% 时触发。摘要必须保留「关键决策与未决事项」——原文一旦
被替换就再也回不来，有损压缩不可逆。""",
    "BM25：字面匹配的倒排索引": """稀疏检索老将：词越稀有越值钱（IDF），词频有边际递减（k1 饱和），
长文档做长度归一化（b）。搜「HTTP-403」这类精确术语时完胜语义检索，
搜「退货要多久」就露咎——同义改写它全看不见。与稠密检索互补：字面
匹配与语义匹配各管一半，RRF 排名融合把两路投票合起来——序数可比、
基数不可比，融合只用排名不用原始分。""",
    "GraphRAG：实体链接+邻域扩展": """先从文档抽实体建图（谁与谁有什么关系），查询时先做实体链接（把
问句里的名字挂到图节点上），再沿边扩展邻域取相关内容——问「与 X 相关
的东西」时，图结构比扁平 chunk 的相似度靠谱得多。微软 GraphRAG 的
local search 就是这个骨架。建图成本高（LLM 抽取），适合多跳关系
问题，不是 RAG 的默认选项。""",
    "MCP：工具生态的 JDBC": """各家框架定义工具的方式不同，开发者被迫逐家适配。MCP 用一套
JSON-RPC 协议统一接入：一次开发，所有客户端可用——就像 JDBC 统一了
数据库驱动。握手先协商协议版本与能力，再干活。代价是全部工具 schema
进静态前缀吃 token，工具一多既选错又烧钱——交给按需披露（目录/
发现/Skills 目录）解决，那是规模化一天的课题。""",
    "工具发现：discover 元工具": """工具上千个全量注入既选错又烧钱。主动发现模式：常驻工具集缩到最小
——只留一个 discover_tools 元工具，Agent 执行中自己声明能力缺口，两层
语义路由（先选服务器再选工具）命中后 schema 动态注入、注册进可调用集；
未命中明确返回 404。MCP-Zero 在 2800 工具场景实测 token 省 98%——
索引常驻、详情按需，索引思想在工具侧的翻版。""",
}

TITLES = list(DOCS)
IDX = BM25Index([f"{t}。{body}" for t, body in DOCS.items()])


def ranked_hits(query: str):
    """BM25 打分并滤掉零分——「命中数」是真实相关数，不是全库"""
    return [TITLES[i] for s, i in IDX.search(query, len(TITLES)) if s > 0]


def search_full(query: str, k: int = 5) -> str:
    """反模式：top-k 命中直接拼接全文——搜索结果页塞给你整本书"""
    return "\n\n".join(f"【{t}】\n{DOCS[t]}" for t in ranked_hits(query)[:k])


def search_candidates(query: str, cursor: int = 0, page_size: int = 2) -> str:
    """三板斧①②：结构化候选（编号+标题+摘要+指针）+ 游标（总数+翻页方式）"""
    hits = ranked_hits(query)
    page = hits[cursor:cursor + page_size]
    lines = [f"[{cursor + i}] {t}｜{DOCS[t][:36]}…｜read_doc({cursor + i})"
             for i, t in enumerate(page, 1)]
    if cursor + page_size < len(hits):
        lines.append(f"（共 {len(hits)} 条命中，已显示第 {cursor + 1}-{cursor + len(page)} 条；"
                     f"传 cursor={cursor + page_size} 看下一页）")
    else:
        lines.append(f"（共 {len(hits)} 条命中，已显示到末页）")
    return "\n".join(lines)


if __name__ == "__main__":
    query = "上下文太长 token 开销怎么省"
    print(f"查询：{query}（同一查询，看返回形状的差别）")

    full = search_full(query)
    print(f"\n=== 反模式：拼接命中全文 ===\n{full}")
    print(f"  → {len(full)} 字符一次性灌进上下文；Agent 往往只深读 1 篇，其余白烧窗口")

    cands = search_candidates(query)
    print(f"\n=== 三板斧①②：结构化候选 + 分页游标 ===\n{cands}")
    print(f"  → {len(cands)} 字符，只有全文拼接的 {len(cands) / len(full):.0%}；")
    print("    真实网页单页 3k-30k 字符、摘要 ~160 字符——比例一到两个数量级")

    print("\n=== 翻页：Agent 自主决定是否继续（cursor 由返回值递进） ===")
    print(search_candidates(query, cursor=2))
    print("  停一停：候选为什么必须带指针 read_doc(N)（网页搜索里是 URL）？")
    print("  答：那是第二跳的入口——浏览候选后决定深读谁，深读用 25.1 的 offset/")
    print("    limit；没有指针，看完目录也无处可去。两斧合璧 = search→read")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（纯本地，BM25 检索 + 字符量对比）：
#   1. 6 篇学习笔记建 BM25 索引；查询「上下文太长 token 开销怎么省」
#   2. search_full：命中文档全文拼接——反模式的返回形状与字符数
#   3. search_candidates：同一查询的结构化候选列表（page_size=2）+ 字符数
#   4. cursor=2 翻第二页——游标契约：末页明说、翻页方式写进返回值
# 验证内容：
#   - 命中 3-4 篇（KV Cache/工具发现/上下文压缩主命中；BM25 靠单字
#     「长」陥分垫底——BM25 字面性的真实趣味），GraphRAG/MCP 零分不入列
#   - 候选列表每行 = 编号+标题+摘要+指针，末行注明总数与翻页方式
#   - 候选列表约为全文拼接的 30% 上下（语料只是缩小版：真实网页单页
#     3k-30k 字符 vs 摘要 160，比例一到两个数量级——hook 在输出里）
#   - 第二页出现「已显示到末页」——游标走到头也有明确交代，不静默
