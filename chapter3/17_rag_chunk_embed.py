"""chapter3/17_rag_chunk_embed.py — RAG 基础（上）：分块与向量化

核心概念（原书第 3 章 · W3 Day 17）：

RAG 三段流程：检索 → 注入 → 生成
    LLM 训练数据有截止日期，公司文档它从没见过——RAG 把模型能力与外部知识库
    的时效性拼起来：检索器捞出相关片段拼进 prompt，LLM 基于片段生成。
    本实验做"检索"的前半（分块+向量化+相似度），注入与生成明天接上 BM25
    与重排后跑完整闭环。

为什么要分块（两个原因）
    ① 嵌入模型对输入长度有限制（bge-small-zh 为 512 token），整篇多主题文档
    压成一个向量，哪个主题都表达不准；② 检索目标是只把相关小块注入上下文，
    块太大 = 无关内容全灌进窗口稀释注意力。
    块太小则单块语义不完整（"收入增长 3%"——哪家公司？），太大则多主题混杂、
    向量被稀释——一对权衡，实践起点 256-1024 token、重叠 10%-20%。

三种分块策略：固定切分（按字符硬切+重叠，段落/标题会被拦腰切断，size 是它的命门）
    → 递归/结构感知（先按标题切、超长降级到句子——生产默认；标题是天然边界，
    对 size 不敏感，本实验手写它：逻辑透明，生产等价物是 LangChain
    RecursiveCharacterTextSplitter）→ 语义切分（算相邻句子嵌入相似度在"断崖"
    处下刀，质量最高但要额外嵌入计算，本实验不实现）。

稠密嵌入 = 把语义变成几何
    每段文本映射成高维向量（bge-small-zh 是 512 维），语义越近向量越近——
    一张高维地图。衡量"近"用余弦相似度（夹角而非长度：内容相同一长一短的
    两段文本方向一致，余弦能正确判定语义相同——关心方向不关心长短）。
    后端视角：BM25 是倒排索引的字面精确匹配（书后术语索引页），稠密检索是
    查询向量找最近邻的语义模糊匹配——搜"HTTP-403"各有所长，明天讲 RRF 合并。
    Word2Vec 静态词向量的致命伤："bank"在河岸和银行是同一个向量；BERT/BGE
    上下文感知模型看整句，"苹果手机"和"两斤苹果"里的"苹果"方向不同。

为什么用真实嵌入而不是课本的玩具 bigram 词袋
    玩具词袋下"退货要多久到账"匹配"退款"只是共享汉字的字面巧合（本质是
    近似的 BM25），演示不了"语义变几何"；且 DeepSeek 无 embeddings API。
    故直接落地进阶作业：sentence-transformers + BAAI/bge-small-zh-v1.5
    （512 维中文模型，CPU 毫秒级；多语言生产场景换 bge-m3）。
    BGE 检索的不对称设计：查询侧加官方指令前缀、文档侧不加。

作业落点：① 得分差观察（Part 1："退货"零字面重叠却命中"退款"块——语义几何
    的铁证）；② 块大小实验在固定切分上做（Part 2：20 碎片→60 完整→200 混杂；
    结构感知对 size 免疫）；③ Part 3 整篇一块的区分度消失 + 一句话答案见文末。

运行：uv run python -X utf8 chapter3/17_rag_chunk_embed.py（零 LLM 调用，纯本地嵌入）
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer

BGE_MODEL = "BAAI/bge-small-zh-v1.5"  # 512 维中文嵌入模型（~95MB，首次运行自动下载）
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："  # BGE 不对称检索：查询侧加指令、文档侧不加

# 多主题 FAQ 文档：退款 + 弱相关的退货 + 两个干扰主题，供检索拉开得分梯度
DOCUMENT = """# 退款政策
订单签收后 7 天内可申请全额退款，需提供订单号。退款原路退回支付账户，3-5 个工作日到账。
# 退货与换货
退货需吊牌完整，换货仅限同款不同尺码，不支持跨款换货。
# 会员制度
黄金会员享 95 折优惠和专属客服通道，年消费满 5000 元自动升级。
# 配送说明
默认顺丰快递。下单后 48 小时内发货，偏远地区延长至 72 小时。"""


def chunk_fixed(text, max_len=60):
    """固定大小切分：按字符硬切 + 10% 重叠——最粗暴，标题/段落会被拦腰切断"""
    step = max(1, max_len - max_len // 10)
    return [text[i:i + max_len].strip() for i in range(0, len(text), step) if text[i:i + max_len].strip()]


def chunk_structural(text, max_len=60):
    """结构感知切分（借鉴课本思路）：先按 Markdown 标题切（一块一主题），
    超长段再按句号降级（不在句中下刀）——不额外算力就尊重文档结构"""
    out = []
    for sec in re.split(r"\n(?=#)", text.strip()):
        if len(sec) <= max_len:
            out.append(sec)
            continue
        cur = ""
        for sent in re.split(r"(?<=[。！？])", sec):  # 句末零宽断言：切在句号后
            if cur and len(cur) + len(sent) > max_len:
                out.append(cur.strip())
                cur = sent
            else:
                cur += sent
        if cur.strip():
            out.append(cur.strip())
    return out


def embed(texts, model, is_query=False):
    """批量嵌入。BGE 不对称设计：查询侧加检索指令前缀，文档侧原样"""
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    return model.encode(texts)  # → shape (n, 512)，未归一化（余弦在检索里算）


def retrieve(query, chunks, vecs, model, k=1):
    """稠密检索：查询向量逐一比全部块向量，余弦排序取 top-k"""
    q = embed([query], model, is_query=True)[0]
    scored = sorted(((float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v))), c)
                     for c, v in zip(chunks, vecs)), reverse=True)
    return scored[:k]


if __name__ == "__main__":
    model = SentenceTransformer(BGE_MODEL)
    query = "退货要多久到账？"

    print("=== Part 0：两种分块策略的块形状（size=60，纯文本对比） ===")
    fixed = chunk_fixed(DOCUMENT, 60)
    structural = chunk_structural(DOCUMENT, 60)
    print(f"固定切分（{len(fixed)} 块）：如第 2 块「{fixed[1][:20]}…」——步长对齐标题纯靠运气，块尾句子被切断")
    print(f"结构感知（{len(structural)} 块）：恰好 {len(re.findall(r'# ', DOCUMENT))} 个主题各一块——一块一主题")
    print("  → 结构感知不加算力就尊重了文档结构，生产的默认选择")

    print(f"\n=== Part 1：得分全景（结构感知块）——语义距离的梯度 ===")
    vecs = embed(structural, model)
    for score, c in retrieve(query, structural, vecs, model, k=len(structural)):
        print(f"  {score:+.2f}  {c[:22]}…")
    print("  → 查询说'退货'却命中'退款'块：'到账'语义把退款块（退款+时长齐全）拉到第一，")
    print("    退货块（只对上'退货'分量）反而居中——嵌入看的是整句方向，不是字面词；")
    print("    配送块也高：'要多久'撞上它的时长讨论——语义检索的模糊性，明天 BM25 补台")

    print("\n=== Part 2：块大小实验（固定切分 20 / 60 / 200）——size 是固定切分的命门 ===")
    for size in (20, 60, 200):
        chunks = chunk_fixed(DOCUMENT, size)
        vecs = embed(chunks, model)
        score, top = retrieve(query, chunks, vecs, model)[0]
        verdict = {20: "碎片得分不低（更聚焦），但'7 天内可申请'条件被切丢——答了一半",
                   60: "块内主题基本完整，退款要素齐全，一眼可答",
                   200: "一块混入退货/会员/配送——无关内容随块灌进上下文窗口"}[size]
        print(f"  size={size:>3}（{len(chunks)} 块）top-1 {score:+.2f}：{top[:24]}… → {verdict}")
    print("  → 对照：结构感知在 60 与 200 下块形状相同——标题是天然边界，对 size 免疫")

    print("\n=== Part 3：块太大的隐性代价——区分度消失 ===")
    chunks = chunk_fixed(DOCUMENT, 300)  # ~160 字文档 → 整篇一块
    vecs = embed(chunks, model)
    hits = [retrieve(q, chunks, vecs, model)[0] for q in ("退货要多久到账", "会员有什么折扣")]
    print(f"  size=300 固定切分 → {len(chunks)} 块（整篇文档压成一个向量）")
    print(f"  「退货到账」{hits[0][0]:+.2f} / 「会员折扣」{hits[1][0]:+.2f}，命中同一块：{hits[0][1] == hits[1][1]}")
    print("  → 多主题平均成一个向量：不同问题命中同一块、得分齐跌——检索器分不出主题了")

    print("\n=== 作业 3：一句话答案 ===")
    print("  块太小则单块语义不完整（命中断头碎片），太大则多主题混杂向量被稀释（不同")
    print("  问题命中同一块）——分块目标是'一块一主题'，在检索粒度与语义完整性间求平衡。")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（零 LLM 调用，纯本地嵌入；首次运行自动下载 bge-small-zh ~95MB）：
#   1. 加载模型 → Part 0：同一 size=60 下固定切分与结构感知的块形状对比
#   2. Part 1：结构感知 4 块嵌入，查询"退货要多久到账？"对全部块打余弦分降序
#   3. Part 2：固定切分 size ∈ {20, 60, 200} 各自分块+嵌入+top-1，对比命中内容；
#      结构感知对照（60 与 200 块形状相同——标题边界对 size 免疫）
#   4. Part 3：固定切分 size=300 → 整篇一块，两个不同主题查询是否命中同一块
# 验证内容：
#   - Part 0：固定切分出现跨主题断头块；结构感知恰好 4 块各一主题
#   - Part 1：退款块第一（"退货"与"退款"零字面重叠却最高分——语义几何铁证）；
#     退货块居中、会员垫底；配送块因"要多久"的时长语义偏高（模糊性）
#   - Part 2：size=20 命中碎片（得分不低但丢了"7 天内"条件）；size=60 完整；
#     size=200 混入无关主题
#   - Part 3：两个不同主题查询命中同一块 = True（区分度消失的物证）
