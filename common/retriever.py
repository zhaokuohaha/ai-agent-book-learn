"""common/retriever.py — 检索器组件：BM25 稀疏 + 稠密 + RRF 融合（第 3 章沉淀）

两路引擎统一 search(query, k) → [(score, doc_id)] 接口，可互换、可融合：
    稀疏 BM25Index：倒排思路的字面精确匹配（书后术语索引页）——k1 词频饱和
    与 b 长度归一化修正 TF-IDF 的两个坑；IDF 是变体公式（分子为"不含该词
    的文档数"，+0.5 平滑），词出现在过半文档时会变负，须设下限
    稠密 DenseIndex：bge 嵌入余弦（语义地图）——Bi-Encoder：查询/文档各自
    独立编码再比距离，快而粗，适合海量初筛；查询侧加 BGE 官方检索指令前缀
    rrf_fuse：多路排名融合——两路得分分布不可比（余弦 0~1 vs BM25 可到
    几十），RRF 抛弃原始得分只用排名倒数求和：序数可比，基数不可比

共享演示语料与 load_faq 也在这里：实验脚本名以数字开头（如 18.1）不能被
import，跨实验共享的语料/函数只能沉淀到 common（D16 MemoryStore 同款先例）。
"""

import math
import re
from collections import Counter

import jieba
import numpy as np
from sentence_transformers import SentenceTransformer

BGE_MODEL = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
STOPWORDS = {"的", "了", "吗", "呢", "我", "你", "在", "是", "和", "与",
             "有", "要", "能", "什么", "多少", "怎么"}

# 多主题 FAQ 演示语料（18.x 系列共享；五个主题给 BM25 的 IDF 留出稀有度梯度）
CORPUS = """# 退款政策
订单签收后 7 天内可申请全额退款，需提供订单号。退款原路退回支付账户，3-5 个工作日到账。
# 退货与换货
退货需吊牌完整，换货仅限同款不同尺码，不支持跨款换货。
# 会员制度
黄金会员享 95 折优惠和专属客服通道，年消费满 5000 元自动升级。
# 配送说明
默认顺丰快递。下单后 48 小时内发货，偏远地区延长至 72 小时，满 99 元免运费。
# 开票政策
支持开具电子普通发票，需在下单后 30 天内提交抬头，跨月不再受理。"""


def chunk_structural(text, max_len=60):
    """结构感知分块（17 实验沉淀）：先按标题切（一块一主题），超长降级到句子"""
    out = []
    for sec in re.split(r"\n(?=#)", text.strip()):
        if len(sec) <= max_len:
            out.append(sec)
            continue
        cur = ""
        for sent in re.split(r"(?<=[。！？])", sec):
            if cur and len(cur) + len(sent) > max_len:
                out.append(cur.strip())
                cur = sent
            else:
                cur += sent
        if cur.strip():
            out.append(cur.strip())
    return out


def load_faq():
    """演示语料的 (chunks, topics)：chunk 下标即 doc_id，主题名取自标题行"""
    chunks = chunk_structural(CORPUS, 60)
    topics = [c.splitlines()[0].lstrip("# ") for c in chunks]
    return chunks, topics


def tokenize(text):
    """中文分词 + 停用词/符号过滤——BM25 的"词频"以词为单位，先得有词"""
    return [w for w in jieba.lcut(text) if w.isalnum() and w not in STOPWORDS]


class BM25Index:
    """稀疏检索引擎：字面精确匹配。k1=1.5 词频饱和（边际递减），b=0.75 长度归一化"""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.avgdl = sum(map(len, self.tokens)) / len(self.tokens)
        self.df = Counter(w for t in self.tokens for w in set(t))
        self.N = len(docs)

    def _idf(self, w):
        # BM25 的 IDF 变体：分子是"不含该词的文档数"；词出现在过半文档会变负，设下限
        return max(math.log((self.N - self.df[w] + 0.5) / (self.df[w] + 0.5)), 0)

    def search(self, query, k=3):
        q_tokens = tokenize(query)
        scored = []
        for i, tokens in enumerate(self.tokens):
            tf = Counter(tokens)
            len_norm = self.k1 * (1 - self.b + self.b * len(tokens) / self.avgdl)
            score = sum(self._idf(w) * tf[w] * (self.k1 + 1) / (tf[w] + len_norm)
                        for w in q_tokens if tf[w])
            scored.append((score, i))
        return sorted(scored, reverse=True)[:k]


class DenseIndex:
    """稠密检索引擎：bge 嵌入余弦——Bi-Encoder（查询/文档独立编码），语义模糊匹配"""

    def __init__(self, docs, model=None):
        self.docs = docs
        self.model = model or SentenceTransformer(BGE_MODEL)
        self.vecs = self.model.encode(docs)

    def search(self, query, k=3):
        q = self.model.encode([QUERY_PREFIX + query])[0]  # 包一层 list → (1, 512) 再取行向量
        cos = self.vecs @ q / (np.linalg.norm(self.vecs, axis=1) * np.linalg.norm(q))
        order = np.argsort(-cos)[:k]
        return [(float(cos[i]), int(i)) for i in order]


def rrf_fuse(rankings, k=60):
    """RRF 多路融合：score = Σ 1/(k+rank)，只用排名不用原始得分（序数可比基数不可比）"""
    scores = Counter()
    for ranking in rankings:  # 每路是 doc_id 列表（相关性降序）
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] += 1 / (k + rank)
    return [doc_id for doc_id, _ in scores.most_common()]
