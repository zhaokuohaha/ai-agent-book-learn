"""chapter3/19.3_mini_graph_rag.py — 结构化索引 · 三：mini GraphRAG——多跳与消歧

本实验新增的主概念：GraphRAG 三元组图（伴生：三元组的有损压缩；
    Part 3 给出生产形状——微软 local search 的最小版）。

两大强项（三层讲解）
    直觉：扁平库是"一堆便签"，图是"一张关系网"——便签各说各话，网能顺藤摸瓜。
    机制：知识拆成 (主语, 关系, 宾语) 三元组；多跳查询 = 沿关系边逐跳遍历
          （等价于多表 JOIN）；实体消歧 = 同名实体是不同节点、各连各的边。
    代码落点：neighbors(s) 拿某实体的出边；query(start, path) 按关系序列
          逐跳扩展——课件的十行代码，本实验加扁平库对照与消歧实验。

对照设计：同一批事实同时存两份——扁平库（每条三元组写成一句自然语言）
    与图（原始三元组）。多跳问题两边都问：扁平检索单次捞不齐链条（起点句
    和终点句都命中，中间"就职于"那环容易缺）；图上三跳 JOIN 一气呵成。

实体消歧 ≠ 词义消歧：bank 是银行还是河岸，上下文嵌入就能分（D17 的
    语义地图）；两个张医生是两个真人，靠的是实体知识/图结构（图上天然分开）。

代价：三元组是有损压缩——"如果下周还下雨就取消海边改去博物馆"拆成三条
    后，if-then 的条件逻辑与时间依赖全丢。生产分层互补：核心信息保留原文
    保语义完整，结构化元数据只负责索引检索（Advanced JSON Cards 的思路，
    D16 伏笔回收）。

生产形状（Part 3，示例偏实际应用）：课件的 graph_query 要预知关系路径
    （用户不知道也不该知道"主诊医生→就职于→地址"）——真实主流两条：
    ① 微软 GraphRAG local search：实体链接（query→种子实体）→ k 跳邻域
    扩展 → 收集事实 → LLM 合成，只需跳数不需路径（生产另有 Leiden 社区
    检测 + 全局摘要，本实验简化掉）；② Neo4j text2cypher：LLM 把问题
    翻译成 Cypher 直接执行。本实验实现①的最小版：实体卡嵌入链接（生产
    用 NER + 实体嵌入表）+ 双向 k 跳扩展 + LLM 合成。

运行：uv run python -X utf8 chapter3/19.3_mini_graph_rag.py（需 .env，1 次真实调用）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL
from common.retriever import DenseIndex

TRIPLES = [
    ("用户", "主诊医生", "张医生-A"),
    ("张医生-A", "科室", "牙科"),
    ("张医生-B", "科室", "心脏科"),  # 同名不同人：图上两个节点各连各的边
    ("张医生-A", "就职于", "协和医院"),
    ("协和医院", "地址", "北京市东城区帅府园1号"),
]

# 同一批事实的扁平形态：每条三元组一句自然语言（对照实验用）
FLAT_DOCS = [f"{s}的{p}是{o}。" for s, p, o in TRIPLES]


def neighbors(entity):
    return [(s, p, o) for s, p, o in TRIPLES if s == entity]


def graph_query(start, path):
    """多跳遍历：按关系序列逐跳扩展，等价于多表 JOIN（课件的十行代码）"""
    cur, chain = [start], []
    for rel in path:
        nxt = []
        for e in cur:
            for s, p, o in neighbors(e):
                if p == rel:
                    chain.append((s, p, o))
                    nxt.append(o)
        cur = nxt
    return cur, chain


# ---------- 生产形状：微软 GraphRAG local search 的最小版（Part 3 用） ----------

ENTITY_NAMES = list({s for s, _, _ in TRIPLES})
ENTITY_CARDS = [f"{e}：" + "；".join(f"{p}是{o}" for s, p, o in TRIPLES if s == e)
                for e in ENTITY_NAMES]  # 实体名片：实体+它的关系描述


def link_entities(query, k=2):
    """实体链接：query 匹配实体卡 top-k → 种子实体（生产用 NER+实体嵌入表）"""
    hits = DenseIndex(ENTITY_CARDS).search(query, k)
    return [ENTITY_NAMES[i] for _, i in hits]


def expand(seeds, hops=2):
    """k 跳邻域扩展（双向）：不预知路径只给跳数——沿边扩散收集可达事实"""
    facts, frontier, seen = [], set(seeds), set()
    for _ in range(hops):
        nxt = set()
        for s, p, o in TRIPLES:
            if (s, p, o) in facts:
                continue  # 边级去重：同一条边不因双向遍历收两次
            if s in frontier and o not in seen:
                facts.append((s, p, o))
                nxt.add(o)
                seen.add(o)
            elif o in frontier and s not in seen:
                facts.append((s, p, o))
                nxt.add(s)
                seen.add(s)
        frontier = nxt
    return facts


if __name__ == "__main__":
    model = ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0,
    )
    flat = DenseIndex(FLAT_DOCS)
    multi_hop = "我的主诊医生所在医院的地址是什么？"

    print("=== Part 0：多跳问题——扁平库断链 vs 图一气呵成 ===")
    print(f"「{multi_hop}」")
    print("  扁平库 top-3（单次检索）：")
    for score, i in flat.search(multi_hop, 3):
        print(f"    {score:+.2f}  {FLAT_DOCS[i]}")
    print("  → 实测比'断中间'更惨：起点句（主诊医生）命中了，但'就职于协和'与")
    print("    '医院地址'两句都没进 top-3——捞回的是两句'科室'这种语义近邻但不搭链的")
    print("    句子；终点缺席，模型只能靠猜补链")
    answers, chain = graph_query("用户", ["主诊医生", "就职于", "地址"])
    print("  图上三跳 JOIN：")
    print("    " + " -> ".join(f"{s}--{p}-->{o}" for s, p, o in chain))
    print(f"    答案：{answers[0]}")

    print("\n=== Part 1：实体消歧——两个张医生，图上天然分开 ===")
    print("  图上查询（实体名即节点 id，必须指定是谁）：")
    for who in ("张医生-A", "张医生-B"):
        print(f"    {who} 的科室：{neighbors(who)[0][2]}")
    print("  扁平库检索「张医生的科室」（没说哪个张医生）：")
    for score, i in flat.search("张医生的科室", 3):
        print(f"    {score:+.2f}  {FLAT_DOCS[i]}")
    print("  → 实测 B(0.65) 与 A(0.62) 近乎并列，谁在前纯属浮点噪声——扁平库给不出")
    print("    '哪个张医生'的区分；这不是词义消歧（bank 靠上下文嵌入能分），是实体")
    print("    消歧（同名真人，靠图结构/实体知识区分）——图查询天生强制先指定节点")

    print("\n=== Part 2：代价——三元组是有损压缩 ===")
    plan = "如果下周还下雨，就取消海边行程改去博物馆。"
    print(f"  原句：{plan}")
    lossy = [("下周行程", "条件", "下雨"), ("下周行程", "取消", "海边行程"), ("下周行程", "改为", "博物馆")]
    for t in lossy:
        print(f"    {t}")
    print("  → 拆完只剩'下雨/取消/改去'三个孤立事实——if-then 条件、时间依赖全丢；")
    print("    生产分层互补：核心信息保留自然语言原文，结构化元数据只负责索引")

    print("\n=== Part 3：生产形状——local search，不预知路径 ===")
    seeds = link_entities(multi_hop)
    print(f"  实体链接（query→实体卡 top-2）：{seeds}")
    facts = expand(seeds, hops=2)
    print("  2 跳邻域扩展收集的事实：")
    for s, p, o in facts:
        print(f"    {s} --{p}--> {o}")
    answer = model.invoke([
        SystemMessage(content="仅依据给定事实回答问题；事实不足就明说。"),
        HumanMessage(content="\n".join(f"{s}的{p}是{o}" for s, p, o in facts) + f"\n\n问题：{multi_hop}"),
    ]).content
    print(f"  LLM 合成：{answer[:80]}")
    print("  → 对比 Part 0：玩具 graph_query 预知路径（课件示意问题）；local search")
    print("    只给跳数——实体链接找种子、邻域扩展收事实、LLM 合成答案，即微软")
    print("    GraphRAG local search 的骨架（生产另有社区检测/重排/真图库 Neo4j）")

    print("\n=== 作业③ 思考：业务里天然多跳的查询 ===")
    print("  例：电商'这单买的商品，它的供应商开的发票抬头是什么'——")
    print("    订单→商品→供应商→发票，四跳 JOIN；扁平 RAG 要多次检索拼接，")
    print("    图上一次遍历。多跳刚需场景（医疗/法律/供应链）才值得为图买单")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（1 次真实 LLM 调用；扁平对照与实体链接用本地嵌入）：
#   1. Part 0：同一批事实的两种形态——FLAT_DOCS 建稠密索引，多跳问题检索
#      top-3 看断链；graph_query 沿三条边遍历出完整链与答案（玩具：预知路径）
#   2. Part 1：图上分别查两个张医生的科室（节点天然区分）；扁平检索
#      「张医生的科室」看 A/B 是否并列混淆
#   3. Part 2：条件句拆三元组的丢失演示（定性）
#   4. Part 3（生产形状）：link_entities 实体卡嵌入链接出种子 → expand
#      双向 2 跳收集事实 → LLM 合成回答同一个多跳问题——全程没给路径
# 验证内容（得分随嵌入版本浮动，以实际输出为准）：
#   - Part 0 图侧三跳链完整：用户→张医生-A→协和医院→地址，答案为协和地址
#   - Part 1 图侧 A=牙科、B=心脏科；扁平侧两句近乎并列（浮点噪声定先后）
#   - Part 3：种子应含用户或张医生-A/协和医院之一（链接正确）；扩展事实
#     覆盖主诊医生→就职于→地址链条；LLM 答案含"帅府园"——不预知路径
#     也能答出与玩具一致的结果（local search 骨架生效）
