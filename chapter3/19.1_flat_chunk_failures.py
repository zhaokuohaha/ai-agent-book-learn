"""chapter3/19.1_flat_chunk_failures.py — 结构化索引 · 一：扁平 chunk 的翻车现场

本实验新增的主概念：扁平知识库的三种翻车（伴生：规则卡——索引阶段离线提炼）。

隐藏假设被击穿
    D18 管道默认"答案就躺在某几个 chunk 里"，两个反例：
    计数问题（top-k 漏召）：10 份档案问比例，top-3 只捞回碎片，模型基于
        残缺样本瞎算——调大 k 也只是 30/100 与 100/100 的区别；
    边界问题（最近邻偏置 + 边界语义缺失）：工单库里全是个案（军人过审、
        军医过审、教师被拒），"仅限军人及军队医护"这条规则本身不存在于
        任何单条工单——护士来问，语义最近的"军医 Sarah"工单排最前，模型
        顺势推断"医护可以"（错）。
    解法都在索引阶段：离线汇总统计摘要 / LLM 通读全库提炼规则卡入库。
    后端视角：拿原始日志直接撑报表 → 正确姿势是先建聚合表/物化视图——
    原始材料直接入库，等于把难题甩给检索和模型。

统计摘要用代码算、规则卡用 LLM 提炼（作业①）：计数是确定性计算
    （D16"能确定性计算的别让模型算"在索引阶段回响），提炼边界规则
    要通读理解个案背后的共性——这才是 LLM 的活。

运行：uv run python -X utf8 chapter3/19.1_flat_chunk_failures.py（需 .env，约 5 次真实调用）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL
from common.retriever import DenseIndex

# 语料一：10 份宠物档案（9 黑 1 白）——计数问题的翻车现场
PETS = [f"宠物档案 #{i}：黑猫，已接种疫苗，登记在册。" for i in range(1, 10)] + [
    "宠物档案 #10：白猫，已接种疫苗，登记在册。"]

# 语料二：3 条客服工单——边界问题的翻车现场（资格规则不在任何单条里）
TICKETS = [
    "工单 #1：退伍军人老王申请军人专属折扣，资质审核通过，已发放。",
    "工单 #2：军队医院医生 Sarah 申请军人专属折扣，资质审核通过，已发放。",
    "工单 #3：中学教师老李申请军人专属折扣，资质审核不通过。",
]


def rag_answer(model, docs, question, k=3):
    """D18 闭环最小复刻：稠密检索 top-k → 注入 → 生成（仅依据资料）。
    返回 (命中文档, 回答)——命中列表用来展示"喂了什么给模型" """
    hits = DenseIndex(docs).search(question, k)
    refs = "\n\n".join(f"【资料 {n}】{docs[i]}" for n, (_, i) in enumerate(hits, 1))
    answer = model.invoke([
        SystemMessage(content="你是客服助理。仅依据参考资料回答；资料不足以回答就明说，不要编造。"),
        HumanMessage(content=f"{refs}\n\n问题：{question}"),
    ]).content
    return [docs[i] for _, i in hits], answer


if __name__ == "__main__":
    model = ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0,
    )

    print("=== Part 0：计数问题——top-k 漏召，残缺样本瞎算 ===")
    question = "登记库里黑猫和白猫的比例是多少？"
    hits, answer = rag_answer(model, PETS, question, k=3)
    print(f"「{question}」检索 top-3（库里共 {len(PETS)} 份，只捞回 {len(hits)} 份）：")
    for h in hits:
        print(f"    {h[:16]}…")
    print(f"  翻车答案：{answer[:110]}")
    print("  → 捞回的 3 份全是黑猫——白猫连上下文都没进；top-k 捞的是'语义最近的几份'")
    print("    不是'全貌'，调大 k 也只是 30/100 与 100/100 的区别")

    print("\n  [解法] 索引阶段离线汇总统计摘要（代码算，确定性）：")
    n_black = sum("黑猫" in p for p in PETS)
    summary = f"全库统计摘要：共 {len(PETS)} 份宠物档案，黑猫 {n_black} 只（{n_black * 10}%），白猫 {len(PETS) - n_black} 只（{(len(PETS) - n_black) * 10}%）。"
    print(f"    {summary}")
    hits, answer = rag_answer(model, PETS + [summary], question, k=3)
    print(f"  修正答案：{answer[:110]}")
    print("  → 聚合表入库，一次检索命中全貌——原始日志撑报表不如先建物化视图")

    print("\n=== Part 1：边界问题——最近邻偏置 + 规则不在库里（作业①） ===")
    question = "我是地方医院的护士，能申请军人专属折扣吗？"
    hits, answer = rag_answer(model, TICKETS, question, k=3)
    print(f"「{question}」检索命中：")
    for h in hits:
        print(f"    {h[:24]}…")
    print(f"  翻车答案：{answer[:130]}")
    print("  → 最近邻偏置：军医 Sarah 工单排最前，但'仅限军人及军队医护'的否定与")
    print("    全称语义不在任何单条工单里——含糊不答（'无法确定'）也是失败：")
    print("    用户要的是可执行的答案，调大 k 也没用")

    print("\n  [解法] LLM 通读全库提炼规则卡（作业①；个案背后的共性要'理解'，这是 LLM 的活）：")
    rule = model.invoke([
        SystemMessage(content=(
            "通读以下客服工单，推断军人专属折扣的资格边界：谁适用、谁不适用、"
            "分界标准是什么（注意'军队医院'与普通医院的区别，规则要覆盖未出现的职业）。"
            "只输出规则一句话。"
        )),
        HumanMessage(content="\n".join(TICKETS)),
    ]).content.strip()
    rule_card = f"规则卡：{rule}"
    print(f"    {rule_card}")
    hits, answer = rag_answer(model, TICKETS + [rule_card], question, k=3)
    print(f"  修正答案：{answer[:130]}")
    print("\n  → 三种翻车一个药方：索引阶段投入算力提炼（统计摘要/规则卡），")
    print("    把难题在离线阶段解决，而不是甩给在线检索和模型")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（约 5 次真实 LLM 调用；嵌入本地）：
#   1. Part 0 前：10 份档案问比例，top-3 注入生成——残缺样本的答案
#   2. Part 0 后：代码算统计摘要追加入库，同一问题再答——对比
#   3. Part 1 前：护士问资格，top-3 工单注入生成——最近邻偏置的答案
#   4. Part 1 中：LLM 通读 3 条工单提炼规则卡（1 次调用）
#   5. Part 1 后：规则卡入库再答——对比
# 验证内容（模型输出动态，验收看要点不看文案）：
#   - Part 0 翻车答案基于 3 份残缺样本（实测捞回 3 份全黑——白猫没进上下文，
#     模型'无法计算'）；修正答案应给出 9:1 / 90%:10%（统计摘要被命中并采信）
#   - Part 1 翻车答案含糊不答（'无法确定'——不答也是失败）；首次实测简单提示
#     词只能提炼出描述性归纳（'退伍军人√军医√教师×'），加强为'推断分界标准'
#     后规则卡才含'军队编制'边界——规则卡质量取决于源信号与提示词设计；
#     修正答案应明确拒绝（地方护士不在军内编制）
#   - 命中列表展示翻车根因：Part 0 只见 3 份档案、Part 1 全是个案无规则
