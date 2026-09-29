"""chapter3/20.3_pr_pipeline.py — 知识更新 PR 流水线：把知识库当代码库管

本实验新增的主概念：知识更新的 Proposer/Reviewer 流水线（承接 chapter1 的
04/05 proposer-reviewer 模式，这次管的是知识库）。

三层分离（文件的物证，Git 是唯一 source of truth）
    原始证据层 evidence.md：只增不改（对话/工单——D15 轨迹 append-only 回响）
    知识层 knowledge.md：审核后的条目，diff 通过才合入（主数据）
    服务层（向量索引）：全部可重建的派生物（缓存）——合入后重建
    后端一句话：Git 里的知识才是主数据，向量库只是缓存。

PR 流水线（课件伪码的真实版）
    Proposer：读证据+现有知识 → 工作分支上出小而完整的 diff；
    Reviewer：拿 diff+原始证据独立审核，只核对证据不顺着说；
    通过才合入 → 重建索引 → 新知识可检索。
    幕 A 演示驳回（手写一个过度泛化的坏 diff——证据只说家具，diff 却说
    "所有商品"）；幕 B 真实 LLM 全流程。驳回场景必须确定性可复现。

异源互审（工程细节）：生产用能力相近但不同家族的模型（如 Claude+GPT）
    降低同犯一类错误的概率；本实验同模型两次调用+角色提示词隔离（简化点）。
    定期整理=定期重构：去重合并、回读原始证据核查、冲突标注适用场景；
    失效内容打生效时间元数据，检索层直接过滤；多租户权限过滤必须下推到
    检索层（敏感内容一旦进了上下文就很难不泄露）——本实验不实现，记钩子。

运行：uv run python -X utf8 chapter3/20.3_pr_pipeline.py（需 .env，约 4 次真实调用）
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL
from common.retriever import DenseIndex

# 新证据：一条客服工单——质保卡是原始证据层的典型输入
EVIDENCE = ("客服工单 #88：用户 4 月购入的布艺沙发 6 月接缝开裂，已超 30 天退货期。"
            "用户出示随货质保卡，卡片载明'家具类商品两年质保，非人为损坏免费维修或更换'。"
            "客服已按质保通道登记免费维修。")
BAD_DIFF = "+ 质保政策：所有商品享受两年质保，非人为损坏免费维修或更换。"  # 幕 A：手写的坏 diff（过度泛化）


def reviewer(model, diff, knowledge):
    """Reviewer：拿 diff+证据独立审核——只核对证据支撑，不顺着 Proposer 说"""
    verdict = model.invoke([
        SystemMessage(content=(
            "你是知识库审核员（Reviewer）。判断 diff 是否被证据完全支撑："
            "范围不扩大、细节不编造。只输出 JSON：{\"approved\": true/false, \"comment\": \"理由\"}"
        )),
        HumanMessage(content=f"现有知识：{knowledge}\n原始证据：{EVIDENCE}\n待审 diff：{diff}"),
    ]).content
    return "true" in verdict.lower(), verdict[verdict.index("{"): verdict.rindex("}") + 1]


if __name__ == "__main__":
    model = ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=60, max_retries=2,
    )  # SDK 重试治瞬时超时/限流（实测 DeepSeek 偶发 timeout）
    workdir = Path(tempfile.mkdtemp(prefix="agblearn_d20_"))
    evidence_f, knowledge_f = workdir / "evidence.md", workdir / "knowledge.md"

    # 三层分离的物证：证据只增、知识是主数据
    evidence_f.write_text(EVIDENCE, encoding="utf-8")
    knowledge = [
        "退款政策：签收后 7 天内可全额退款，原路退回 3-5 个工作日到账。",
        "退货与换货：退货需吊牌完整，换货仅限同款不同尺码。",
        "会员制度：黄金会员享 95 折优惠和专属客服通道。",
    ]
    knowledge_f.write_text("\n".join(knowledge), encoding="utf-8")
    print(f"三层分离：证据层 {evidence_f.name}（只增）｜知识层 {knowledge_f.name}（主数据）")
    print(f"新证据入库：{EVIDENCE[:40]}…\n")

    print("=== 幕 A：坏 diff 被驳回（审什么：范围扩大 = 过度泛化） ===")
    print(f"  Proposer 提案：{BAD_DIFF}")
    approved, verdict = reviewer(model, BAD_DIFF, knowledge)
    print(f"  Reviewer 裁决：approved={approved}  {verdict[:120]}")
    print("  → 证据只支撑'家具类'，diff 扩大到'所有商品'——驳回。知识库合入的")
    print("    每一条都要能追溯到原始证据（不顺着说，是 Reviewer 的职业素养）")

    print("\n=== 幕 B：真实 Proposer 出 diff → 审核通过 → 合入重建 ===")
    diff = model.invoke([
        SystemMessage(content=(
            "你是知识维护员（Proposer）。基于新证据对现有知识库提出最小改动 diff："
            "只加一条新政策条目，格式'+ 条目名：内容'，范围严格限于证据支撑。只输出 diff。"
        )),
        HumanMessage(content=f"现有知识：{knowledge}\n新证据：{EVIDENCE}"),
    ]).content.strip()
    print(f"  Proposer 提案：{diff[:80]}")
    approved, verdict = reviewer(model, diff, knowledge)
    print(f"  Reviewer 裁决：approved={approved}  {verdict[:100]}")
    if not approved:
        raise SystemExit("幕 B 被驳回（动态输出，可重跑）——流水线正确拦截了它")
    entry = diff.lstrip("+").strip()
    knowledge.append(entry)                                   # 合入主数据
    knowledge_f.write_text("\n".join(knowledge), encoding="utf-8")
    print(f"  合入 knowledge.md（现 {len(knowledge)} 条）→ 重建向量索引（派生物）")

    print("\n=== 验证：新知识可检索 ===")
    index = DenseIndex(knowledge)  # 服务层=缓存：合入后重建
    q = "沙发送修要钱吗？"
    hits = index.search(q, 2)
    print(f"  「{q}」top-1：{knowledge[hits[0][1]][:40]}…")
    answer = model.invoke([
        SystemMessage(content="仅依据给定知识回答；不足就明说。"),
        HumanMessage(content=f"知识：{[knowledge[i] for _, i in hits]}\n问题：{q}"),
    ]).content
    print(f"  答：{answer[:100]}")
    print("\n  → 流水线闭环：证据（只增）→ PR（diff+审核）→ 知识（主数据）→")
    print("    索引（缓存重建）→ 新知识生效。乱改必乱库，合入必过审——")
    print("    Git 里的知识才是主数据，向量库只是可重建的缓存")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（约 4 次真实 LLM 调用）：
#   1. 临时目录落两层文件：evidence.md（只增）与 knowledge.md（主数据 3 条）
#   2. 幕 A：手写坏 diff（"所有商品两年质保"——证据只支撑家具类）→ Reviewer
#      裁决，预期驳回（驳回场景确定性可复现）
#   3. 幕 B：Proposer 真实出 diff（限家具类）→ Reviewer 复审 → 通过合入
#      knowledge.md → DenseIndex 重建（服务层=派生物）
#   4. 查询"沙发送修要钱吗"验证新知识生效
# 验证内容（模型输出动态，验收看要点不看文案）：
#   - 幕 A approved=False（过度泛化被拦截——若模型放行属动态异常，可重跑）
#   - 幕 B diff 应含"家具"限定词；approved=True 后 knowledge 增至 4 条
#   - 最终查询 top-1 命中新质保条目，答案要点"两年内免费维修"——
#     证据→审核→合入→重建→生效的完整闭环
