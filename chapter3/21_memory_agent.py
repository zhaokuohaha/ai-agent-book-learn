"""chapter3/21_memory_agent.py — W3 收官：记得住、查得到的玩具 Agent（阶段性整合）

三周知识的合体（W1 agent 循环 + W2 上下文注入 + W3 记忆与 RAG）：

双通道架构（课件知识地图的落地，比骨架玩具升级在哪）
    用户消息 ──┬─→ 记忆通道（读）：store.render() 注入 system——记忆是跨会话
               │    档案，每次都要在场（D11 角色信任体系，D16.2 先例）
               │    记忆通道（写）：会话结束 → LLM 提取（D15 ETL）→ 四检查
               │    → MemoryStore 落库（CQRS：读路径只读，写路径异步，绝不直写）
               └─→ 知识库通道：knowledge_base_search 工具——D20 Agentic RAG，
                    agent 自己判断何时搜、搜什么（混合检索：稠密+BM25+RRF）
    LLM 走 D02 手写 while 循环：思考 → 调工具 → 观察 → 回答。

四检查（写路径的核验，课件的"来源/时间/冲突/隐私"）：来源=本次会话可追溯、
    时间=带会话日期入库、冲突=同 key 旧值进 history 留痕、隐私=演示跳过脱敏。

作业落点：① 两会话跑通（会话 2 新进程 load 记忆，不问自答）；② 冲突测试
    （"我搬家去上海了"→ 新值覆盖 + history 保留杭州）；③ 思考题 100 字方案
    见章节 README——MemoryStore 本身就是"时间戳+来源+读时消歧"的代码实现。

运行：uv run python -X utf8 chapter3/21_memory_agent.py（需 .env，约 5 次真实调用）
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL
from common.memory_store import MemoryStore
from common.retriever import BM25Index, DenseIndex, load_faq, rrf_fuse

STORE_PATH = Path(__file__).parent / "agent_memory.json"  # 跨会话记忆文件（git 已忽略）
TS_1, TS_2 = "2026-09-28", "2026-09-29"  # 两个会话的日期（写死保证可复现）

chunks, _ = load_faq()  # D17/D18 的 FAQ 知识库（5 主题）


@tool
def knowledge_base_search(query: str) -> str:
    """在店铺知识库中检索政策信息（退货/退款/会员/配送/开票等）"""
    # 教学直白：每次重建索引；真实系统索引入库时离线建一次（D20.3：服务层=缓存）
    d = [i for _, i in DenseIndex(chunks).search(query, 3)]
    b = [i for _, i in BM25Index(chunks).search(query, 3)]
    top = rrf_fuse([d, b])[:2]  # D18 双路混合：稠密+BM25→RRF 融合取 top-2
    return "\n".join(f"[{n}] {chunks[i]}" for n, i in enumerate(top, 1))


def agent_loop(model, store, user_input, ts):
    """D02 手写循环 + 双通道：system 带记忆档案，KB 是工具（agent 按需检索）"""
    messages = [
        SystemMessage(content=(
            f"你是私人购物助理。今天是 {ts}。用户长期记忆（跨会话档案，直接采信）：\n"
            f"{store.render()}\n需要店铺政策时调用 knowledge_base_search 工具。"
        )),
        HumanMessage(content=user_input),
    ]
    llm = model.bind_tools([knowledge_base_search])
    for _ in range(6):
        resp = llm.invoke(messages)
        if not resp.tool_calls:
            return resp.content
        messages.append(resp)
        for call in resp.tool_calls:
            result = knowledge_base_search.invoke(call["args"])
            print(f"    [tool] 搜索「{call['args']['query']}」")
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    return "（达到最大轮数）"


def extract_and_store(model, store, dialogue, ts):
    """写路径（会话结束）：先读已有字段 → LLM 提取（D15 ETL）→ 四检查 → 落库。
    首跑教训：不告知已有字段时，提取器两次生成不同 key（city vs
    residence_city）——覆盖根本不触发，同一个人变成两个住址字段。
    修复即 D20.3 Proposer 原则：先检索已有知识再增删改（复用同义 key）"""
    known = ", ".join(store.active()) or "（无）"
    resp = model.invoke([
        SystemMessage(content=(
            "从对话提取值得长期记住的用户事实（选择性/抽象化：丢任务细节）。"
            f"已有记忆字段：{known}——新事实若与已有字段语义相同（如住址），必须复用该字段名。"
            '只输出 JSON：{"facts": [{"key": "snake_case", "value": "短语"}]}'
        )),
        HumanMessage(content=dialogue),
    ])
    text = resp.content.strip()
    facts = json.loads(text[text.index("{"): text.rindex("}") + 1])["facts"]
    for f in facts:
        conflict = f["key"] in store.active()  # 四检查之三：冲突检测
        store.remember(f["key"], f["value"], "semantic", ts)  # 时间戳随会话日期入库
        checks = f"来源=本次会话✓ 时间={ts}✓ 冲突={'覆盖，旧值入 history✓' if conflict else '无✓'}"
        print(f"    [入库] {f['key']} = {f['value']}（{checks}）")


if __name__ == "__main__":
    model = ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=60, max_retries=2,
    )
    STORE_PATH.unlink(missing_ok=True)  # 每次从零演示两会话

    print(f"=== 会话 1（{TS_1}）：初次见面 → 对话 → 提取落盘 ===")
    store = MemoryStore(STORE_PATH)
    s1_dialogue = "用户：我在杭州，平时网购多。对了，你们退货政策是什么？"
    print(f"  {s1_dialogue}")
    print(f"  助手：{agent_loop(model, store, '我在杭州，平时网购多。你们退货政策是什么？', TS_1)[:80]}")
    print("  [会话结束 → 写路径：LLM 提取 + 四检查]")
    extract_and_store(model, store, s1_dialogue, TS_1)
    print(f"  落盘 {STORE_PATH.name}，进程退出\n")

    print(f"=== 会话 2（{TS_2}）：新进程 load 记忆 → 不问自答 + 冲突测试 ===")
    store = MemoryStore(STORE_PATH)  # 新会话第一件事：档案搬回内存
    print("  [验证 1] 记得你：问政策时 agent 已知用户在杭州（不再问城市）")
    print(f"  助手：{agent_loop(model, store, '帮我看看退款几天到账？', TS_2)[:80]}\n")
    print("  [验证 2] 冲突（作业②）：用户改口搬家")
    s2_dialogue = "用户：对了，我上个月搬家去上海了，以后发货按上海算。"
    print(f"  {s2_dialogue}")
    print("  [会话结束 → 写路径]")
    extract_and_store(model, store, s2_dialogue, TS_2)
    # 动态找发生过覆盖的字段（history 长度 >1）——提取器的 key 命名是动态的
    # （city/location/residence_city 都可能），硬编码具体名字会 KeyError
    conflicted = [k for k in store.active() if len(store.history(k)) > 1]
    for k in conflicted:
        cur = store.active()[k]
        print(f"  冲突字段「{k}」：当前 = {cur.content}（{cur.ts}）")
        print("  完整历史（时间推理与审计的原料）：")
        for e in store.history(k):
            print(f"    {e.ts}  {e.content}")
    if not conflicted:
        print("  （未检测到覆盖——key 又漂移了，可重跑；key 一致性是记忆系统的真实难题）")
    print("\n  → '我以前住哪？'可答：杭州（09-28）；'现在发货到哪？'可答：上海（09-29）")
    print("  → 轨迹不可变/档案可演进（D15 铁律）；冲突不丢证据，读时消歧取最新（D16）")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（约 5 次真实 LLM 调用：会话1 问答+提取、会话2 问答+提取）：
#   1. 会话 1：空库起步，agent 检索 KB 答退货政策（工具轨迹）→ 对话文本送
#      extract_and_store：LLM 提取（city=杭州等）→ 四检查打印 → 落盘退出
#   2. 会话 2：新构造 MemoryStore 即 load → agent 答"退款几天"时 system 已带
#      用户档案（不问自答）→ "搬家上海"对话再走写路径
#   3. 冲突验证：city 当前值=上海，history=[杭州(09-28), 上海(09-29)]
# 验证内容（模型输出动态，验收看要点不看文案）：
#   - 会话 1 提取出 city=杭州（可能伴生网购偏好等合理条目——选择性/抽象化）
#   - 会话 2 agent 回答直接给 3-5 工作日（不再反问用户信息——记忆生效）
#   - 冲突后 active=上海、history 完整保留杭州——新值覆盖且旧值留痕（作业②）；
#     首跑曾因 key 漂移（city vs residence_city）覆盖未触发——修复=提取前
#     告知已有字段复用同名（先读后写，D20.3 Proposer 原则），key 漂移是
#     记忆系统的真实难题（Mem0 v2 写入时消歧正是在解它）
#   - 四检查逐条打印：来源/时间/冲突/（隐私演示跳过）——CQRS 写路径具象化
