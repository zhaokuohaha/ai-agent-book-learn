"""chapter4/24.2_tool_discovery.py — 主动工具发现：discover_tools 元工具 + KV 保命

本实验新增的主概念是「主动工具发现」（三层按需方案的第二层）。承接 24.1
（算清了两笔账：全量注入既选错又烧钱）——解法：常驻工具集缩到最小（一个
元工具），其余工具 Agent 在执行中自己声明能力缺口，系统动态发现、动态注入。

MCP-Zero 模式（原书第二层方案）：
    Agent 执行中调 discover_tools(need="我需要查股价的能力") 声明缺口 →
    系统做两层语义路由（先选服务器，再选工具）→ 命中则把 schema 注入、
    注册进可调用集；未命中明确返回「未找到」（返回 404，别返回模糊的空
    结果——让 Agent 改写需求或用基础工具兜底）。2800 工具场景省约 98%。
    工程等价物：Anthropic 的 Tool Search Tool、OpenAI 的 tool_search。

KV Cache 保命（作业③，本实验用 DeepSeek 真实缓存计数字段实测）：
    动态加载的天然代价是毁缓存——schema 若插进静态前缀（tools 字段），
    整段作废。三原则：① schema 作为普通消息（user 槽位）追加到上下文
    末尾（D13 Skill 注入同款）；② 固定首次注入的原位置，之后只增不减；
    ③ 新工具加进 tools 列表只追加、不修改不重排。真正导致重算的只有：
    TTL 过期、修改/移除/重排已加载工具集。「工具定义必须放最前面」从此
    不再是铁律。

课件差异注：课件示例用字符集 Jaccard 近似语义相似度（玩具）；本实验用
第 3 章沉淀的 bge 嵌入余弦做两层路由——对齐 Anthropic Tool Search Tool
的 embedding 检索机制。教学简化：动态工具统一 target 参数（真实系统的
schema 来自服务端 tools/list，形状任意）。

运行：uv run python -X utf8 chapter4/24.2_tool_discovery.py（需 .env，约 6 次真实调用）。
bge 加载收口 common/retriever.py（get_bge 共享单例；已缓存自动离线，跳过
huggingface.co 在线检查——国内网络下“卡十几分钟宛妳死机”的教训，详见其文件头）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL
from common.retriever import DenseIndex

# ---- 工具库：4 服务器 × 3 工具（name: 描述, 参数提示, stub 结果） ----

TOOL_SERVERS = {
    "金融": {
        "get_stock_price": ("查询股票实时价格", "股票代码，如 'AAPL'", "{target}：189.50 美元，涨 1.2%"),
        "get_exchange_rate": ("查询货币汇率", "货币对，如 'USD/CNY'", "{target}：7.21"),
        "get_company_financials": ("查询公司财报摘要", "公司代码，如 'AAPL'", "{target}：Q3 营收 949 亿美元"),
    },
    "网页": {
        "search_web": ("搜索互联网实时信息", "搜索词", "（{target} 的搜索结果摘要）"),
        "fetch_url": ("抓取网页全文", "URL", "（{target} 的正文，已截断）"),
        "monitor_url": ("监控网页变更", "URL", "{target}：24 小时内无变更"),
    },
    "文件": {
        "read_file": ("读取本地文件内容", "文件路径", "（{target} 的内容，已截断）"),
        "grep_file": ("按关键词搜索文件", "关键词", "（{target} 的 3 处匹配）"),
        "parse_file": ("解析文件结构", "文件路径", "（{target} 的结构树）"),
    },
    "数据": {
        "sql_query": ("执行 SQL 查询", "SQL 语句", "（{target} → 5 行结果）"),
        "get_table_schema": ("查看数据表结构", "表名", "{target}(id, user_id, amount, created_at)"),
        "export_data": ("导出查询结果", "导出描述", "（{target} 已导出 CSV）"),
    },
}

# ---- 两层语义路由的索引（bge 经 get_bge 共享；真实系统离线建一次，D20.3 同款） ----

SERVER_IDX = DenseIndex(
    [f"{s} 服务器：" + "；".join(f"{n}：{d}" for n, (d, _, _) in t.items())
     for s, t in TOOL_SERVERS.items()])
TOOL_IDX = {s: DenseIndex([f"{n}：{d}" for n, (d, _, _) in t.items()])
            for s, t in TOOL_SERVERS.items()}
SERVERS = list(TOOL_SERVERS)
THRESHOLD = 0.55  # bge 余弦阈值：低于它视为「查无此能力」

CATALOG = "\n".join(f"- {s}：{'、'.join(t)}" for s, t in
                    {s: list(v) for s, v in TOOL_SERVERS.items()}.items())  # 作业①：分组目录

SYSTEM = f"""你是全能助理。你可调用的工具只有 discover_tools——完整工具库不在上下文里。

工具库分组目录（常驻索引，完整 schema 需要时再取）：
{CATALOG}

工作方式：遇到目录中能力相关的任务，先调 discover_tools(need="我需要……的能力")
声明缺口，取回的工具即可直接调用；与目录无关的任务，直接回答或建议改写。"""

LOADED = {}  # 动态注册表：发现即入——bind_tools 只追加，不重排（KV 保命③）


def make_tool(name):
    """动态工具：统一 target 参数（教学简化；真实系统 schema 来自 tools/list）"""
    desc, hint, stub = next(t[name] for t in TOOL_SERVERS.values() if name in t)

    def _impl(target: str) -> str:
        return stub.format(target=target)

    return StructuredTool.from_function(_impl, name=name, description=f"{desc}。参数 target：{hint}")


@tool
def discover_tools(need: str) -> str:
    """声明能力缺口，检索工具库。need 用一句话描述需要什么能力，如"我需要查询股票价格的能力"。"""
    server = SERVERS[SERVER_IDX.search(need, 1)[0][1]]  # 第一层：选服务器
    ranked = TOOL_IDX[server].search(need, 2)           # 第二层：服务器内选工具
    hits = [(s, list(TOOL_SERVERS[server])[i]) for s, i in ranked if s >= THRESHOLD]
    if not hits:
        return "未找到合适工具：请改写需求，或用基础工具手工实现"  # 404：明确拒绝，不给模糊空结果
    for _, name in hits:
        LOADED[name] = make_tool(name)
    lines = [f"- {n}：{TOOL_SERVERS[server][n][0]}｜参数 target：{TOOL_SERVERS[server][n][1]}"
             for _, n in hits]
    return (f"已路由到[{server}]服务器，命中 {len(hits)} 个工具（schema 见下一条注入消息，"
            f"此后可直接调用）：\n" + "\n".join(lines))


def agent_loop(model, task, max_turns=5):
    """手写循环：每轮打印缓存命中 token（作业③）；schema 以 user 消息注入并固定原位"""
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=task)]
    for turn in range(1, max_turns + 1):
        llm = model.bind_tools([discover_tools] + list(LOADED.values()))  # 只追加不重排
        resp = llm.invoke(messages)
        hit = resp.response_metadata.get("token_usage", {}).get("prompt_cache_hit_tokens", 0)
        print(f"  [turn {turn}] 缓存命中 {hit} token｜已加载工具 {len(LOADED)} 个")
        messages.append(resp)
        if not resp.tool_calls:
            return messages
        injections = []  # 翻车修复：回执必须紧跟 assistant（D09 配对铁律），
        for call in resp.tool_calls:  # 一轮多个调用时，schema 注入要等回执全部补齐
            t = {discover_tools.name: discover_tools, **LOADED}.get(call["name"])
            result = t.invoke(call["args"]) if t else "错误：工具未加载，请先 discover_tools"
            is_hit = call["name"] == "discover_tools" and not str(result).startswith("未找到")
            placeholder = "工具发现完成，schema 见下一条注入消息。" if is_hit else None
            messages.append(ToolMessage(content=placeholder or str(result), tool_call_id=call["id"]))
            print(f"  [dispatch] {call['name']}({call['args']})")
            if is_hit:
                injections.append(result)
        for schema in injections:  # KV 保命①：schema 作为普通消息（user 槽位）追加，D13 同款
            messages.append(HumanMessage(content=f"<tool_schema>\n{schema}\n</tool_schema>"))
    return messages


if __name__ == "__main__":
    model = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0)

    print("=== 任务 1：查股价（能力缺口 → 发现 → 注入 → 调用闭环） ===")
    msgs = agent_loop(model, "查一下苹果公司（AAPL）的股价。")
    print(f"  回答节选：{next(m.content for m in reversed(msgs) if m.type == 'ai' and m.content)[:100]}")

    print("\n=== 任务 2：弹钢琴（未命中 → 404 → 兜底） ===")
    msgs = agent_loop(model, "帮我用工具弹一首《致爱丽丝》。")
    print(f"  回答节选：{next(m.content for m in reversed(msgs) if m.type == 'ai' and m.content)[:100]}")

    print("\n[作业③·上下文结构] 消息序列（schema 追加进入后原位保持）：")
    for i, m in enumerate(msgs):
        tag = " ← schema 注入处，此后每轮原位命中缓存" if "tool_schema" in str(m.content)[:30] else ""
        print(f"  {i:>2} {m.type:<9}{str(m.content)[:40]!r}{tag}")
    print("  → 动态加载的两难与解法：schema 进 tools 前缀=缓存全灭；")
    print("    以 user 消息追加+固定原位=只追加不重排，前缀只断一次、之后持续命中")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（约 6 次真实 LLM 调用；bge 本地毫秒级）：
#   1. 模块级建两层索引（服务器层+每服务器工具层，共享 bge 实例）
#   2. 任务 1「查 AAPL 股价」：agent 调 discover_tools → 两层路由命中
#      get_stock_price → ToolMessage 占位符 + schema 以 user 消息注入 +
#      LOADED 注册（bind_tools 追加）→ agent 调 get_stock_price('AAPL')
#      → stub 回传 → 回答
#   3. 任务 2「弹钢琴」：discover 未命中（余弦 < 0.55）→ 返回 404 文本
#      → agent 兜底（404 分支本地直测过：「弹钢琴」「播放音乐」均明确拒绝）
#   4. 终局打印消息序列 + schema 注入位置标注（作业③）
#   首跑翻车记：模型一轮发两个 discover 调用，schema 注入消息夹在两个
#   tool 回执之间 → 破坏 tool_call_id 配对链 → API 400（D09 铁律的又一
#   实证）——修复：先补齐全部回执、再统一追加注入消息
# 验证内容（模型输出动态，验收看轨迹不看具体数字）：
#   - 任务 1 轨迹：discover_tools → get_stock_price('AAPL')，且
#     get_stock_price 是注入 schema 之后才出现在 bind_tools 里的
#   - 任务 2（模型两种合理形态都实测过）：其一，调 discover 改写需求
#     重试（「播放音乐」404 后改「生成音频/MIDI」再 404）后兜底；其二，
#     直接看目录判断无此能力兜底（更省调用）——目录可见性本身就在第一层
#     防无效发现；404 作为观察直接回传（D12），不走 schema 注入
#   - 缓存命中轨迹（DeepSeek prompt_cache_hit_tokens，跨请求缓存动态
#     看相对形状不看绝对值）：发现工具那一轮 tools 前缀变长 → 命中只剩
#     system 段（如 384）；之后 tools 稳定、消息只追加 → 命中回升（512）
#     ——「工具集一变击穿缓存、只追加则持续命中」的活证据
#   - 阈值 0.55 为实测起点：相关查询（查股价）≥0.6，无关（钢琴）<0.55
