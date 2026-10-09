"""chapter4/24.1_tool_scale_cost.py — 工具规模的两笔账：选错率与 token/缓存开销

本实验新增的主概念是「工具规模膨胀的代价」。承接 D23（接入了 MCP，工具
可以无限挂）——今天面对必然到来的问题：工具从 10 个涨到几百个怎么办。

两笔账（原书先算账再开药方）：
    账一·正确性：工具超过 100 个时，即使最先进的模型也容易选错——原书
    实验 Qwen3-4B 面对 120+ 工具全量注入，指令遵循能力直接崩掉（查股价
    却去调 web search）。模型的选择注意力被稀释：选项越多，单选项越浅。
    账二·钱包：仅 5 个 MCP server 就可能带来数万 token 的工具定义，200K
    窗口还没开口先烧近三成；且 tools 字段是静态前缀（D09），工具集一变
    就击穿 KV Cache（D10 铁律回响）。

后端类比：全量注入 = 每次建连都把全库 DDL 下发给客户端——正解是先
SHOW TABLES 给索引，要哪张表再 DESC。

三档按需披露（本实验实算的对照）：
    全量：100 份完整 JSON Schema 常驻（MCP 接入的默认行为）
    索引：分组目录（组名 + 工具名）常驻——Cursor 模式，A/B 实测省 46.9%
    预筛：按语义只注入 top-k 完整 schema——Anthropic 实验中 Opus 4 的
    工具使用准确率 49% → 74%（预筛/发现是 24.2 的主角）

运行：uv run python -X utf8 chapter4/24.1_tool_scale_cost.py（纯本地，不烧 API）
"""

import json

# ---- 造一个 100 工具的真实感注册表：4 组 × (5 种能力 × 5 个变体) ----

VARIANTS = {  # 每组同一能力族配 5 个变体——工具一多，名字必然长得像
    "金融": ["A股", "港股", "美股", "期货", "外汇"],
    "网页": ["通用", "新闻", "学术", "代码", "地图"],
    "文件": ["文本", "JSON", "CSV", "PDF", "日志"],
    "数据": ["订单库", "用户库", "库存库", "日志库", "指标库"],
}
CAPABILITIES = {  # 每组 5 个能力族 × 5 变体 = 25 个工具
    "金融": [("get_price", "查询实时价格"), ("get_history", "查询历史行情"),
             ("get_chart", "生成走势图"), ("get_depth", "查询盘口深度"),
             ("get_news", "查询相关资讯")],
    "网页": [("search", "搜索"), ("fetch", "抓取全文"), ("summarize", "总结网页"),
             ("extract", "提取结构化数据"), ("monitor", "监控变更")],
    "文件": [("read", "读取"), ("grep", "按关键词搜"), ("parse", "解析结构"),
             ("convert", "格式转换"), ("validate", "校验")],
    "数据": [("query", "执行查询"), ("schema", "查看表结构"), ("insert", "写入"),
             ("stats", "统计画像"), ("export", "导出")],
}


def build_registry():
    """100 个工具：name/description/inputSchema 完整 JSON 形状（MCP tools/list 同款）"""
    tools = []
    for group, caps in CAPABILITIES.items():
        for name, verb in caps:
            for var in VARIANTS[group]:
                tools.append({
                    "name": f"{name}_{var}",
                    "description": f"{verb}（{group}·{var}）。何时用：需要{var}{verb}时。"
                                   f"边界：仅覆盖{var}，不含其他变体。",
                    "inputSchema": {"type": "object",
                                    "properties": {"target": {"type": "string", "description": "目标标识"},
                                                   "options": {"type": "object", "description": "可选参数"}},
                                    "required": ["target"]},
                })
    return tools


REGISTRY = build_registry()
TOK = 0.6  # DeepSeek 中文约 0.6 token/字——量级估算用，不追求精确


def show_scale_cost():
    print(f"账二·钱包：{len(REGISTRY)} 个工具（4 服务器×25）全量注入的静态前缀有多大")
    full = len(json.dumps(REGISTRY, ensure_ascii=False))
    catalog = len("\n".join(f"{g}: {', '.join(c + '_' + v for c, _ in caps for v in VARIANTS[g])}"
                            for g, caps in CAPABILITIES.items()))
    top2 = 2 * full // len(REGISTRY)  # 预筛：只注入语义 top-2 的完整 schema
    for label, chars, note in [
        ("全量（MCP 默认）", full, "200K 窗口先烧一大块，且每次会话都带"),
        ("索引（分组目录）", catalog, "Cursor 模式：常驻只有名字，A/B 实测省 46.9%"),
        ("预筛（语义 top-2）", top2, "Anthropic 模式：准确率 49%→74%（24.2 展开）"),
    ]:
        print(f"  {label:<10} ≈{chars:>6} 字符 ≈{int(chars * TOK):>6} token｜{note}")
    print(f"  全量/索引 ≈ {full / catalog:.0f} 倍——规模越大，按需披露越划算")


def show_correctness_cost():
    print(f"\n账一·正确性：{len(REGISTRY)} 个工具里挑 1 个有多难")
    lookalikes = [t["name"] for t in REGISTRY if "price" in t["name"] or "history" in t["name"]]
    print(f"  「查苹果股价」的近邻候选就有 {len(lookalikes)} 个：{', '.join(lookalikes[:6])}…")
    print("  原书实验：Qwen3-4B 面对 120+ 工具全量注入，指令遵循崩掉——查股价去调 web search")
    print("  机制：选项越多注意力越稀释；分组目录就是给模型的路标（作业①：写进系统提示词）")


def show_cache_cost():
    print("\n缓存账（D10 铁律回响）：tools 是静态前缀，工具集一变全灭")
    print("  第 N 轮 [system | tools(100) | 轨迹…] ← 前缀稳定，跨请求缓存命中")
    print("  接入新 server：[system | tools(125) | 轨迹…] ← tools 段变，其后全部重算")
    print("  所以按需方案必须回答「动态加载怎么不毁缓存」——24.2 的 KV 保命三原则")


def render_catalog():
    """作业①：分组目录渲染——写进系统提示词的形状（信息源性质分组）"""
    print("\n[作业①] 分组目录（写进 system，常驻的只有这些名字）：")
    for g, caps in CAPABILITIES.items():
        names = ", ".join(f"{c}_{v}" for c, _ in caps for v in VARIANTS[g][:2])
        print(f"  {g}（{len(VARIANTS[g])} 变体×5 能力）: {names}…")

if __name__ == "__main__":
    show_scale_cost()
    show_correctness_cost()
    show_cache_cost()
    render_catalog()

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（纯本地，不烧 API）：
#   1. build_registry 生成 100 工具（4 组×5 能力×5 变体，完整 JSON Schema 形状）
#   2. show_scale_cost：全量 / 索引目录 / 预筛 top-2 三档字符与 token 估算对比
#   3. show_correctness_cost：近邻候选计数（选错率的直观代理）+ 原书数字钩子
#   4. show_cache_cost：tools 前缀变化的缓存失效示意；render_catalog 演示分组目录
# 验证内容：
#   - 三档对比：全量 ≈ 数万字符（估算上万 token），索引只有它的几十分之一
#     ——「5 个 MCP server 数万 token」的教学版实证
#   - 近邻候选 10 个（2 能力族×5 变体）——名字长得像正是选错的温床
#   - 目录四组齐全，每组带变体×能力计数——作业①的落点形状
