"""chapter2/14_context_compression.py — 上下文压缩：滑动窗口 + 摘要，第 2 章收官

核心概念（原书第 2 章 · W2 Day 14）：

为什么上下文没满也要压缩（三大动机）
    ① 窗口与成本：工具结果动辄几万字符，几次调用撑爆 128K（原书实验：
       无压缩 5 轮就爆，7 次搜索累计 36.7 万字符）
    ② 总结后的知识更好用：注意力擅长检索不擅长归纳——问"黑猫白猫各几只"，
       模型每次都要遍历 100 个笼子现场重数；提前写入"黑猫 90 只、白猫 10 只"
       就变成可直接检索的知识（与状态栏是同一枚硬币的两面：状态栏是加结论，
       压缩是把原始记录换成结论）
    ③ 上下文焦虑：模型感觉窗口快满时会在任务没完成前草草收尾
    上下文腐化 ≠ 溢出：溢出是"装不下"，腐化是"装得下但找不到"——表越
    大索引越膨胀，查询命中率反而越低。后端类比：原始工具输出=明细日志，
    摘要=物化视图——读预聚合结果远快于每次全表扫描。

压缩与 KV Cache：看似矛盾实则互补
    压缩发生在两次 API 调用之间，不动静态前缀，只替换历史中的 tool
    results——替换点之后缓存失效，之前仍然有效。所以不要每轮都压，
    攒到阈值（窗口 80%）批量压一次。本实验打印压缩前后 payload 的公共
    前缀，肉眼验证"替换点之前仍命中"。

策略光谱（原书实验 2-10，压缩率=压缩后/原体积，越小越狠）
    无压缩：5 轮爆窗失败
    逐条独立摘要：压缩率 10.9%，12 次迭代——信息碎片化，多页重复信息浪费
    上下文感知摘要（带查询意图+已有信息）：压缩率 3.0%，7 次迭代，token 省 75%+
    生产五层组合：工具结果存盘留预览 → 噪声直接删（对噪声做摘要是浪费token）
    → API 层微压缩 → 归档式逐轮摘要（像 git log 不像 squash）→ 最后才全量
    压缩且配熔断器（防困在反复压缩失败的循环里烧钱）
    隔离优于压缩：大范围搜索委派子 Agent，中间噪声根本不进主上下文，
    只回传几百 token 结论（第 4 章展开）

设计四原则：信息价值非均匀（名单>证据>噪声）/ 语义完整性（"2024年5月离开
OpenAI"不能压成"离开"）/ 任务相关性（查名单留广度、析背景留深度）/ 压缩即
理解（压缩模块要接近主模型能力——模型调用模型的递归架构）。

程序逻辑地图（复习用：按执行顺序读）
    主循环 run(model, compressor, label, compress_enabled)：
        ① 每轮开头：若开压缩 → maybe_compress（压缩插在两次 API 调用之间！）
        ② 算总字符 → 仍超 WINDOW 才判溢出（先压缩后判溢出，给压缩一次机会）
        ③ model.invoke 全量重发 → 无 tool_calls 即收工
        ④ 逐个 tool_call 执行 → ToolMessage 追加进轨迹 → 回到①
    maybe_compress(compressor, messages)：
        ① 找受害者：所有 ToolMessage，去掉最后 KEEP_RECENT 条，去掉已带
           [COMPRESSED] 前缀的（防重复）
        ② 阈值门：总字符 ≤ THRESHOLD 或无受害者 → 直接返回（攒着批量压）
        ③ 逐条调压缩器：替换该消息 content 为 "[COMPRESSED] 摘要"（原地替换，
            消息条数不变——tool_call_id 配对链不破）
        ④ dump 前后对照：commonprefix 长度 = 仍可命中的缓存前缀（止于首个
            被替换的消息，替换点之后失效，之前全保）
    对照组 vs 实验组：同任务同窗口，唯一变量是 compress_enabled——
        无压缩：4 家明细 2746 字符 > 1500 → 溢出失败
        压缩：2746 → 897，模型检索摘要结论直接答对，不再逐笼重数
    两个模型两种身份（关键设计）：base（裸模型）当压缩器只做摘要；
        base.bind_tools([search_pets]) 当决策模型——菜单不同，行为面不同。

踩坑留档（2026-09-24 真 bug，后来复习时重点看）
    现象：实验组被压缩的店中，部分店"⚠️ 数据缺失"，模型诚实报缺不编造。
    根因：压缩器误用了绑着 search_pets 的模型——压缩时它有时会决定
        "先调工具查一下"而非直接写摘要；工具调用响应的 content 为空 →
        summary="" → "[COMPRESSED] "空摘要 → 数据丢失。四家命运可对照：
        最后一条（KEEP_RECENT 保护，原文保留）✓；压缩器老实写摘要的 ✓；
        压缩器去调工具的 ✗。
    连带教训：早前一次运行里"模型重查了两家"曾被误读为"有损压缩后的验证"，
        其实是同一 bug——模型在找回丢失的数据，不是在验证。数据丢失类问题
        先查管道，别急着怪模型。
    修复（三处）：压缩器改用裸模型（不给不需要的菜单）；空摘要兑底保留原文
        （失败的压缩要可见，不能静默吞数据）；每条摘要打印可见（压缩不再是黑盒）。
    教训一句话：**给模型的菜单决定它的行为面——不需要工具的调用就别绑工具**。

作业落点：
    1. D09 循环 + 滑动窗口 + 摘要压缩 → run()：阈值触发批量压缩最老未标记
       的 tool 消息，上下文感知摘要（带任务意图），近 1 条原文不压缩
    2. [COMPRESSED] 标记防重复 + 每轮打印字符数与真实 prompt_tokens →
       maybe_compress() + 主循环
    3. 选做（KV Cache 约束检查）→ 压缩前后 dump 公共前缀：止于首个替换点，
       system 与其前消息字节级不动

运行：uv run python -X utf8 chapter2/14_context_compression.py（需 .env，约 10 次真实调用）
"""

import json
import sys
from os.path import commonprefix
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL

# ---------- 模拟环境：四家宠物店的巡查记录（逐笼明细，刻意做大逼出压缩） ----------

SHOPS = {"萌宠之家": (18, 12), "喵星球": (22, 8), "汪喵之家": (15, 15), "尾巴工厂": (10, 20)}
TOTAL_BLACK, TOTAL_WHITE = sum(b for b, _ in SHOPS.values()), sum(w for _, w in SHOPS.values())


@tool
def search_pets(shop: str) -> str:
    """查询指定宠物店的完整巡查记录（逐笼明细）。可选店铺：萌宠之家、喵星球、汪喵之家、尾巴工厂。"""
    if shop not in SHOPS:
        return f"错误：没有这家店 {shop}（可选：{'、'.join(SHOPS)}）"
    black, white = SHOPS[shop]
    # 每笼带编号描述：把记录做大（~450 字/家），4 家就超窗口——逼出压缩
    cages = "；".join(f"笼子{i:02d}：{'黑猫' if i <= black else '白猫'}（个体编号{i:03d}，已检疫）"
                      for i in range(1, black + white + 1))
    return f"{shop}巡查记录（共{black + white}笼）：{cages}。"


# ---------- 窗口模拟：字符数近似 token（教学用；真实 token 见 usage） ----------

WINDOW = 1500     # 模拟上下文窗口
THRESHOLD = 1200  # 80% 触发压缩：攒到阈值批量压，不是每轮都压
KEEP_RECENT = 1   # 滑动窗口保护：最近 1 条 tool 原文不压缩（当前任务细节无损）

SYSTEM = """你是宠物店巡查统计助手。用 search_pets 逐家查询巡查记录，最后汇总：
每家黑猫、白猫各多少只，以及全城总数。注意：以 [COMPRESSED] 开头的内容是
已压缩的历史摘要，直接采信其中的统计数字，不要试图还原逐笼明细。"""

TASK = "请统计这四家宠物店：每家黑猫白猫各几只？全城黑猫白猫总数是多少？"


def total_chars(messages):
    return sum(len(m.content or "") for m in messages)


def dump(messages):
    return "\n".join(json.dumps({"role": m.type, "content": m.content}, ensure_ascii=False) for m in messages)


def maybe_compress(compressor, messages):
    """阈值触发 + 批量压缩 + [COMPRESSED] 防重复 + KV Cache 前缀证据

    压缩器必须用未绑工具的裸模型：绑着 search_pets 的模型压缩时可能
    "先查一下"而非直接摘要——工具调用的 content 为空 → 空摘要 → 数据丢失
    （实测踩坑：模型诚实报"数据缺失"而非编造，才暴露此 bug）。
    """
    tool_idx = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    # 受害者 = 除最后 KEEP_RECENT 条外的、尚未标记的 tool 消息：
    # 近期原文保留（当前任务细节无损）+ [COMPRESSED] 前缀防重复压缩
    victims = [i for i in tool_idx[:-KEEP_RECENT] if not messages[i].content.startswith("[COMPRESSED]")]
    if total_chars(messages) <= THRESHOLD or not victims:
        return  # 阈值门：攒到 80% 窗口才批量压，不每轮压（少破坏缓存）
    before = dump(messages)
    size_before = total_chars(messages)
    for i in victims:  # 上下文感知压缩：把任务意图写进压缩提示，只留统计结论
        summary = (compressor.invoke([
            SystemMessage(content="你是压缩器。把巡查记录压成一行统计，保留店名与黑猫白猫数量"
                                  "（格式：店名：黑猫X只、白猫Y只），丢弃逐笼明细。只输出这一行。"),
            HumanMessage(content=messages[i].content),
        ]).content or "").strip()
        if not summary:  # 压缩失败（空摘要）：宁可留原文也不丢数据
            print(f"    [压缩失败] 第 {i} 条摘要为空，保留原文")
            continue
        messages[i].content = f"[COMPRESSED] {summary}"  # 原地替换：条数不变，tool_call_id 配对链不破
        print(f"    [摘要] {summary}")  # 压缩产物可见，不是黑盒
    after = dump(messages)
    # 公共前缀长度 = 仍可命中的缓存范围：止于首个被替换的消息
    # （替换点之后失效一次，之前全保——这就是"压缩与 KV Cache 互补"的字节证据）
    shared = len(commonprefix([before, after]))
    print(f"    [压缩] 批量替换完成；总字符 {size_before} → {total_chars(messages)}；"
          f"公共前缀 {shared}/{len(before)} 字符（替换点之前的前缀缓存仍命中，system 纹丝不动）")


def run(model, compressor, label, compress_enabled, max_turns=8):
    """D09 骨架 + 压缩插在两次调用之间；无压缩组撞 WINDOW 即模拟溢出"""
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=TASK)]
    print(f"\n--- [{label}] ---")
    for turn in range(1, max_turns + 1):
        if compress_enabled:
            maybe_compress(compressor, messages)  # 压缩插在两次调用之间（请求间预处理，不是调用中）
        size = total_chars(messages)
        if size > WINDOW:  # 先压缩后判溢出：给压缩一次机会，压完仍超才算真溢出
            print(f"  [溢出] {size} > {WINDOW} 字符，任务失败——原书实验：无压缩 5 轮爆窗")
            return None, messages
        response = model.invoke(messages)
        usage = response.response_metadata.get("token_usage") or {}
        print(f"  [turn {turn}] 发送 {size} 字符｜真实 prompt_tokens={usage.get('prompt_tokens')}"
              f"（cache_hit={usage.get('prompt_cache_hit_tokens')}）")
        messages.append(response)
        if not response.tool_calls:
            return response.text, messages
        for call in response.tool_calls:
            observation = search_pets.invoke(call)
            messages.append(ToolMessage(content=observation.content, tool_call_id=call["id"]))
            print(f"    [dispatch] search_pets({call['args']}) → 返回 {len(observation.content)} 字符明细")
    return "（达到轮数上限）", messages


if __name__ == "__main__":
    base = ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0,
    )
    model = base.bind_tools([search_pets])  # 主循环：绑工具（决策用）

    print("=== 对照组：无压缩（模拟 1500 字符窗口） ===")
    answer_plain, _ = run(model, base, "无压缩", compress_enabled=False)

    print("\n=== 实验组：阈值触发压缩（同一任务，同一窗口） ===")
    answer_compressed, messages = run(model, base, "压缩", compress_enabled=True)

    print("\n=== 结果对照 ===")
    if answer_plain is None:
        print("  无压缩：上下文溢出，任务失败")
    print(f"  压缩组回答节选：{(answer_compressed or '')[:150]}")
    print(f"  [验收] 正确总数应为 黑猫={TOTAL_BLACK}、白猫={TOTAL_WHITE}——"
          f"压缩摘要保留了统计数字，模型检索结论而非逐笼重数（总结后的知识更好用）")
    compressed = [m.content for m in messages if isinstance(m, ToolMessage) and m.content.startswith("[COMPRESSED]")]
    print(f"  [防重复验证] 压缩后未被二次处理的消息数 = {len(compressed)}（[COMPRESSED] 标记生效）")
    print("\n[收官] 第 2 章一句话：上下文 = 静态前缀 + ReAct 轨迹，KV Cache 是架构约束；")
    print("       提示词/Skills/状态栏负责加内容，压缩与隔离负责减内容。")
