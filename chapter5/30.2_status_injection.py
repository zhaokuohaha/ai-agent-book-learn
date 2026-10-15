"""chapter5/30.2_status_injection.py — 状态栏注入：环境感知与 KV 前缀保卫

本实验新增的主概念是「状态栏动态注入」——Coding Agent 高度依赖执行环境
状态（当前目录/git 分支/未提交变更），这些信息每次推理前以 <agent_status>
消息注入上下文（D13 回响：13.1 验证机制、13.2 生产化中间件）。

注入位置学（本实验的核心对照）：
    状态信息每轮都在变（时间/轮次/环境快照）。写进 system = 静态前缀每轮
    被改 → KV 缓存全灭，每轮从头重算；作为普通消息追加到尾部 = 历史一个
    字节不动，前缀稳定持续命中，只算增量。用 DeepSeek 真实缓存计数字段
    prompt_cache_hit_tokens 实测两种注入的缓存形状（D24.2 同款取数）——
    与 24.2 的 schema 注入是同一原则的两次落地：动态内容一律走尾部追加。

教学简化注：真实 Claude Code 注入 cwd/git 分支/未提交变更；本工作区非
git 仓库，用「文件清单+大小」代替。文件大小正好喂给「读最大的」任务——
状态栏让模型不用探查就知道环境，这就是注入的价值（13.1 的催办任务同构：
把状态送到模型眼前，省一轮工具往返）。

运行：uv run python -X utf8 chapter5/30.2_status_injection.py（需 .env，约 5-6 次调用）
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from common.coding_tools import make_tools
from common.config import API_KEY, BASE_URL, MODEL

WS = Path(__file__).parent / ".agent_ws2"  # 独立工作区（与 30.1 的 trace 不混）

SYSTEM = "你是工作区里的 Coding Agent。回答必须基于真实读到的内容；环境信息以 <agent_status> 状态栏为准。"

TASK = "工作区里有哪些文件？分别读最大的和最小的文件，各用一句话概括内容。"

NOTES = '''# 晨间简报生成器

每天 7 点抓取天气 + 日历前三个日程，生成一页 Markdown 简报发到邮箱。

- v0.1 已跑通天气抓取
- v0.2 日历对接中
- 已知问题：周末日历为空时简报会缺一块
'''

SMALL = "购物清单：燕麦、牛奶、蓝莓。\n"


def status_bar(ws: Path, turn: int) -> str:
    """环境快照：轮次 + 时间戳保证每轮内容必然变化（真实场景 git 状态也在变）"""
    files = "、".join(f"{p.name}({p.stat().st_size}B)" for p in sorted(ws.iterdir()) if p.is_file())
    return (f"<agent_status>\n工作区文件：{files}\n"
            f"当前轮次：{turn}｜{datetime.now():%H:%M:%S}\n</agent_status>")


def run_loop(model, task, ws: Path, mode: str, max_turns: int = 5):
    """A/B 对照唯一变量 = 注入位置：mutate 改 system（毁前缀）| append 尾部追加"""
    by_name = {t.name: t for t in make_tools(ws)}
    llm = model.bind_tools(list(by_name.values()))
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=task)]
    for turn in range(1, max_turns + 1):
        if mode == "mutate":  # 反模式：改写历史消息——静态前缀变了，缓存全灭
            messages[0] = SystemMessage(content=SYSTEM + "\n\n" + status_bar(ws, turn))
        else:                  # 正解：只追加——历史一个字节不动，前缀稳定
            messages.append(HumanMessage(content=status_bar(ws, turn)))
        resp = llm.invoke(messages)
        hit = resp.response_metadata.get("token_usage", {}).get("prompt_cache_hit_tokens", 0)
        print(f"  [轮 {turn}] 缓存命中 {hit} token｜消息 {len(messages)} 条｜本轮 {len(resp.tool_calls)} 个调用")
        messages.append(resp)
        if not resp.tool_calls:
            return messages
        for call in resp.tool_calls:
            try:
                result = str(by_name[call["name"]].invoke(call["args"]))
            except Exception as e:  # 错误变输入（30.1 规约保留）
                result = f"[错误] {type(e).__name__}: {e}"
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return messages


if __name__ == "__main__":
    shutil.rmtree(WS, ignore_errors=True)
    WS.mkdir(parents=True)
    (WS / "notes.md").write_text(NOTES, encoding="utf-8")
    (WS / "small.txt").write_text(SMALL, encoding="utf-8")
    model = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=60,
                       max_retries=3)  # 29.3 一级恢复接进循环：SDK 内置退避，503/429 静默重试

    print("=== A 反模式：状态栏写进 system 且每轮更新（前缀每轮变）===")
    run_loop(model, TASK, WS, mode="mutate")

    print("\n=== B 正解：状态栏作为尾部追加消息（前缀稳定）===")
    msgs = run_loop(model, TASK, WS, mode="append")
    answer = next(m.content for m in reversed(msgs) if m.type == "ai" and m.content)
    print(f"  [回答节选] {str(answer)[:120]}")

    print("\n结论：同样的环境信息，注入位置决定缓存形状——A 每轮从头重算，")
    print("  B 前缀复用只算增量。「不能写进 system，只在尾部追加」不是风格偏好，是账。")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（约 5-6 次真实调用）：
#   1. 重建工作区（notes.md 约 300B + small.txt 约 30B——大小差是任务线索）
#   2. A 版：每轮 messages[0] 换新 system（SYSTEM+状态栏）→ 同一任务跑完
#   3. B 版：状态栏以 user 消息追加尾部 → 同一任务跑完
# 验证内容（缓存数字跨请求动态，看相对形状不看绝对值——24.2 先例）：
#   - A 版每轮命中 ≈ 0（system 一变，前缀从第一字节重算）
#   - B 版第 1 轮低（首次无缓存），第 2 轮起命中回升且随消息增长抬升
#     ——「只追加则持续命中」的活证据
#   - 两版行为侧：状态栏里有文件大小 → 模型可直接挑 notes.md/small.txt 读，
#     少一轮 glob 探查（注入的价值）；B 版历史里能看到多条 <agent_status>
#     逐轮堆叠——append-only 的代价与红利同在
#   - 模型可能仍先 glob 核实（工具结果比状态栏新鲜）——合理路径都算通过
#   - DeepSeek 前缀缓存按 64 token 块计——短请求的命中数字有粒度噪声，
#     A 与 B 的差异看趋势对比
