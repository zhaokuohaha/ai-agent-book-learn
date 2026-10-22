"""chapter6/37.1_event_router.py — 紧急度路由：语义判定 vs 关键词匹配

本实验新增的主概念是「结构化事件 + 紧急度路由」——每个输入建模为四维
结构化事件（来源/渠道/内容/上下文），统一 schema 之后三种处理策略才有
统一前提（类比 Kafka 统一消息信封）：取消式（紧急：用户喊停/系统告警，
提前制造安全点重估局势）、队列式（常规：攒到安全点批量追加省往返）、
并行式（独立轻量：旁路小会话即答）。为什么来源维度是提示注入的防线：
只有来源清晰，Agent 才不会把工具结果里藏的指令当成用户指令。

路由器怎么做（课件原话：紧急度判定别硬编码）——A 关键词匹配（问题侧
玩具）：「停」→取消、「天气」→并行，其余入队。B LLM 语义路由（生产
路线）：分类 LLM 读事件内容判策略+理由——两个棺材钉反例坐实关键词版
不可用：「先别继续了，马上」无「停」字被误入队（该取消）；「别停在
半路，继续跑完」含「停」被误取消（该继续）。

运行：uv run python -X utf8 chapter6/37.1_event_router.py（需 .env，7 次真实调用）
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL


@dataclass
class Event:
    """四维结构化事件（作业①的统一 Event 结构）——统一信封，路由与注入的前提"""

    source: str    # 来源：user / tool / timer / system——注入防线的关键维度
    channel: str   # 渠道：im / result / cron
    content: str   # 内容
    context: str = ""  # 背景：与当前任务的关联


ROUTE = ("你是 Agent 的事件路由器。判定该事件的处理策略，只输出 JSON："
         '{"strategy": "cancel"或"queue"或"parallel", "reason": "一句话"}。'
         "判定标准：cancel=紧急打断（用户喊停/撤回/改指令的强语气）；"
         "queue=常规补充（补充说明/工具结果/定时提醒，攒到安全点批量处理）；"
         "parallel=独立轻量查询（与主任务无关的小问题，旁路即答）。")


def keyword_route(ev: Event) -> str:
    """A 关键词匹配（问题侧玩具）——组合爆炸 + 语义盲区"""
    if "停" in ev.content:
        return "cancel"
    if "天气" in ev.content:
        return "parallel"
    return "queue"


def llm_route(model, ev: Event) -> tuple[str, str]:
    """B LLM 语义路由（生产路线）——理解语气与意图，不匹配字符"""
    raw = model.invoke([HumanMessage(
        content=ROUTE + f"\n事件：来源={ev.source}，渠道={ev.channel}，内容={ev.content}"
        + (f"，背景={ev.context}" if ev.context else ""))]).content
    v = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    return v["strategy"], v.get("reason", "")


CASES = [
    (Event("user", "im", "停！取消刚才的部署"), "cancel", "常规喊停"),
    (Event("user", "im", "今天天气怎么样"), "parallel", "独立轻量"),
    (Event("user", "im", "报告用中文发我"), "queue", "常规补充约束"),
    (Event("tool", "result", "搜索返回 3 条结果"), "queue", "工具结果天然队列式"),
    (Event("timer", "cron", "每小时巡检到点"), "queue", "定时提醒不紧急"),
    (Event("user", "im", "先别继续了，马上"), "cancel", "棺材钉一：无「停」字，关键词版误入队"),
    (Event("user", "im", "别停在半路，继续跑完"), "queue", "棺材钉二：含「停」字，关键词版误取消"),
]

if __name__ == "__main__":
    model = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL,
                       timeout=60, max_retries=3)
    print(f"{'事件（来源/内容）':<38}{'关键词版':<10}{'LLM 语义路由':<12}期望")
    kw_ok = llm_ok = 0
    for ev, want, note in CASES:
        kw = keyword_route(ev)
        st, why = llm_route(model, ev)
        kw_ok += kw == want
        llm_ok += st == want
        print(f"  {ev.source}/{ev.content:<26}{kw:<10}{st:<12}{want}"
              + ("｜" + note if "棺材钉" in note else ""))
        print(f"    └ 理由：{why[:60]}")
    print(f"\n→ 关键词版 {kw_ok}/7，LLM 语义路由 {llm_ok}/7——两个棺材钉反例"
          "（无「停」字的取消指令/含「停」字的继续指令）只有语义路由能判对；"
          "结构化事件的 source 维度同时是注入防线：工具结果里藏的指令"
          "（来源=tool）不会获得用户指令的处置权重。")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（7 次真实 LLM 调用）：
#   1. 七个四维事件逐条过两套路由器：keyword_route（字符串包含判定）
#      与 llm_route（分类 LLM 输出 strategy+reason 的 JSON）
#   2. 末行统计两版命中数，棺材钉反例单独标注
# 验证内容（模型输出动态，验收看对照结构）：
#   - 关键词版在棺材钉二（含「停」的继续指令）必判 cancel——确定性错误
#   - LLM 版预期 7/7 判对（含两个反例），理由应能区分「喊停撤回」与
#     「别停在半路」的语气差异；个别摇摆属语义边界（22.3 三跑同理），
#     输出统计如实呈现
#   - Event 四维结构本身是确定性资产：source/channel 字段可供路由与
#     注入防线共用（37.2 与 study_agent/core.py 复用同款结构）
