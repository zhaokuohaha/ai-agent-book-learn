"""chapter4/27.2_subagent_hub.py — 协作原语：spawn / send / cancel / list_agents

27.1 解决了「交出去什么」，本实验解决「交出去后怎么管」——协作接口的三组原语
（类比微服务治理）：

    启动与取消  spawn_subagent 创建子 Agent、立即返回任务 ID（异步语义）；
                cancel_subagent 在任务失去意义时及时止损——类比服务调用+超时
                熔断，避免白烧 token
    消息传递    send_message_to_subagent 双向通信：子 Agent 汇报进展、请求澄清
                （类比请求-响应+回调）
    发现        list_agents 列出可用 Agent 的职责和状态——和 MCP 的 tools/list
                同一思路（只不过列的是 Agent），相当于服务注册中心

这组原语上跑出四种协作形态，本实验的轨迹里都能指认：
    同步调用   等结果再继续——适合快任务（文末对照说明）
    异步调用   spawn 立即返回、完成时再收——MQ 异步削峰模式的 Agent 版（主线）
    流式协作   子 Agent 持续发增量消息——每步往 outbox 推一条进展（主线）
    多轮交互   send 补充指令、步间消费 inbox——对话式协作（主线）

实现用 threading + queue（课件骨架的加固版）：子 Agent 在独立线程真实调 LLM
分步执行，主线程继续干别的、轮询收集进展——异步的红利看得见（时间线打印实证）。

运行：uv run python -X utf8 chapter4/27.2_subagent_hub.py（需 .env，约 5 次调用）
"""

import queue
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL

PERSONA = "你是子 Agent，只做被指派的分步分析；每步输出一句话结论。"
DATA = ("（模拟数据）昨日订单 1100 单、异常 177（P0 12/P1 45/P2 120）、退款 23 单；"
        "近 7 日异常率 12%~18% 波动；全年月报明细已就绪。")
STEPS = {
    "analyst": ["统计异常等级分布", "对比近 7 日趋势", "给出改进建议"],
    "reporter": ["汇总月度指标", "绘制季度对比", "生成结论摘要", "校对数据", "输出终版"],
}


class SubagentHub:
    """三组原语的骨架：spawn（启动）/ send（消息）/ cancel（止损）/ list_agents（发现）"""

    def __init__(self, model: ChatOpenAI):
        self.model = model
        self.registry = {  # 注册中心：谁在、能干什么（发现原语的数据源）
            "analyst": "小安 · 数据分析（订单/指标统计）",
            "reporter": "小林 · 报表生成（多步长任务）",
        }
        self.agents = {}

    def list_agents(self) -> dict:
        return self.registry

    def spawn(self, kind: str, task: str) -> str:
        """启动原语：拉起线程、立即返回任务 ID——不等结果（异步语义）"""
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        box = {"inbox": queue.Queue(), "outbox": queue.Queue(), "cancelled": False}
        box["thread"] = threading.Thread(target=self._run, args=(sid, kind, task, box), daemon=True)
        self.agents[sid] = box
        box["thread"].start()
        return sid

    def send(self, sid: str, msg: str) -> None:
        """消息原语：主→子（子→主走 outbox）——子 Agent 在步间消费"""
        self.agents[sid]["inbox"].put(msg)

    def cancel(self, sid: str) -> None:
        """取消原语：打上止损标记——子 Agent 下一步前检查、余步不再执行"""
        self.agents[sid]["cancelled"] = True

    def _run(self, sid: str, kind: str, task: str, box: dict) -> None:
        """子 Agent 本体（独立线程）：分步执行，每步查取消、收消息、推进展"""
        steps = STEPS[kind]
        for i, step in enumerate(steps, 1):
            if box["cancelled"]:
                box["outbox"].put(f"收到取消，第 {i}/{len(steps)} 步起停止（止损省下 {len(steps) - i + 1} 次调用）")
                return
            while not box["inbox"].empty():  # 多轮交互：步间消费补充指令
                extra = box["inbox"].get()
                box["outbox"].put(f"收到主 Agent 补充：{extra}，并入后续步")
                task += f"；补充要求：{extra}"
            resp = self.model.invoke([HumanMessage(f"{PERSONA}\n{DATA}\n任务:{task}\n当前步:{step}")])
            box["outbox"].put(f"步骤 {i}/{len(steps)}（{step}）→ {str(resp.content)[:36]}")


def drain(hub: SubagentHub, sid_rep: str) -> None:
    """主线程：轮询收集各子 Agent 的进展（流式协作的接收端）；报表任务第 2 步后撤销"""
    active = set(hub.agents)
    while active:
        for sid in list(active):
            box = hub.agents[sid]
            try:
                msg = box["outbox"].get_nowait()
            except queue.Empty:
                msg = None
            if msg:
                print(f"  [{sid}] {msg}")
                if sid == sid_rep and "步骤 2/" in msg:
                    hub.cancel(sid_rep)
                    print(f"  [主 Agent] 报表需求已作废 → cancel({sid_rep})")
            elif not box["thread"].is_alive():  # 线程收工且消息取净
                active.discard(sid)
        time.sleep(0.05)


if __name__ == "__main__":
    model = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0)
    hub = SubagentHub(model)

    print("=== ① 发现：list_agents（注册中心里谁在）===")
    for key, desc in hub.list_agents().items():
        print(f"  {key}: {desc}")

    print("\n=== ② 异步：spawn 立即返回；③ send 追加指令；④ 再拉一个长任务 ===")
    t0 = time.perf_counter()
    sid_an = hub.spawn("analyst", "分析昨日订单异常分布并给建议")
    print(f"  spawn → {sid_an}（耗时 {time.perf_counter() - t0:.3f}s——只是拉起，不是完成）")
    hub.send(sid_an, "顺带看一眼退款订单占比")
    print(f"  send → {sid_an}（已投递，子 Agent 步间消费）")
    sid_rep = hub.spawn("reporter", "生成全年订单报表")
    print(f"  spawn → {sid_rep}（长任务，5 步）")

    print("\n=== 进展流（主线程边干边收——流式协作）===")
    drain(hub, sid_rep)

    print("\n看轨迹：spawn 立即返回、主线程不等 → 进展流实时进出 → reporter 中途被")
    print("cancel（在途步骤跑完、下个检查点生效——取消是「协作式」的）→ analyst 全程")
    print("走完。同步形态对照：spawn 后原地等结果就是同步调用（快任务可用）——异步")
    print("把「等待」还给了主线程")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（约 5 次真实调用，2 个真实子 Agent 线程）：
#   1. list_agents 打印注册表（发现原语）
#   2. spawn(analyst)：立即返回 ID（耗时 ≈0.00Xs——异步语义实证）
#   3. send 补充指令 → analyst 步间消费（多轮交互）
#   4. spawn(reporter) 长任务 → 进展流实时打印（流式协作）
#   5. reporter 到第 2 步时 cancel → 余 3 步不执行（止损）
# 验证内容：
#   - spawn 的耗时打印 ≈0（不阻塞）——对照「等结果才继续」的同步调用
#   - analyst 的步骤 2/3 应体现「退款订单」补充要求——send 被消费
#   - reporter 的消息止于「收到取消，第 N/5 步起停止」——之后没有新步骤消息；
#     实际停在 3 或 4 步：主线程收到「步骤 2」消息再 cancel 时，子 Agent 往往已
#     冲进第 3 步（在途）——取消是协作式的，检查点粒度决定止损速度（真实系统
#     cancel 也总能省「剩余步骤」、省不掉「当前步骤」）
#   - 两个子 Agent 的进展交错出现——独立线程在并发跑
#   - cancel 语义 = 及时止损：省下的不是时间，是真金白银的 token
