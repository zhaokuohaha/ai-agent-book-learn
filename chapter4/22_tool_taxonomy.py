"""chapter4/22_tool_taxonomy.py — 工具五分类：Agent 的手和脚长在哪

本实验新增的主概念是「工具五分类」：一张两轴地图（谁发起 × 作用于谁）。
D01 说 Agent = LLM + 上下文 + 工具，今天把「工具」这个黑盒展开成地图——
第 4 章从地图开始。

五类一览（前四类 Agent 主动调用，第五类是特殊款）：
    感知     Agent 主动，读信息    web_search / read_file（D12 的 read/fetch 即此类）
    执行     Agent 主动，改世界    write_file / send_email（错误代价高，安全第一）
    协作     Agent 主动，驱动他人  spawn_subagent（并行跑不相关任务/不同模型分工）
    沟通     Agent 主动，推送用户  notify_user（多渠道异步时「说话」也是显式工具）
    事件触发 注册时 Agent 主动声明，触发时外部异步回调——set_timer / monitor_shell

后端一句话对齐：感知 = GET 只读接口，执行 = POST/DELETE 写接口（鉴权+审计），
协作 = RPC 调下游服务，沟通 = 推送通知，事件触发 = MQ 消费者 / Webhook 回调。

本实验的演示重点是第五类的本质差异——控制反转：
    前四类：Agent 是主语（「我要结果」），同步等回执继续循环——D09 的
    核心循环处理的全部是它们
    事件触发：外部是主语（「到点叫醒我」）——Agent 先注册回调然后挂起/干
    别的；未来某刻事件发生，回调带着事件把 Agent 重新拉起。没有第五类，
    Agent 只能被动应答：无法定时行动，无法对新邮件/告警做出反应。

运行：uv run python -X utf8 chapter4/22_tool_taxonomy.py（纯本地，不烧 API）
"""

from dataclasses import dataclass


@dataclass
class Tool:
    name: str
    kind: str     # 五类之一；前四类的共同点 = Agent 主动调用，第五类 = 外部触发
    backend: str  # 后端类比——帮后端工程师一秒对齐


TOOLBOX = [
    Tool("web_search", "感知", "GET 只读接口"),
    Tool("read_file", "感知", "GET 只读接口"),
    Tool("write_file", "执行", "POST 写接口"),
    Tool("send_email", "执行", "POST 写接口"),
    Tool("spawn_subagent", "协作", "RPC 调下游"),
    Tool("notify_user", "沟通", "推送通知"),
    Tool("set_timer", "事件触发", "MQ 消费者/Webhook"),
]

KIND_ORDER = ["感知", "执行", "协作", "沟通", "事件触发"]
KIND_TARGET = {  # 第二轴：作用于谁（第一轴「谁发起」由 kind 自带：前四类主动）
    "感知": "读信息", "执行": "改世界", "协作": "驱动他人",
    "沟通": "推送用户", "事件触发": "唤醒 Agent",
}


def show_map():
    print("工具地图（两轴：谁发起 × 作用于谁）")
    for kind in KIND_ORDER:
        tools = [t for t in TOOLBOX if t.kind == kind]
        who = "Agent 主动" if kind != "事件触发" else "注册→外部触发"
        names = ", ".join(t.name for t in tools)
        print(f"  {who:>8}｜{kind}（{KIND_TARGET[kind]}，{tools[0].backend}）：{names}")


def call(name, **args):
    """前四类的统一形状：Agent 发起 → 同步拿结果 → 回到循环（D09 的 dispatch）"""
    print(f"  Agent ──调用──> {name}({args}) ──结果──> 回到循环")
    return f"（{name} 的结果）"


def run_proactive_demo():
    """前四类各走一遍：一次『调研→派活→保存→汇报』，主语始终是 Agent"""
    call("web_search", query="agent tool design")
    call("spawn_subagent", task="把结论并行翻译成中文")
    call("write_file", path="notes.md", content="…调研结论…")
    call("notify_user", channel="slack", text="调研完成")


TRIGGERS = {}  # 注册表：Agent 留下的「未来叫醒我」清单


def set_timer(minutes: int, note: str) -> str:
    """事件触发类的注册侧：Agent 主动调用，但拿到的不是结果，是注册凭证"""
    TRIGGERS[minutes] = note
    return f"已注册：{minutes} 分钟后唤醒"


def world_tick(minutes: int):
    """触发侧：外部世界发起（时间到/新邮件/告警）——回调把 Agent 重新拉起"""
    if note := TRIGGERS.pop(minutes, None):
        print(f"  [外部事件] {minutes} 分钟到 ──唤醒──> Agent（事件：{note}）")


def run_trigger_demo():
    print("第五类：注册时主动，触发时被动（控制反转）")
    receipt = set_timer(5, "提醒写日报")
    print(f"  Agent ──注册──> set_timer(5, '提醒写日报') → 拿到：{receipt}")
    print("  （Agent 挂起或干别的——发起权已交给外部世界，D09 循环管不到这里）")
    world_tick(5)


if __name__ == "__main__":
    show_map()
    print("\n前四类演示：主动调用（一次『调研→派活→保存→汇报』）")
    run_proactive_demo()
    print()
    run_trigger_demo()
    print("\n停一停：set_timer 为什么不像别的工具那样返回结果、只返回注册凭证？")
    print("  主动调用的语义是『现在就要结果』；定时器要的是『未来某刻叫醒我』——")
    print("  真正的『结果』在未来以事件形式到达，此刻能给的只有注册确认。")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（纯本地，不烧 API）：
#   1. show_map：TOOLBOX 按五类分桶打印——两轴坐标（谁发起 × 作用于谁）
#      + 后端类比一栏
#   2. run_proactive_demo：前四类的统一形状——Agent 发起、同步拿结果
#      （感知→协作→执行→沟通的调用链，全是 call() 一种形状）
#   3. run_trigger_demo：set_timer 注册（返回凭证）→ world_tick 模拟外部
#      事件到点 → 回调唤醒——注册与触发分离，发起权在外部
# 验证内容：
#   - 地图五行齐全，前四类标「Agent 主动」、第五类标「注册→外部触发」
#   - 主动调用四条轨迹同构（call → 结果 → 回到循环）
#   - 事件触发：注册返回的是凭证不是结果；唤醒来自 world_tick（外部），
#     不来自 Agent 的循环——控制反转的直观证据
