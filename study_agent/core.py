"""study_agent/core.py — 事件驱动主循环（D37 建，此后稳定不回改）

study_agent 的心脏（36.1 B 版事件驱动的正式化 + 37 课件骨架的落地）：
所有输入统一建模为结构化事件走同一条队列——用户消息、定时器触发都
是 Event（四维：来源/渠道/内容/上下文），push 取代轮询。

设计（三条纪律之「core 稳定」的兑现）：
    依赖倒置——handlers 按 source 注册：skills 向 core 注册处理器，
    core 不知道任何 skill 的存在；新增能力对 core 零改动
    批量消费（队列式）——run() 每次唤醒取走全部积压事件一次处理
    （安全点消费：常规事件攒批，省往返——37 课件队列式策略）
    打断（取消式）的 CLI 诚实实现——单输入流没有真并发打断，
    紧急事件在 skill 的安全点被消费（quiz 的题间检查「停」）；
    真并发场景见 36.1/37.2（块间查队列/占位符），core 不假装拥有它
"""

import queue
from dataclasses import dataclass


@dataclass
class Event:
    """四维结构化事件（37.1 同款结构）——统一信封：路由与注入的前提；
    source 维度是注入防线：timer/tool 来源的内容不获得 user 的处置权重"""

    source: str    # user / timer / system
    channel: str   # cli / cron
    content: str
    context: str = ""
    urgency: str = "normal"  # urgent / normal（紧急度路由输入）


class EventLoop:
    """事件队列 + handler 注册表 + 安全点批量消费"""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self.handlers: dict[str, object] = {}

    def on(self, source: str, fn) -> None:
        """skills 注册处理器（依赖倒置：core 不 import 任何 skill）"""
        self.handlers[source] = fn

    def emit(self, ev: Event) -> None:
        """世界主动推送——定时器线程/输入线程/未来的外部通道都走这里"""
        self.q.put(ev)

    def run(self) -> None:
        """主循环：取事件 → 制造安全点批量消费 → 分发 handler；
        quit 可能在批次中间（先处理完前面的事件再退——不丢已发生的动作）"""
        while True:
            ev = self.q.get()
            for e in [ev] + self._drain():  # 队列式：积压事件攒到安全点一次处理
                if e.source == "system" and e.content == "quit":
                    return
                if fn := self.handlers.get(e.source):
                    fn(e)

    def _drain(self) -> list[Event]:
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out
