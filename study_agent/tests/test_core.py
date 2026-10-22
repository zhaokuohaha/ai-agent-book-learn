"""study_agent/tests/test_core.py — 事件循环的确定性回归（D37 能力包保护）

保护对象：core.EventLoop 的三个语义——handler 按 source 分发（依赖倒置
不破）、批量消费顺序（队列式：积压事件攒到安全点按序处理）、quit 退出。
纯确定性零 LLM，秒级可跑。skills 回归见 test_knowledge.py。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # study_agent/

from core import Event, EventLoop

# ---- 1. handler 注册与分发（core 不知道 handler 内容，按 source 路由） ----
loop = EventLoop()
log: list[str] = []
loop.on("user", lambda ev: log.append(f"user:{ev.content}"))
loop.on("timer", lambda ev: log.append(f"timer:{ev.content}"))
loop.on("system", lambda ev: log.append(f"system:{ev.content}"))  # quit 也走 handler 前的判断

# ---- 2. 批量消费：积压事件攒到安全点一次处理，顺序保持 ----
for c in ("出题 2", "画像"):
    loop.emit(Event("user", "cli", c))
loop.emit(Event("timer", "cron", "复习提醒"))
loop.emit(Event("system", "cli", "quit"))
loop.run()
assert log == ["user:出题 2", "user:画像", "timer:复习提醒"], f"分发/顺序错误：{log}"

# ---- 3. 事件结构：四维字段 + 注入防线语义（source 区分处置权重） ----
ev = Event(source="timer", channel="cron", content="x", context="y", urgency="normal")
assert (ev.source, ev.channel, ev.context, ev.urgency) == ("timer", "cron", "y", "normal")
assert Event("user", "cli", "停！", urgency="urgent").urgency == "urgent"

# ---- 4. 未注册来源的事件不崩（外部新事件源接入前的容错） ----
loop2 = EventLoop()
loop2.on("user", lambda ev: None)
loop2.emit(Event("mail", "imap", "新邮件"))  # 无 handler：静默积压不崩
loop2.emit(Event("system", "cli", "quit"))
loop2.run()

print("全部通过：handler 分发（依赖倒置）、批量消费顺序、quit 退出、四维事件结构、未注册来源容错")
