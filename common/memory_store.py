"""common/memory_store.py — 记忆层组件：三分类 + 仅追加 + 跨会话（第 3 章沉淀）

设计取舍（教学版记忆层，后续章节实验直接 import 复用）：
    User as Code：记忆是带类型对象（MemoryEntry），不是一坨文本——聚合、冲突
    检测交给普通函数的确定性规则（active 同 key 取最新）。能确定性计算的，
    别让模型算。
    WAL 思想：remember 只追加事件（先记日志保留证据），load 重放重建状态
    （检查点恢复）；覆盖也是追加新条目，旧条目原地不动。Mem0 v3 的取舍：
    写入时消歧（v2 四选一）一次误删不可逆，检索时消歧只增不减靠压缩治理。
    与 chapter3/15_memory_intro.py 的嵌套版（当前值 + history）构成同一问题的
    两种解。
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class MemoryEntry:
    """一条记忆 = 带类型的对象；mtype 是认知三分类（定检索路由），ts 定新旧"""

    key: str       # 检索键（snake_case 槽位，如 seat_preference）
    content: str   # 记忆内容
    mtype: str     # 认知类型：episodic（情景/操作日志）| semantic（语义/配置表）| procedural（程序/SOP 脚本）
    ts: str        # 写入时间戳（ISO 日期，字符串比较即先后）


class MemoryStore:
    """平铺 append-only 记忆库（Mem0 v3 思想的教学简化版），JSON 文件持久化"""

    def __init__(self, path):
        self.path = Path(path)
        self.entries = []
        self.load()  # 构造即加载：新会话第一件事是把盘上档案搬回内存（检查点恢复）

    def load(self):
        """跨会话的入口：进程死了记忆不死，全靠这个文件"""
        if self.path.exists():
            rows = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = [MemoryEntry(**r) for r in rows]

    def save(self):
        """落盘：每次写入立即持久化（WAL——先记日志，崩溃不丢）"""
        self.path.write_text(
            json.dumps([asdict(e) for e in self.entries], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def remember(self, key, content, mtype, ts):
        """仅追加：覆盖也是追加新条目，旧条目原地不动（误删不可逆的教训）"""
        self.entries.append(MemoryEntry(key, content, mtype, ts))
        self.save()

    def active(self):
        """当前事实视图：同 key 取 ts 最新——确定性规则，轮不到 LLM 心算"""
        latest = {}
        for e in self.entries:
            if e.key not in latest or e.ts > latest[e.key].ts:
                latest[e.key] = e
        return latest

    def history(self, key):
        """某槽位的完整时间线：时间推理的原料（"我以前是不是喜欢靠窗？"）"""
        return sorted((e for e in self.entries if e.key == key), key=lambda e: e.ts)

    def by_type(self, mtype):
        """按认知类型取当前事实：检索路由的雏形（架构图 3-4）"""
        return {k: e for k, e in self.active().items() if e.mtype == mtype}

    def render(self):
        """注入 system 的视图：只给当前事实（历史是查询用的，不占上下文）"""
        if not self.active():
            return "用户长期记忆：（暂无——首次会话）"
        lines = [f"[{e.mtype}] {k} = {e.content}" for k, e in sorted(self.active().items())]
        return "用户长期记忆（跨会话档案，非本次对话内容）：\n" + "\n".join(lines)
