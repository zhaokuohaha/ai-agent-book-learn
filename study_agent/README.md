# study_agent — AI 学习伙伴（贯穿性示例项目）

> 它懂你的课本（各章 README 是知识库）、懂你（学习者画像）、按计划陪你学（定时复习）、随叫随到（事件响应）——服务学习《深入理解 AI Agent》的私人 Agent。
>
> 定位：chapterN/ **学概念**（每日小实验+笔记），study_agent/ **用概念**（能力逐章长上去的载体）。双轨不互噬：概念日不动本项目，能力日先小实验讲概念、再把能力接进来。

## 项目方案（D36 与用户共同确认）

```
study_agent/
  README.md           本文件：方案 + 能力演进日志
  agent.py            入口+组装（唯一知道所有模块的地方）
  core.py             事件驱动主循环（D37 建，此后稳定不回改）
  memory.py           学习者画像（复用 common/memory_store）
  knowledge.py        课本知识库（解析各章 README 自测题；后续接 RAG）
  skills/             能力包——每章新增，向 core 注册，core 零改动
  tests/              确定性回归（每次增量必跑——D29/D35 的持续实践）
  data/               运行产物（画像等，gitignore）
```

三条纪律：①双轨不互噬；②core 稳定（新能力只进 skills/，非改不可=设计错了先修设计）；③每 skill 最小可用（不做工程化包装）。

模块边界（依赖倒置——扩展动作空间不改主循环）：

```
agent.py（组装层）
   ├── core.py      ← 不知道任何 skill：只管事件队列/句柄/抢占调度
   ├── memory.py    ← 被动服务：读写画像
   ├── knowledge.py ← 被动服务：课本检索
   └── skills/*.py  → 向 core 注册事件处理器；可调 memory/knowledge/common 工具
```

## 能力演进日志

### D36 · 骨架落地 + 第一个能力包：quiz 自测问答

- **knowledge.py**：课本即题库——解析各章 README 自测段「N. 题干？（参考答案）」为 Question 对象，新章节自动入库（确定性纯函数）
- **memory.py**：学习者画像——复用第 3 章 MemoryStore（append-only + active 取最新）；**「错题重答对即出池」天然就是间隔复习语义**，history 保留完整时间线
- **skills/quiz.py**：抽题（画像加权：未掌握优先重出 → 没答过随机 → 已掌握不出）→ 问答 → LLM 判分（意思对即可，结构化 JSON）→ 记画像
- **agent.py**：临时回合制主循环（D36 论断「回合制是交互约定」的活体现；D37 升级为事件驱动 core.py）
- **tests/test_knowledge.py**：题库解析 + 画像语义回归（错→重答对→出池）
- 复用资产：common/memory_store（第 3 章）、结构化 JSON 判分（33.1 场景同款）、配置走 common/config

### D37 · 能力包二：事件驱动 core + 定时复习

- **core.py**（正式主循环，此后稳定不回改）：Event 四维结构（与 37.1 同款）+ EventLoop（队列 + handler 注册 + 安全点批量消费）——36.1 B 版的正式化；**依赖倒置**：skills 向 core 注册处理器，core 不知道任何 skill（新能力零改动 core——「core 稳定」纪律兑现）
- **skills/daily_review.py**：set_timer 循环定时器——到点 emit 事件（source=timer：注入防线，提醒不当用户指令）；提醒自带上下文（未掌握数 + 优先重出清单，减少二次查询）
- **agent.py**：主循环从临时回合制升级为事件驱动——用户输入由输入线程 emit（与定时器同一条队列、同一种信封，D36 论断的反转落地）
- **quiz 增语义中止**：题间安全点，回答表达中止意图则中止本轮（已完成部分照常入账，不伪造未答的题）。**教训（用户退回）**：首版写了 `if reply == "停"` 精确匹配——37.1 刚证完「含『停』/不含『停』不能硬编码」转身就在答案位犯同款错；正解：中止判定并入判分调用（单次 LLM 同时输出 action+verdict，成本零增加）。答案位是自然语言层（中止意图开放语义）与 agent.py 指令层（出题/画像——CLI 协议集，确定性分发正确）的分界线
- **tests/test_core.py**：分发/批量顺序/quit 回归；首跑 bug：quit 只在批次首检查，drain 进批次的 quit 导致主循环永久阻塞——修为批内遇 quit 先处理完前面事件再退（不丢已发生的动作）
- **冒烟实证**：timer 2s 先到点 emit（积压）→ user 指令后到 → 同队列批量消费干净退出；画像里已有用户真实数据（4 题未掌握在重出池）——定时提醒带着真数据工作

### 能力路线图（待长上去）

| 天 | 课程内容 | 计划增量 |
| --- | --- | --- |
| ~~D37~~ | ~~异步事件驱动~~ | ~~core.py（事件队列+任务句柄+抢占）+ daily_review 定时复习提醒~~ ✅ 已落地（见上） |
| D38 | 语音 | voice_quiz（边说边听/可打断） |
| D39 | Computer Use | vision_explain（看截图讲 ASCII 图/表格） |
| D41-42 | W6 收尾 | report 学习周报 + 全量体检回归 |
| W7 | 评估 | 被评估对象：指标采集（正确率/复习及时性/轨迹）+ 定量评估 |
| W9 | 持续进化 | 经验沉淀进 MEMORY.md（怎么教你更有效） |

## 使用

```bash
uv run python -X utf8 study_agent/agent.py     # 交互：出题 [N] / 画像 / 题库 / 退出
uv run python -X utf8 study_agent/tests/test_knowledge.py   # 确定性回归
```

判分需 `.env`（真实 LLM）；抽题与记录全是确定性函数——能确定性计算的别让模型算。
