"""study_agent/skills/diagram_explain.py — 讲解课本里的 ASCII 图/对照表

D39 能力（vision_explain 的降级版）：路线图原计划「看截图讲 ASCII 图/表
格」，但 DeepSeek 无视觉端点——ASCII 图和对照表本来就是**文本**，直接扫
chapter*/README.md 的小节找到它喂给 LLM 讲解，零视觉依赖即可落地
（真截图讲解待有 vision 端点时接入）。这是「章节内容讲解」能力的首块
拼图：看不懂图来找讲解——学习 agent 的核心场景之一。

设计同 knowledge.py 约定：检索确定性（块定位是纯文本匹配——能确定性
计算的别让模型算），讲解生成交给 LLM（动态输出，验证看轨迹不看文案）。
"""

from langchain_core.messages import HumanMessage

from knowledge import ROOT

EXPLAIN = """你是 AI 学习伙伴的讲解员，学生正在学这门 Agent 课程。下面是课本（{chapter}）
中标题为「{title}」的小节（含图/表）。请用中文讲解它：
- 图/表中每个角色、行列各代表什么，箭头或流向意味着什么
- 它呼应课程的哪个核心概念（一两句点透，可提与前面知识的关联）
- 指出一个最容易误解的点

小节内容：
{block}
"""


def find_sections(keyword: str) -> list[tuple[str, str, str]]:
    """扫 chapter*/README.md：以标题小节为检索单元（标题+正文到下一标题）
    → 关键词命中标题或正文 → (chapter, 标题, 正文)。标题命中优先排前
    （概念词多在标题：如「语音三范式」「快慢分离」）——各章 README 的
    图/表其实是 md 表格/正文缩进，不在 ``` 块里（实测每文件仅 1 对围栏），
    按小节检索才贴合真实结构，LLM 读 md 表格毫无压力。"""
    hit: list[tuple[int, str, str, str]] = []  # (优先级, chapter, 标题, 正文)
    for readme in sorted(ROOT.glob("chapter*/README.md")):
        chapter = readme.parent.name
        title, body = "", []

        def flush() -> None:
            if not title:
                return
            text = "\n".join(body).strip()
            if keyword in title:
                hit.append((0, chapter, title, text))
            elif keyword in text:
                hit.append((1, chapter, title, text))

        for ln in readme.read_text(encoding="utf-8").splitlines():
            if ln.startswith("#"):  # 任意层级标题——新小节开始
                flush()
                title, body = ln.lstrip("# ").strip(), []
            else:
                body.append(ln)
        flush()
    hit.sort(key=lambda h: (h[0], len(h[2])))  # 标题命中优先，同级短标题更聚焦优先
    return [(c, t, b) for _, c, t, b in hit]


def explain_block(model, keyword: str) -> None:
    """找到含关键词的小节并让 LLM 讲解（多处命中时讲最优先的一处并列出其余）"""
    blocks = find_sections(keyword)
    if not blocks:
        print("  没找到相关内容——换个关键词试试（如：时间尺度/三范式/三策略/闭环）")
        return
    chapter, title, body = blocks[0]
    print(f"  讲解 {chapter}「{title}」" + (f"（共 {len(blocks)} 块命中，讲第 1 块）" if len(blocks) > 1 else ""))
    print(model.invoke([HumanMessage(content=EXPLAIN.format(
        chapter=chapter, title=title, block=body))]).content)
