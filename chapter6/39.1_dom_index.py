"""chapter6/39.1_dom_index.py — 结构化元素索引：模型看到的"选择题"长什么样

本实验新增的主概念是「视觉定位（Grounding）三条路线」之一的**结构化元素
索引**：界面自带结构（Web 的 DOM 树 / 桌面的无障碍树）时，直接枚举可交
互元素并编号，模型输出一个 ID，系统用该元素中心坐标点击——把开放式
"找到按钮并预测坐标"（填空题）变成封闭式"从已编号元素中选一个"（选择
题）。选择题永远比填空题好答。

路线全景（选择逻辑）：
    结构可得（网页/桌面应用）→ DOM/Accessibility Tree（本实验）
    结构不可得（Photoshop/Canvas/WebGL/游戏）→ SoM 标注法或坐标预测
    SoM：分割模型切候选区域叠编号——纯视觉，不依赖界面内部结构
    坐标预测：模型直接看图报坐标（SeeClick/Claude）——需专门训练

本实验用标准库 html.parser 枚举本地 HTML 样本的可交互元素，产出
browser-use 同款的「ID + 元素 + 文本 + 坐标」文本清单——它喂给模型
的就是这个格式（CDP 拿 DOM → 检测可交互 → 编号 + 边界框 → 文本清单）。

运行：uv run python -X utf8 chapter6/39.1_dom_index.py（零 API，瞬时）
"""

from html.parser import HTMLParser

# 一个简单下载页的静态样本（生产场景是 CDP 从活页面实时拿 DOM）
HTML = """
<html><body>
<h1>数据下载中心</h1>
<input id="q" placeholder="搜索数据集">
<button id="search">搜索</button>
<a href="/dl/report.csv">下载 report.csv</a>
<a href="/dl/sales.csv">下载 sales.csv</a>
<select id="year"><option>2025</option><option>2026</option></select>
<div class="btn" onclick="openHelp()">帮助</div>
<p>普通文本不算可交互元素——不会被编号</p>
</body></html>
"""

# 可交互判据（browser-use 同款思路：a/button/input/select/textarea + onclick）
INTERACTIVE = {"a", "button", "input", "select", "textarea"}


class IndexParser(HTMLParser):
    """枚举可交互元素：tag / 可交互原因 / 文本或属性 / 中心坐标"""

    def __init__(self):
        super().__init__()
        self.items: list[dict] = []
        self._in: str | None = None  # 当前是否在可交互标签内（控制文本归属）
        self._y = 40

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in INTERACTIVE or "onclick" in a:
            self._in = tag
            self._y += 40  # 编造顺序布局坐标（真实版来自 CDP DOM.getBoxModel）
            self.items.append({
                "tag": tag,
                "why": "onclick 处理器" if "onclick" in a else "原生可交互标签",
                # 显示文本优先：标签内文本（人类看到的）→ placeholder → href
                "text": a.get("placeholder") or "",
                "href": a.get("href", ""),
                "id": a.get("id", "-"),
                "xy": (60, self._y),
            })

    def handle_endtag(self, tag):
        if self._in == tag:
            self._in = None

    def handle_data(self, data):
        if self._in and data.strip():
            self.items[-1]["text"] = data.strip()  # 标签内文本（按钮/链接名）

    def render(self) -> list[str]:
        """browser-use 同款文本清单：ID + 元素 + 文本 + 坐标"""
        lines = []
        for i, it in enumerate(self.items):
            text = it["text"] or it["href"]  # input 类无内文本，回退属性
            lines.append(f"  [{i}] <{it['tag']} id={it['id']}> {it['why']} ｜ {text} ｜ {it['xy']}")
        return lines


if __name__ == "__main__":
    print("=== 感知路线对照（选择逻辑） ===")
    print("  网页/桌面应用（结构可得）   → DOM/无障碍树：枚举+编号，稳且准（本实验）")
    print("  Photoshop/Canvas/WebGL/游戏 → SoM：分割叠编号，纯视觉不依赖结构")
    print("                                或坐标预测：看图报坐标，需专门训练")

    p = IndexParser()
    p.feed(HTML)
    print("\n=== 结构化元素索引清单（喂给模型的输入即此格式） ===")
    print("  [id] <tag id=..> 可交互原因 ｜ 文本/值 ｜ 中心坐标")
    print("\n".join(p.render()))
    print(f"\n  共 {len(p.items)} 个可交互元素——模型只需回答「点击 [3]」这级选择题，")
    print("  系统换算中心坐标执行；代价是标注信息全量进上下文（不省 token）。")
    print("  普通段落 <p> 不在清单里——只有可交互元素才编号。")

    print("\n=== 三条路线速查 ===")
    print("  SoM：不依赖任何界面内部结构，桌面软件/游戏通用；避开分割漏检误检靠编号收敛")
    print("  DOM 索引：定位准稳、零训练；拿不到结构时无能为力（Canvas/WebGL）")
    print("  坐标预测：像人一样纯视觉；但高度依赖训练分辨率——见 39.2 前的分辨率坑")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（零 API 瞬时完成）：
#   1. IndexParser.feed(HTML)：逐标签走 handle_starttag，命中可交互判据
#      （5 个原生标签 + 1 个带 onclick 的 div）即编号入清单；标签内文本
#      由 handle_data 回填（按钮/链接名），input 类取 placeholder
#   2. 打印「ID + tag + 原因 + 文本 + 坐标」清单——browser-use 喂给模型
#      的输入格式同构（CDP → 可交互检测 → 编号 + 边界框 → 文本清单）
# 验证内容（确定性，可数出来）：
#   - 清单共 6 项：input(placeholder 回填)/button(文本"搜索")/
#     a×2(链接文本)/select/id 无但 onclick 的 div("帮助")
#   - <p> 普通文本不在清单——可交互判据生效
#   - 坐标顺序递增（编造布局；真实版来自 CDP DOM.getBoxModel——简化注）
#   - 简化注：静态解析拿不到 JS 动态渲染的元素（生产用 CDP 连活页面）；
#     真实边界框/中心坐标由浏览器排版引擎给出，非顺序推算
