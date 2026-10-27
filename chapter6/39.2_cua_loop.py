"""chapter6/39.2_cua_loop.py — 感知-思考-行动闭环：点完之后，世界变了吗

本实验新增的主概念是 **Computer Use 的感知-思考-行动闭环**：截图 →
模型看图输出思考+动作 → 执行层真实点击 → 界面响应后再截图，进入下一
轮。关键区分：看懂界面（一次截图问答）≠ 完成任务（把理解和动作放进闭
环）——真正难点是**动作后确认**：每一步动作之后，重新确认现实是否仍符
合计划。后端类比：不是一次 RPC，而是不断读外部状态→决策→写回→再读
的循环；外部系统随时被别人改，写入前都要重读，不能信任缓存旧状态。

本实验把闭环"文本化"：界面渲染成结构化元素索引清单（39.1 同款——选
择题优于填空题：模型只报编号），deepseek-chat 纯文本即可驱动（结构化
索引路线恰好把视觉问题变成文本问题）。执行层是一个文本渲染的下载页
状态机，埋一个 **modal 陷阱**：点击下载后弹出不可逆确认框，弹窗期间
其他点击全部被阻断——模型下一轮必须先处理弹窗（每轮重新 render 即
"重新截图"：世界变了，计划就得跟着变）。

运行：uv run python -X utf8 chapter6/39.2_cua_loop.py（需 .env，真实 LLM）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from common.config import API_KEY, BASE_URL, MODEL

TASK = "搜索 report 数据集并下载 report.csv 文件"


class FakePage:
    """执行层：文本渲染的下载页状态机（"截图"=render，每轮动态编号）"""

    def __init__(self):
        self.query, self.results = "", []
        self.modal, self.downloaded = None, None

    def render(self) -> str:
        if self.modal:  # 弹窗打开：界面只剩弹窗（modal 阻断其他交互）
            return (f"[0] 弹窗：确认下载 {self.modal}？此操作不可逆\n"
                    "[1] 按钮 确认下载   [2] 按钮 取消")
        lines = [f"[0] 输入框 搜索数据集（当前值：{self.query!r}）", "[1] 按钮 搜索"]
        lines += [f"[{i}] 链接 下载 {n}" for i, n in enumerate(self.results, 2)]
        if self.downloaded:
            lines.append(f"状态栏：已下载 {self.downloaded} ✓")
        return "\n".join(lines)

    def ok(self) -> bool:
        return self.downloaded == "report.csv"

    def act(self, kind: str, id: int = 0, text: str = "") -> str:
        """执行动作并返回观察（执行反馈 + 重新"截图"——动作后确认的输入）"""
        if kind == "done":
            return f"任务结束：{'已下载 ' + self.downloaded if self.downloaded else '尚未下载任何文件'}"
        if self.modal:  # 弹窗期间只有确认/取消有效（先于正常分支判定）
            if kind == "click" and id == 1:
                self.downloaded, self.modal = self.modal, None
                return f"已确认下载 {self.downloaded}（不可逆操作已执行），新界面：\n{self.render()}"
            if kind == "click" and id == 2:
                self.modal = None
                return f"已取消下载，新界面：\n{self.render()}"
            return "⚠ 弹窗打开中，其他交互被阻断——先处理弹窗（动作后确认：世界变了）"
        if kind == "type" and id == 0:
            self.query = text
            return f"输入完成，新界面：\n{self.render()}"
        if kind == "click" and id == 1:
            self.results = [n for n in ("report.csv", "sales.csv") if self.query in n]
            return f"搜索完成，新界面：\n{self.render()}"
        if kind == "click" and 2 <= id < 2 + len(self.results):
            self.modal = self.results[id - 2]  # 点击后弹出确认框——世界变了
            return f"点击后弹出确认框（现实已改变，重新确认再行动）：\n{self.render()}"
        return "⚠ 无效动作目标（编号不在当前界面中）"


def make_tools(page: FakePage):
    """动作空间三类工具的最小版：GUI 操作（click/type）+ 完成（done）"""

    @tool
    def click(id: int) -> str:
        """点击当前界面中编号为 id 的元素"""
        return page.act("click", id=id)

    @tool
    def type_text(id: int, text: str) -> str:
        """向编号为 id 的输入框输入文本（覆盖原值）"""
        return page.act("type", id=id, text=text)

    @tool
    def done() -> str:
        """确认任务已全部完成时调用，结束循环"""
        return page.act("done")

    return [click, type_text, done]


if __name__ == "__main__":
    model = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=60, max_retries=3)
    page = FakePage()
    tools = make_tools(page)
    by_name = {t.name: t for t in tools}

    msgs = [
        SystemMessage(content=(
            f"你是 GUI 自动化 Agent。任务：{TASK}。\n每轮你会看到当前界面的元素清单"
            "（结构化元素索引——只报编号，不预测坐标）。调用一个动作工具，观察执行"
            "反馈中的新界面，重新确认现实后再决定下一步。界面可能弹窗，弹窗会阻断其他交互。")),
        HumanMessage(content=f"当前界面：\n{page.render()}\n请决定第一步动作。"),
    ]
    for turn in range(1, 9):
        r = model.bind_tools(tools).invoke(msgs)
        msgs.append(r)
        if not r.tool_calls:
            msgs.append(HumanMessage(content="请用工具表达动作（click/type_text/done）"))
            continue
        for tc in r.tool_calls:  # deepseek 可能并行多个 tool_call——逐个配对（37.2 教训）
            obs = by_name[tc["name"]].invoke(tc["args"])
            msgs.append(ToolMessage(content=obs, tool_call_id=tc["id"]))
            print(f"--- 轮 {turn}｜动作 {tc['name']}({tc['args']})")
            print("    " + obs.replace("\n", "\n    "))
        if any(tc["name"] == "done" for tc in r.tool_calls):
            break
    print(f"\n=== 验收：任务={TASK}")
    print(f"    已下载={page.downloaded!r}（正确目标 report.csv）→ {'✓ 通过' if page.ok else '✗ 未达'}")
    if page.ok():
        print("    闭环走通：输入→搜索→点击下载→弹窗（世界变了）→确认→done——")
        print("    每轮重新 render 即「重新截图」：动作后确认现实，不信任缓存状态。")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（真实 LLM，模型输出动态但轨迹结构确定）：
#   1. make_tools(page) 闭包绑定同一状态机——三个动作工具都作用于它
#   2. 循环 ≤8 轮：界面清单+历史 → bind_tools → tool_calls → 执行 →
#      观察含"新界面"（执行反馈即下一轮感知——wait_for_ui 后再截图的同构）
#   3. modal 陷阱：click 下载链接 → act 弹出确认框 → 其余交互返回
#      "⚠ 弹窗打开中……被阻断"——模型必须看到反馈后先处理弹窗
# 验证内容（看轨迹不看文案）：
#   - 动作链覆盖：type_text(0, "report") → click(1) 搜索 → click(2)
#     下载 report.csv → 弹窗出现 → click(1) 确认（或先试其他点击被阻断
#     后再处理弹窗——两种轨迹都算通过：都体现"动作后确认"）
#   - 终态验收：page.downloaded == "report.csv" 且最终调了 done
#   - 弹窗阻断实证：若模型在弹窗期点了非弹窗元素，反馈是 ⚠ 阻断
#     ——外部系统不会配合你的旧计划，这就是"点完之后世界变了吗"
#   - 简化注：界面是文本清单非真截图（结构化索引把视觉变文本）；执行层
#     是内存状态机非真鼠标键盘（生产用 pyautogui/Playwright）
