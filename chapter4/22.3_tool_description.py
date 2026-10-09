"""chapter4/22.3_tool_description.py — ACI 原则二：描述的艺术——写「何时用」而非「能做什么」

本实验新增的主概念是「描述三件套」。承接 22.2（定了形态：工具做多大），
本实验定「说明书怎么写」——描述是工具与模型之间唯一的接口文档，模型选
工具的全部依据就是那几行字。

描述三件套（v2 注册表每个工具都补齐——作业②落点）：
    1. 何时用：触发条件（「需要实时信息或未知事实时用」），不是功能罗列
    2. 边界：做不到什么、失败时怎么办——大多数调用失败的根因是模型
       不知道工具「不能」做什么，清晰列出边界往往比描述能力更重要
    3. 参数示例：RFC3339 不如「RFC3339，例如 2024-03-15T14:30:00Z」——
       模型在复杂任务中只能分出一小部分注意力确认格式，例子让它直接套用；
       每工具附 1-5 个真实调用示例，某些基准准确率约 72% → 90%

好工具的 5 条原则（作业①，本仓库玩具 Agent 注册表的宪法）：
    1. 通用优先：能用通用执行器解决就不写专用工具（22.2）
    2. 粒度整合：同族专用工具合并成一个，参数区分形态（22.2）
    3. 描述三件套：何时用 + 边界 + 参数示例（本实验）
    4. 参数保真：工具看到什么就用什么，不静默转换/注入（22.4）
    5. 四维决策：安全权限/参数复杂度/变更频率/模型能力定形态（22.2）

实验设计（问题先行）：同一 Agent、同一工具实现、同一任务，只换描述——
任务要求「逐字完整」并给出 80MB 事实：v1 描述没写 50MB 上限，模型没有
失败预判，必然先撞 read_document 的墙；v2 描述把上限知识交给模型，一步
推理出 80 > 50 直接走可行路径。看两版轨迹的失败调用差。
调试口诀：Agent 频繁选错工具，先查描述，再怀疑模型——修描述的投入
产出比远高于换更强的模型。

首跑翻车记：任务曾用「内容给我看看」——模型看 80MB 直接选了 summarize，
v1 根本不踩坑（强模型有自由裁量权，弱描述也可能被常识救场）；改成
「逐字完整」锁死 read 的语义必然性后，对照才成立。

运行：uv run python -X utf8 chapter4/22.3_tool_description.py（需 .env，约 4 次真实调用）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL

FILE_MB = {"report.pdf": 80}  # stub 文件元数据：不真造 80MB 文件，工具按它判断
SUMMARY = "（stub 摘要）Q3 销售报告要点：营收增长 23%，华东区贡献过半；Q4 主推两条新产品线。"


def _read_document(path: str, file_type: str) -> str:
    """v1/v2 共用同一实现——两版实验唯一的差异是描述，这是本实验的控制变量"""
    size = FILE_MB.get(path, 1)
    if size > 50:
        return f"错误：文件过大（{size}MB，超过 50MB 上限）"  # 干巴巴的报错，不指路
    return f"（{path} 全文，约 {size}MB……）"


def _summarize_document(path: str) -> str:
    return SUMMARY


DESCRIPTIONS = {
    "v1": {  # 能做什么式：功能罗列——模型不知道工具「不能」做什么
        "read_document": "读取文档文本。",
        "summarize_document": "总结文档内容。",
    },
    "v2": {  # 作业②：三件套（何时用 + 边界 + 参数示例）
        "read_document": (
            "读取文档的完整文本。何时用：需要逐字引用原文或精确分析时。"
            "边界：文件超过 50MB 时本工具必然失败——不要调用，改用 "
            "summarize_document；只返回文本，不含图片和排版。"
            "参数示例：read_document(path='notes.txt', file_type='txt')"
        ),
        "summarize_document": (
            "生成文档摘要。何时用：只需了解文档大意，或文件超过 50MB 无法整读时。"
            "边界：不返回原文，无法用于精确引用。"
            "参数示例：summarize_document(path='report.pdf')"
        ),
    },
}


def make_tools(version):
    """同一实现、不同描述——描述是 ACI 的全部：模型眼里「工具 = 描述 + schema」"""
    d = DESCRIPTIONS[version]
    return [
        StructuredTool.from_function(
            _read_document, name="read_document", description=d["read_document"]),
        StructuredTool.from_function(
            _summarize_document, name="summarize_document", description=d["summarize_document"]),
    ]


def agent_loop(model, tools, task, max_turns=4):
    """D09 手写循环；打印每步轨迹，返回调用名序列"""
    registry = {t.name: t for t in tools}
    llm = model.bind_tools(tools)
    messages = [
        SystemMessage(content="你是文档助手，按工具描述选择合适的工具。"),
        HumanMessage(content=task),
    ]
    calls = []
    for turn in range(1, max_turns + 1):
        resp = llm.invoke(messages)
        messages.append(resp)
        if not resp.tool_calls:
            return calls, resp.content
        for call in resp.tool_calls:
            calls.append(call["name"])
            result = registry[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
            print(f"    [turn {turn}] {call['name']}({call['args']}) → {result[:50]}")
    return calls, "（达到轮数上限）"


if __name__ == "__main__":
    model = ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0,
    )
    # 「逐字完整」锁死 read_document 的语义必然性（首跑教训：任务含糊时
    # 强模型会凭常识绕开坑，对照就消失了）；80MB 是给 v2 用的推理素材。
    # 不加「摘要也行」兜底：有退路时模型倾向先冲主路径，预判就没了
    task = "把 report.pdf（约 80MB）的完整内容逐字读给我。"

    print("=== v1：能做什么式描述（「读取文档文本。」/「总结文档内容。」） ===")
    calls_v1, ans_v1 = agent_loop(model, make_tools("v1"), task)
    print(f"  → 回答节选：{ans_v1[:80]}")

    print("\n=== v2：三件套式描述（何时用 + 边界 + 参数示例） ===")
    calls_v2, ans_v2 = agent_loop(model, make_tools("v2"), task)
    print(f"  → 回答节选：{ans_v2[:80]}")

    print("\n[对照] 任务与工具实现完全相同，只有描述不同：")
    print(f"  v1 轨迹 {calls_v1}——「逐字」语义必然先选 read_document，v1 描述没给")
    print("     「50MB 上限」知识：模型只能撞墙后从报错里学（多烧一轮 + 失败记录）")
    print(f"  v2 轨迹 {calls_v2}——描述给了上限知识 + 替代路径：模型预判 80 > 50")
    print("     直接 summarize 一步到位，或零调用进入协商——无论哪种，注定失败的")
    print("     调用都省了；判据：v2 无失败调用，回答能引用 50MB 上限（知识只能")
    print("     来自描述而非报错——边界生效的铁证）")
    print("  调试口诀：Agent 频繁选错工具，先查描述，再怀疑模型")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（约 4 次真实 LLM 调用）：
#   1. v1：make_tools("v1") 构造「能做什么」式描述的两工具 → agent_loop 跑
#      「读 80MB 的 report.pdf」→ 打印每步轨迹（含失败的调用）
#   2. v2：同一实现换三件套描述 → 同一任务再跑 → 对照两版轨迹
# 验证内容（模型输出动态，验收看要点不看文案）：
#   - v1 轨迹第一步是 read_document 且收到「文件过大」报错（「逐字完整」
#     锁死语义必然性；v1 无预判知识，必然撞墙），之后才自救切换
#   - v2 无失败调用：直接 summarize_document 一步到位，或零调用直接协商
#   - 三跑实录：含糊任务 v1 都不踩坑（常识绕行）；明确任务两版都先试；
#     描述强化「不要尝试」后 v2 才稳定预判——模型策略会摇摆，但 v2 的
#     回答始终能引用「50MB 上限」（知识来自描述而非报错）——这是边界
#     生效的铁证；修任务措辞与描述强度，正是在「调描述而非换模型」
#   - 更深一层：v1 撞墙后自救与否也摇摆（有的跑切 summarize、有的跑
#     直接放弃）——坏描述下行为不可预期；v2 每跑都零失败——描述质量
#     决定行为的可预期性
