"""chapter4/22.3_param_fidelity.py — ACI 原则三：参数保真——模型的世界不能被偷换

本实验新增的主概念是「参数保真性」：模型感知的世界与工具操作的世界不能有
系统性偏差。承接 22.2（描述说清楚了）；本实验管「参数传递的诚实」。

两种隐蔽反模式（参数层动手脚，都不声张）：
    静默输入转换：工具悄悄「修正」参数。书中案例：某版 Cursor 的替换工具
        把中文弯引号静默转成直引号——模型刚 read_file 看到弯引号，原样传参
        却报「未找到匹配」。它永远无法理解自己为什么失败：文件里有、参数
        也对，唯独工具在参数层偷换了。同样的参数必然同样失败，重试没有
        任何新信息——模型陷入无限重试。
    静默参数注入：偷偷附加模型没传的参数。某 IDE 给所有 git commit 静默
        附加选项，旧版 Git 不认识直接报错——模型看报错永远定位不到原因
        （它根本不知道那个参数存在）。

正道：参数原样传递；确需规范化（如去首尾空白），必须写进工具描述并在
返回值里明确告知，绝不偷偷改。

实验设计（问题先行，复现弯引号案例）：文件里是中文弯引号 “深度学习”，
任务是连同引号替换成 “机器学习”。Agent 循环跑两遍，replace 工具的描述
完全相同，差异全在参数层一行：
    v1 静默转换：old/new 的弯引号被悄悄转成直引号 → 带引号的正确参数
        永远「未找到」（真正拿去匹配的直引号版，文件里根本没有）→ 报错
        回显的却是模型传来的原文（弯引号）——偏差彻底隐形；模型只能
        降级绕路（去引号只换词）或反复失败
    v2 保真：参数原样 → 一次成功
首跑教训：v1/v2 不能共用同一文件——v1 的绕路替换会污染 v2 的初始状态，
v2 跑前必须重置。

运行：uv run python -X utf8 chapter4/22.3_param_fidelity.py（需 .env，约 6 次真实调用）
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL

REPLACE_DESC = ("替换文件中的文本。参数示例：replace_text(path='quote.txt', "
                "old='“深度学习”', new='“机器学习”')")


def _read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _replace_silent(path: str, old: str, new: str) -> str:
    """v1：参数层「善意」统一引号风格——弯引号悄悄转直引号（反模式一）"""
    old2 = old.replace("“", '"').replace("”", '"')
    new2 = new.replace("“", '"').replace("”", '"')
    text = Path(path).read_text(encoding="utf-8")
    if old2 not in text:
        return f"错误：文件中未找到 {old}"  # 回显模型传来的原文——偏差隐形
    Path(path).write_text(text.replace(old2, new2), encoding="utf-8")
    return f"已替换：{old2} → {new2}"


def _replace_faithful(path: str, old: str, new: str) -> str:
    """v2：保真——模型传什么就用什么，原样匹配"""
    text = Path(path).read_text(encoding="utf-8")
    if old not in text:
        return f"错误：文件中未找到 {old}"
    Path(path).write_text(text.replace(old, new), encoding="utf-8")
    return f"已替换：{old} → {new}"


def make_tools(replace_impl):
    """两版描述完全相同——差异全在参数层（行为），不在文案"""
    return [
        StructuredTool.from_function(_read_file, name="read_file", description="读取文件内容"),
        StructuredTool.from_function(replace_impl, name="replace_text", description=REPLACE_DESC),
    ]


def agent_loop(model, tools, task, max_turns=4):
    """D09 手写循环；打印每步轨迹（v1/v2 描述一致，差异只藏在工具内部）"""
    registry = {t.name: t for t in tools}
    llm = model.bind_tools(tools)
    messages = [SystemMessage(content="你是文件编辑助手。"), HumanMessage(content=task)]
    calls = []
    for turn in range(1, max_turns + 1):
        resp = llm.invoke(messages)
        messages.append(resp)
        if not resp.tool_calls:
            return calls
        for call in resp.tool_calls:
            calls.append(call["name"])
            result = registry[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
            print(f"    [turn {turn}] {call['name']}({call['args']}) → {result[:60]}")
    return calls


def show_injection():
    """反模式二：静默参数注入（本地演示，几行看穿）"""
    def git_commit_v1(message: str) -> str:
        return f"git commit -m {message!r} --no-gpg-sign"  # 模型没传的参数出现了
    def git_commit_v2(message: str, no_gpg_sign: bool = False) -> str:
        return f"git commit -m {message!r}" + (" --no-gpg-sign" if no_gpg_sign else "")
    print("反模式二：静默参数注入（本地演示）")
    print(f"  模型只传了 message，v1 实际执行：{git_commit_v1('fix: 登录页崩溃')}")
    print("  旧版 Git 不认识注入的参数直接报错——模型看报错永远定位不到（它没传过）")
    print(f"  v2 把附加项变成显式参数：{git_commit_v2('fix: 登录页崩溃')}——所见即所执行")


if __name__ == "__main__":
    model = ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0,
    )
    workdir = Path(tempfile.mkdtemp(prefix="agblearn_d22_"))
    quote = workdir / "quote.txt"
    ORIGINAL = "有人说“深度学习”是通往 AGI 的路。"
    task = f"把 {quote} 里的 “深度学习” 替换成 “机器学习”（连同两侧引号一起替换）"

    try:
        print("=== v1：静默转换（参数层把弯引号悄悄转成直引号） ===")
        quote.write_text(ORIGINAL, encoding="utf-8")
        print(f"  文件原文：{ORIGINAL}")
        calls = agent_loop(model, make_tools(_replace_silent), task)
        print(f"  轨迹：{calls}")
        final = quote.read_text(encoding="utf-8")
        print(f"  文件现状：{final}")
        if "机器学习" in final:
            print("  （看着像成功了？——但带引号的正确参数从未成功过，这是模型被")
            print("    假失败逼出来的绕路，多烧两轮才碰到同样结果）")
        else:
            print("  （替换始终未成功——同样的正确参数永远同样失败）")

        print("\n停一停：模型第一次调用（带引号，与文件一字不差）为什么会失败？")
        print("  工具在参数层把弯引号悄悄换成直引号再去匹配——真正拿去匹配的 old")
        print("  （直引号版）文件里根本没有；报错回显的却是模型传来的原文（弯引号），")
        print("  偏差彻底隐形：模型的世界与工具的世界成了平行宇宙。")

        print("\n=== v2：保真（参数原样传递） ===")
        quote.write_text(ORIGINAL, encoding="utf-8")  # 重置：清掉 v1 的污染
        calls = agent_loop(model, make_tools(_replace_faithful), task)
        print(f"  轨迹：{calls}")
        print(f"  文件现状：{quote.read_text(encoding='utf-8')}（替换成功）")
        print("  正道：确需规范化（如去首尾空白）——写进描述 + 返回里告知，绝不偷偷改")

        print()
        show_injection()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)  # 演示文件用完即焚

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（约 6 次真实 LLM 调用）：
#   1. quote.txt 写入含弯引号的句子 → v1（静默转换版 replace_text）跑
#      agent_loop，最多 4 轮——观察 read → replace 失败 → 重试的轨迹
#   2. v2（保真版）同一任务再跑 → 一次替换成功
#   3. show_injection：本地几行演示静默参数注入（git commit 被附加参数）
# 验证内容（模型输出动态，验收看轨迹不看文案）：
#   - v1：第一次 replace_text（old 含弯引号，与文件一字不差）必然「未找到」
#     ——静默转换在参数层偷换了匹配目标；之后模型或降级绕路（去引号只
#     换词——首跑实录，表面成功实则踩着假失败过来）、或重试/放弃
#   - v2：重置文件后同一任务一次成功（带引号的参数原样匹配）
#   - 首跑教训：v1/v2 曾共用文件——v1 的绕路替换把「深度学习」从文件里
#     消掉了，v2 的正确参数也无从匹配；对照实验必须重置初始状态
#   - 注入演示：v1 命令行里出现模型没传的 --no-gpg-sign，v2 所见即所执行
