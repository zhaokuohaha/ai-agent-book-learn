"""chapter6/37.2_async_compat.py — 占位符与任务句柄：同步接口上的异步兼容

本实验新增的主概念是「同步接口上表达异步语义」的两个机制（课件作业②③）：
    ① 占位符规则——打断到达时，被中断的工具调用已落轨迹但结果未到：
       同步接口要求配对合法（D30 工具环规约），就为未完成工具生成占位符
       ToolMessage（「正在后台执行，请优先处理新事件」）——维持格式合法
       的同时绝不伪造成功；「收到停止不撤销已发生的动作」——被打断的
       查询在后台其实跑完了，结果作为事件迟到（已提交的事务不自动回滚）
    ② 任务句柄 initiate_xxx——接口层解耦启动与完成：工具名暗示「发起而非
       完成」（工具命名就是给模型的 API 文档），立即返回 task_id，
       真实结果稍后以事件注入轨迹——submit 返回 job_id/Future 的同构

同步观察点（课件风险提示）：模型是否把「任务已启动/已打断」误当「已有
结果」而在真实结果到达前**编造数据**——可靠性三检查之首（结果归属）。

运行：uv run python -X utf8 chapter6/37.2_async_compat.py（需 .env，4-5 次真实调用）
"""

import sys
from itertools import count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL

SYSTEM = """你是资料调研 Agent。工具 search 是同步搜索（会阻塞直到返回）；
initiate_search 是异步发起（立即返回任务句柄，真实结果稍后以【事件】消息
到达——收到句柄时任务尚未完成，没有结果可用）。"""

RAG_LATE = ("【事件·迟到结果】刚才被打断的 RAG 查询其实在后台跑完了（已提交"
            "的事务不会自动回滚）：RAG=检索增强生成——先从外挂知识库检索相关"
            "片段，拼进提示词再生成；解决知识时效与幻觉，代价是检索质量决定上限。")
A2A_LATE = "【事件·任务结果】你发起的 A2A 协议调研任务完成：A2A=Agent-to-Agent 互操作协议——Agent Card 声明能力、任务由发起方与执行方异步协作完成、产物以 Artifact 交换。"


def show(msgs, mark: str) -> None:
    m = msgs[-1]
    kind = type(m).__name__.removesuffix("Message")
    text = (m.content or f"tool_call:{m.tool_calls[0]['name']}{m.tool_calls[0]['args']}")[:56]
    print(f"    [{kind:<9}] {text}")
    print(f"    ←（{mark}｜轨迹共 {len(msgs)} 条，配对完整）")


if __name__ == "__main__":
    model = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL,
                       timeout=60, max_retries=3)

    _seq = count(1)

    @tool
    def search(query: str) -> str:
        """同步搜索：阻塞直到返回结果（本实验由编排模拟其被打断/迟到）"""
        return "（同步搜索——本实验不会真正执行到这）"

    @tool
    def initiate_search(query: str) -> str:
        """异步发起调研：立即返回任务句柄（启动≠完成），结果稍后以【事件】消息到达"""
        return (f"已发起 t{next(_seq)}（{query}）——任务句柄；"
                "启动≠完成，暂无结果，勿编造")

    def drain(msgs, r, fn) -> None:
        """响应落账 + 工具调用逐个配对（deepseek 常并行发多个 tool_call——
        缺一个配对，下一轮 invoke 就 400：配对规约是硬约束）"""
        msgs.append(r)
        for tc in r.tool_calls:
            msgs.append(ToolMessage(content=str(fn.invoke(tc["args"])),
                                     tool_call_id=tc["id"]))

    print("=== 幕一：占位符——被打断的同步工具，配对合法且不伪造 ===")
    msgs = [SystemMessage(content=SYSTEM),
            HumanMessage(content="查一下 RAG 的核心思想，写两句话总结。")]
    r1 = model.bind_tools([search]).invoke(msgs)
    msgs.append(r1)
    show(msgs, "轮 1：模型发起同步 search")
    for tc in r1.tool_calls:  # 编排：执行期间打断到达——占位符顶替真实结果
        msgs.append(ToolMessage(content="[占位] search 被打断，未完成——请优先处理新事件",
                                 tool_call_id=tc["id"]))
    show(msgs, "占位符：未完成工具不伪造成功（逐个配对）")
    msgs.append(HumanMessage(content="停！RAG 先放放，别调工具——等你手头的资料事件。"))
    show(msgs, "打断事件（urgent→取消式）")
    r2 = model.bind_tools([search]).invoke(msgs)
    drain(msgs, r2, search)  # 防御：若模型又调 search 也配对——后续 invoke 才不 400
    show(msgs, "轮 2：重估局势（关键观察：是否编造 RAG 结果）")
    print(f"    ↳ 模型回应：{(r2.content or '')[:100]}")
    print("    ↳ 无 tool_calls 则轨迹配对完整；若模型编造了 RAG 细节即风险实证")
    # 编排：迟到结果作为事件注入（结果归属）
    msgs.append(HumanMessage(content=RAG_LATE))
    show(msgs, "迟到事件：被打断的查询后台完成")
    r3 = model.bind_tools([search]).invoke(msgs)
    msgs.append(r3)
    show(msgs, "轮 3：用真结果恢复任务")
    print(f"    ↳ 两句话总结：{(r3.content or '')[:120]}")

    print("\n=== 幕二：任务句柄 initiate_xxx——启动与完成解耦 ===")
    msgs2 = [SystemMessage(content=SYSTEM),
             HumanMessage(content="帮我调研 A2A 协议的核心机制，写两句话总结。")]
    r4 = model.bind_tools([initiate_search]).invoke(msgs2)
    drain(msgs2, r4, initiate_search)
    show(msgs2, "轮 1：发起 initiate_search（命名即文档，句柄即返回）")
    r5 = model.bind_tools([initiate_search]).invoke(msgs2)
    drain(msgs2, r5, initiate_search)
    show(msgs2, "轮 2：模型得知任务已发起（关键观察：句柄在手不编造）")
    print(f"    ↳ 模型回应：{(r5.content or '')[:100]}")
    msgs2.append(HumanMessage(content=A2A_LATE))
    show(msgs2, "任务结果事件到达（归属 t2）")
    r6 = model.bind_tools([initiate_search]).invoke(msgs2)
    msgs2.append(r6)
    show(msgs2, "轮 3：综合完成")
    print(f"    ↳ 总结：{(r6.content or '')[:120]}")

    print("\n→ 占位符维持同步轨迹格式合法且不伪造；句柄把「启动」从「完成」里解耦——")
    print("  两者都是同步接口上的异步语义补丁；原生异步（GPT-6 Astra）可省掉这层，")
    print("  但结果归属/恢复控制/多条更新综合三检查仍是应用层的责任。")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（4-6 次真实 LLM 调用，机械步骤编排合成、决策点真实 invoke）：
#   幕一：模型调 search → 编排注入占位符+打断 → 轮 2 重估 → 注入迟到
#   RAG 结果 → 轮 3 恢复总结
#   幕二：模型调 initiate_search → 句柄即返回 → 轮 2 得知已发起 →
#   结果事件到达 → 轮 3 综合
# 验证内容（模型输出动态，验收看轨迹结构与行为）：
#   - 每步 show() 打印轨迹长度与末条消息——占位符/句柄注入后配对完整
#     （LangChain 不报 tool_call 无配对错误即合法证据）
#   - 幕一轮 2：模型不应编造 RAG 细节（应表示等待/确认新指令）——若
#     编造即课件风险实证（输出如实呈现）
#   - 幕一轮 3：总结应使用迟到事件里的 RAG 内容（结果归属正确）
#   - 幕二轮 2：模型得知「启动≠完成」不应输出 A2A 调研结论（句柄≠结果）
#   - 简化注：search 的「执行中被打断」由编排模拟（同步 invoke 无法中途
#     插入事件消费）；生产在事件循环的安全点消费事件（课件骨架）
