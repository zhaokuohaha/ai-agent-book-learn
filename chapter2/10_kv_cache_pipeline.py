"""chapter2/10_kv_cache_pipeline.py — KV Cache 友好的上下文设计：前缀冻结，轨迹只追加

核心概念（原书第 2 章 · W2 Day 10）：

KV Cache 直觉
    模型每生成一个 token 都要"回头看"前文所有 token 的注意力中间结果
    （K/V 向量）：无缓存时计算量随上下文长度平方级增长（正比 N²）；KV Cache
    让每个 token 的 K/V 只在第一次进入上下文时算一次，之后一直复用。
    前提：被复用的前缀必须 token 级不变——从第一个不同的 token 起，缓存
    全部作废、全部重算；变动点越靠前，代价越大。
    Docker 分层构建类比：改 Dockerfile 第 3 行，第 3 层起全部重建；只在末尾
    追加新层，前面层全部命中。messages 数组就是 Dockerfile——system + tools
    是最底下、最不该动的基础层。

原书真实事故
    客服 Agent 在 system prompt 加了一行 Current time: {{now}} 注入实时
    时间。次日告警：首 token 延迟 0.5s → 3-5s，月推理账单近乎翻倍。模型没换、
    代码没 bug——只是时间戳让每次请求的前缀从那一行起全变了（实验 B 复现）。

三条铁律（记不住原理也要记住这三条）
    1. system + tools 定稿即冻结：多一个空格也会让缓存失效；不按使用热度
       动态调整工具顺序（工具定义动辄数百 token，重排=大面积失效）
    2. 动态信息（时间/余额/用户状态）永远作为新消息追加到末尾，绝不回头改前缀
    3. 用标准消息格式，不自拼 "USER: ... ASSISTANT: ..." 纯文本——Chat
       Template（信封格式）翻译成训练时见过的 token 序列，偏离会破坏模型
       的多步思维链保留；缓存只认 token 字节，格式稳定的拼接照样命中，
       但拼接不稳定（前缀注入动态内容）时缓存随之失效

两个层级的缓存
    KV Cache：模型内部机制，加速单次请求内的 token 生成
    Prompt Cache：服务商跨请求复用相同前缀，命中部分通常只按约 1/10 计费
    （DeepSeek 自动开启；本实验从 usage.prompt_cache_hit_tokens 读真实命中数）
    推论：缓存不是事后优化，而是架构约束——Claude Code 把动态元素放在缓存
    边界之后（N 个二值条件会产生 2^N 种缓存键）；子 Agent 继承父上下文时
    要求字节级对齐（作业 3：对齐=命中同一份 Prompt Cache，差一个字节就从
    该字节起全部重算）；工具结果的替换字符串首次出现即冻结，重启也用同一份。

作业落点：
    1. ChatPipeline（原书 stable_prefix + stable_tools 骨架）→ 实验 A：
       前缀冻结、轨迹只追加，每轮打印前缀指纹 + 与上轮公共前缀 + 真实缓存命中
    2. 审查 D09 玩具 Agent 的"会变的东西" → 见下方清单
    3. 思考题答案 → 上文"子 Agent 字节级对齐"处

作业 2 审查清单：D09 管道里的"会变的东西"
    ① 时间戳：无（没往 system 塞）✓  ② 用户状态/余额注入：无 ✓
    ③ 工具顺序：dict 保序、列表固定 ✓（隐患：若按"使用热度"重排就破坏缓存，
       本实验 TOOL_ORDER 冻结示范正解）  ④ 动态 few-shot：无 ✓
    ⑤ 滑动窗口截断：无（轨迹只增）✓
    结论：D09 天然合规；真正的诱惑是"顺手加个时间戳"——实验 B 实测其代价。

代码风格说明：本文件按 AGENTS.md 的 Python 规范重写——依赖经构造器注入而非
    模块级全局；正/反例共用同一份轨迹数据（重复即坏味道）；公共前缀用标准库
    os.path.commonprefix（惯用法替代手写循环）；tuple 表达"顺序冻结"的不可变语义。

运行：uv run python -X utf8 chapter2/10_kv_cache_pipeline.py（需 .env，真实 API）
"""

import hashlib
import json
import sys
from os.path import commonprefix
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL


# ---------- stub 工具（tuple 表达顺序冻结：不按使用热度重排——铁律 1） ----------


@tool
def get_weather(city: str) -> str:
    """查询城市天气（模拟数据）"""
    return f"{city}：晴，26°C（模拟数据）"


@tool
def get_time(city: str) -> str:
    """查询城市当前时间（模拟数据）"""
    return f"{city}：12:00（模拟数据）"


TOOL_ORDER = (get_weather, get_time)

# 前缀要超过缓存最小单位（DeepSeek 为 64 token）才有缓存可谈——故意写得长
SYSTEM_PROMPT = """你是"随身助手"，一个严谨的学习型 Agent。职责与规范：
1. 天气、时间等事实性问题必须先调用工具查询，严禁凭记忆编造；
2. 工具返回的是模拟数据，回答时如实转述并注明"模拟数据"；
3. 回答用简洁的中文，先给结论再给细节；
4. 与任务无关的请求，礼貌拒答并说明原因。
本提示词与工具清单构成静态前缀：任何改动（哪怕一个空格）都会让服务商的
前缀缓存从变动点起全部失效——这段话自己就是最好的备忘。"""


# ---------- 字节级对比的小工具（纯函数，无副作用） ----------


def dump(messages):
    """逐条消息序列化拼接：追加新消息时，旧消息的字节纹丝不动
    （整列表 dumps 的结尾 ] 会污染公共前缀对比）"""
    return "\n".join(json.dumps(m, ensure_ascii=False, sort_keys=True) for m in messages)


def common_prefix_len(old, new):
    """公共前缀长度：缓存可复用的字节范围，止于首个差异字符"""
    return len(commonprefix([old, new]))


def to_api_message(resp):
    """AIMessage → API 视角 dict（tool_calls 原样放回：轨迹环环相扣，缺环即 400）"""
    message = {"role": "assistant", "content": resp.content}
    if resp.tool_calls:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call["args"], ensure_ascii=False),
                },
            }
            for call in resp.tool_calls
        ]
    return message


def cache_usage(resp):
    """DeepSeek usage 里的真实缓存命中数（Prompt Cache：跨请求复用相同前缀）"""
    usage = resp.response_metadata.get("token_usage") or {}
    return usage.get("prompt_cache_hit_tokens"), usage.get("prompt_cache_miss_tokens")


class ChatPipeline:
    """前缀冻结、轨迹只追加的消息管道（三条铁律的代码落点）

    依赖经构造器注入（模型、工具清单）；构造后 prefix 与 tools 只读（铁律 1），
    trajectory 唯一写入口是 append（铁律 2），消息用标准 role dict（铁律 3，
    交给 Chat Template 翻译成训练时见过的 token 序列）。
    """

    def __init__(self, system_prompt, tool_list, model):
        self._registry = {t.name: t for t in tool_list}  # 名字 → 工具：框架执行用
        self._model = model.bind_tools(tool_list)        # tools 字段：静态前缀的一部分
        self._tools_schema = [convert_to_openai_tool(t) for t in tool_list]
        self._prefix = [{"role": "system", "content": system_prompt}]
        self._trajectory = []

    def append(self, message):
        self._trajectory.append(message)  # 轨迹唯一写入口：只增不改

    def payload(self):
        return self._prefix + self._trajectory  # 无状态 API：每轮全量拼装

    def fingerprint(self):
        """前缀指纹（system + tools 的 sha256）：全程应纹丝不动"""
        blob = json.dumps({"system": self._prefix, "tools": self._tools_schema},
                          ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def ask(self, question, max_turns=4):
        """Agent 核心循环：模型决策，框架执行；每轮全量重发并核对缓存特征"""
        self.append({"role": "user", "content": question})
        last_dump = None
        for turn in range(1, max_turns + 1):
            current_dump = dump(self.payload())
            reused = (f"与上轮公共前缀 {common_prefix_len(last_dump, current_dump)}"
                      f"/{len(last_dump)} 字符" if last_dump else "冷缓存")
            print(f"[turn {turn}] 前缀指纹 {self.fingerprint()}｜"
                  f"轨迹 {' → '.join(m['role'] for m in self._trajectory)}｜{reused}")
            response = self._model.invoke(self.payload())
            self.append(to_api_message(response))  # 决策也要记账，否则回执无处配对
            hit, miss = cache_usage(response)
            print(f"  Prompt Cache：hit={hit}, miss={miss}（命中部分约 1/10 计费）")
            if not response.tool_calls:  # 无调用 = 模型决定收工
                print(f"  [回答] {response.text}")
                return
            for call in response.tool_calls:  # 模型只报意图，动手的是框架
                observation = self._registry[call["name"]].invoke(call)
                self.append({"role": "tool", "tool_call_id": call["id"],
                             "content": observation.content})
                print(f"  [框架执行] {call['name']}({call['args']}) → {observation.content}")
            last_dump = current_dump
        print("（达到轮数上限，任务未完成）")


def experiment_b():
    """反模式对照：同一份轨迹，正解冻结前缀、事故每轮改写 system（本地对比，不调 API）"""
    history = [
        {"role": "user", "content": "上海天气？"},
        {"role": "assistant", "content": "先查工具"},
        {"role": "tool", "tool_call_id": "c1", "content": "晴 26°C"},
    ]

    def stamped(round_no):  # 事故模式：时间戳每轮改写 system（原书的 Current time: {{now}}）
        return f"你是助手。Current time: 10:00:{round_no:02d}"

    scenarios = {
        "正解：冻结前缀+末尾追加": (
            [{"role": "system", "content": SYSTEM_PROMPT}, history[0]],
            [{"role": "system", "content": SYSTEM_PROMPT}, *history],
        ),
        "事故：时间戳进 system": (
            [{"role": "system", "content": stamped(1)}, history[0]],
            [{"role": "system", "content": stamped(2)}, *history],
        ),
    }
    for name, (before, after) in scenarios.items():
        old, new = dump(before), dump(after)
        reusable = common_prefix_len(old, new)
        verdict = ("全部旧内容字节级命中" if reusable == len(old)
                   else f"首异点在第 {reusable} 字符（system 内），其后缓存全灭")
        print(f"  {name}：公共前缀 {reusable}/{len(old)} 字符——{verdict}")
    print("  正解即铁律 2：动态信息（如时间）作为新消息追加到末尾，绝不回头改前缀。")


if __name__ == "__main__":
    print("=== 实验 A：ChatPipeline——前缀冻结，轨迹只追加 ===")
    model = ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0,
    )
    ChatPipeline(SYSTEM_PROMPT, TOOL_ORDER, model).ask(
        "查一下上海的天气，顺便告诉我现在几点了")
    print("\n=== 实验 B：反模式——时间戳塞进 system ===")
    experiment_b()
