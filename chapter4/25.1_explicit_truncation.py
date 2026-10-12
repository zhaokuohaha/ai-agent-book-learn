"""chapter4/25.1_explicit_truncation.py — 显式截断：静默截断之恶（感知工具·读取类量控）

本实验新增的主概念是「感知工具」——Agent 的眼睛：web_search / fetch_url /
read_file 这类只读取证工具，后端视角就是一组 GET 接口（22.1 地图第一类）。
只读天然带来可缓存、可并行两大红利（25.3 实测），头号风险则是返回量失控：
一次抓取几千字符起步，直接塞上下文等于 OOM。本实验看读取类的解法——
三板斧之三「offset/limit + 显式截断提示」。

最危险的是静默截断：Agent 以为看到全部，基于残缺信息自信决策——类比后端
接口悄悄丢字段还返回成功，查起 bug 想打人。模型无法自救：它没有任何
「还有下文」的信号——「不知道自己不知道」是最隐蔽的失败形态。

对照实验（变量是「披露」，A 是生产常见的朴素端点，B 是课件修正版）：同一
本地 HTTP 服务、同一份长日报、同一 LIMIT 切点、同一工具名 fetch_url——
    A 朴素版：返回 text[:LIMIT]，不说话、也不提供分页参数——静默截断的
    完整形态：模型既没信号（末行提示）也没把手（offset），想自救无从下手；
    切点特意选在段落边界，输出像一份完整报告，连「句子断了」的线索都没有
    B 显式版：offset/limit 参数（把手）+ 末尾追加「[已显示 x/y 字符，
    传 offset=x 继续]」（信号）——描述里也写明截断行为（ACI 三件套，D22.3）
日报里埋了陷阱：前四节全是「正常」，数据库坏盘事件藏在切点之后。

运行：uv run python -X utf8 chapter4/25.1_explicit_truncation.py
（需 .env，约 4-6 次真实调用；页面由本地 HTTP 服务托管，不依赖外网）
"""

import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL

# ---- 长日报：前四节全是「正常」，坏盘事件藏在第五节（切点之后） ----

SECTIONS = [
    "# 全栈服务状态日报（2026-10-10）",
    """## 执行摘要
今日全站可用性 99.98%，核心交易链路零故障。流量峰值出现在 20:00-21:00，
较昨日增长 12%，自动扩容预案未触发；三大开放平台的回调对接全部正常，
安全侧无新增告警。整体运行平稳。""",
    """## 应用服务
- api-gateway：QPS 峰值 8.4k，p99 延迟 42ms，错误率 0.01%，正常。
- auth-service：登录成功率 99.9%，token 刷新队列无积压，正常。
- order-service：下单成功率 99.97%，与上周持平，正常。
- pay-service：五个支付渠道全部畅通，日终对账零差异，正常。
- search-service：索引更新延迟 3s，处于正常水位，正常。""",
    """## 基础设施
- k8s 集群：6 节点全部 Ready，资源水位 61%，无驱逐事件。
- 消息队列：堆积峰值 1.2k 条，均在 10 秒内消化完毕。
- 缓存集群：Redis 内存碎片率 1.3，热点 key 无大变更。
- 数据库主库：连接数 210/500，主从复制延迟 0.3s，均在正常范围。""",
    """## 数据库事件（需重点关注）
分片 db-shard-3 本周已发生 3 次写入延迟飙升，p99 从 80ms 冲高至 2.4s，
根因定位为该节点 SSD 出现个别坏块，读重试在写路径上被放大。应急措施：
已将 12% 流量切换至 db-shard-4，延迟回落至 110ms。计划周六 02:00-02:10
更换坏盘，期间该分片只读约 5 分钟。注意：修复前请勿在该分片上调度
大批量写入任务。""",
    """## 行动项
1. 值班同学周六 01:30 到位，换盘前确认从库同步位点。
2. 本周内完成其余分片的 SSD SMART 健康巡检。
3. 离线回填等大批量写入任务统一改期到周日之后执行。""",
]

DOC = "\n\n".join(SECTIONS)
LIMIT = len("\n\n".join(SECTIONS[:4])) + 2  # 切点恰在「数据库事件」节前的段落边界

SESSION = requests.Session()
SESSION.trust_env = False  # 本地回环不走系统代理（本机常挂代理会拦截 127.0.0.1）


class ReportHandler(BaseHTTPRequestHandler):
    """把长日报挂到本地端口——感知工具抓的是真 HTTP，不是内存字符串"""

    def do_GET(self):
        body = DOC.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # 静默访问日志，别污染实验输出
        pass


def make_naive():
    """A 组：朴素端点——没有分页参数（把手）也不说话，静默截断的完整形态"""

    def _fetch(url: str) -> str:
        """抓取网页纯文本。"""
        return SESSION.get(url, timeout=5).text[:LIMIT]

    return StructuredTool.from_function(_fetch, name="fetch_url")


def make_honest():
    """B 组：课件修正版——offset/limit 把手 + 末行续读信号 + 描述写明行为"""

    def _fetch(url: str, offset: int = 0, limit: int = LIMIT) -> str:
        """抓取网页纯文本；超过 limit 会截断，末尾注明剩余量与续读方式。"""
        text = SESSION.get(url, timeout=5).text
        chunk = text[offset:offset + limit]
        if offset + limit < len(text):  # 绝不静默丢内容
            chunk += f"\n...[已显示 {offset + limit}/{len(text)} 字符，传 offset={offset + limit} 继续]"
        return chunk

    return StructuredTool.from_function(_fetch, name="fetch_url")


SYSTEM = "你是运维值班助理。用户消息里的链接必须先用 fetch_url 抓取，读完材料再回答。"


def agent(model, fetch, task, max_turns=4):
    """手写最小循环：打印调用轨迹——静默/显式的差别全在轨迹里"""
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=task)]
    llm = model.bind_tools([fetch])
    for _ in range(max_turns):
        resp = llm.invoke(messages)
        messages.append(resp)
        if not resp.tool_calls:
            return str(resp.content)
        for call in resp.tool_calls:  # 回执逐条紧跟 assistant（D09 配对铁律）
            print(f"  [call] {call['name']}({call['args']})")
            messages.append(ToolMessage(content=str(fetch.invoke(call["args"])),
                                         tool_call_id=call["id"]))
    return "（轮数用尽未作答）"


if __name__ == "__main__":
    model = ChatOpenAI(api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ReportHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/report"
    task = f"通读今天的状态日报：{url} 有什么需要重点关注的问题吗？"

    print(f"=== 就位：{url}（{len(DOC)} 字符；LIMIT={LIMIT}，切在「数据库事件」节前） ===")

    print("\n=== A. 朴素版：切完不说话，也没有分页参数 ===")
    print(f"  回答：{agent(model, make_naive(), task)[:160]}")
    print("  → 没信号（末行提示）也没把手（offset）：想自救无从下手")

    print("\n=== B. 显式版：offset/limit 把手 + 末行续读信号 ===")
    print(f"  回答：{agent(model, make_honest(), task)[:160]}")
    print("  停一停：A/B 的变量是什么？（先想 3 秒再看下一行）")
    print("  答：披露——A 既无信号也无把手；B 两者都有。同一模型同一页面同一")
    print("    切点，行为从「基于看似完整的半份自信作答」变「自主翻页取证」")
    print("    ——显式截断不是礼貌问题，是信息问题")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（本地 HTTP 服务 + 约 4-6 次真实 LLM 调用）：
#   1. 起本地服务托管长日报；LIMIT 定在「数据库事件」节前的段落边界
#   2. A 组：agent 绑定朴素版 fetch_url（无分页参数、切完不说话）→
#      读到看似完整的前四节（全是「正常」）→ 基于残缺信息作答
#   3. B 组：agent 绑定显式版（offset/limit 把手+末行续读信号）→
#      读第一页看到提示 → 自主传 offset 翻页 → 读到坏盘事件 → 作答
# 验证内容（模型输出动态，验收看轨迹不看具体文案）：
#   - A 组轨迹：仅一次调用即作答（想翻页也无参数可传），答案判定无异常
#     ——第一页里唯一的数据库信号是「主从复制延迟 0.3s 正常」，坏盘事件
#     在切点之后；「不知道自己不知道」的实证
#   - B 组轨迹：出现 offset=LIMIT 的第二次调用——「截断提示被 acted upon」
#     的直接证据（作业①验证点）；答案提到 db-shard-3 坏盘事件
#   - 首跑翻车记：A/B 曾共用同一 schema（都带 offset/limit）——LangChain
#     把参数默认值写进工具 schema，模型看到 offset 存在+默认 509，无信号
#     也精确投机翻页自救成功，对照失效。静默截断的完整形态必须连把手也没有
