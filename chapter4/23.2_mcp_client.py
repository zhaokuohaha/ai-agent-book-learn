"""chapter4/23.2_mcp_client.py — MCP 客户端：手拼 JSON-RPC，看清协议交互

本实验新增的主概念是「客户端-服务器架构」：MCP 客户端（Cursor / Claude
Desktop / Agent 框架）通过标准协议与 server 通信。为什么不直接用官方
SDK 的 ClientSession？——SDK 把协议细节全遮住了；本实验手拼原始 JSON-RPC
报文（作业②的脚本版：Inspector 是官方浏览器 UI，适合人肉点调；脚本捕获
原始报文可复现、看得见每个字节），「标准」长什么样一目了然。

协议三要点（本文件的代码落点）：
    JSON-RPC 2.0：每条消息 {"jsonrpc":"2.0","id":N,"method":"...","params":{...}}，
        响应带同 id 回 {"result":...}；通知（notifications/*）无 id 也无响应
    stdio 传输：换行分隔——一行一个 JSON，请求写进 server 的 stdin，
        响应从 stdout 读出（远程场景换 Streamable HTTP，报文格式不变）
    initialize 握手：连接的第一件事——客户端报协议版本与能力，server 回
        它支持的版本与能力（tools/resources/prompts 各自独立声明），
        之后才是 initialized 通知 → 正式会话

与前面实验的关系：D12 的 dispatch 是进程内路由（name 查字典）；今天把
「执行」搬进另一个进程，靠协议对话——工具生态的形状从「函数」升级为
「服务」。

运行：uv run python -X utf8 chapter4/23.2_mcp_client.py（纯本地跨进程，不烧 API）
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SERVER = Path(__file__).parent / "23.1_mcp_server.py"


def rpc(proc, payload):
    """一行请求进（server 的 stdin），一行响应出（stdout）——stdio 的全部"""
    proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("server 提前退出——先单独跑 23.1 看报错")
        if line.strip().startswith("{"):  # 跳过 server 的非 JSON 输出
            return json.loads(line)


def brief(obj, limit=170):
    return json.dumps(obj, ensure_ascii=False)[:limit] + ("…" if len(json.dumps(obj, ensure_ascii=False)) > limit else "")


def request(proc, req_id, method, params=None, label=""):
    """请求-响应包装：打印 method 与响应要点，返回 result"""
    payload = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params:
        payload["params"] = params
    result = rpc(proc, payload)["result"]
    print(f"  → {method}{'（' + label + '）' if label else ''}")
    return result


if __name__ == "__main__":
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # 子进程输出也走 UTF-8
    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8", str(SERVER)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", env=env,
    )
    try:
        print("=== 1. initialize 握手：协议版本 + 能力协商（无状态协议的会话凭证） ===")
        result = request(proc, 1, "initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "toy-client", "version": "0.1"},
        })
        print(f"  ← server：{result['serverInfo']['name']}｜协议版本 {result['protocolVersion']}")
        print(f"  ← 能力声明：{brief(result['capabilities'])}")
        proc.stdin.write('{"jsonrpc": "2.0", "method": "notifications/initialized"}\n')
        proc.stdin.flush()  # 通知：无 id 无响应——握手完成，进入正式会话

        print("\n=== 2. tools/list → tools/call：模型可执行的操作 ===")
        tool = request(proc, 2, "tools/list")["tools"][0]
        print(f"  ← {tool['name']}｜schema：{brief(tool)}")
        print("  （description 就是 D22 ACI 三件套；inputSchema 是 JSON Schema——协议强制）")
        call = request(proc, 3, "tools/call", {"name": "get_weather", "arguments": {"city": "杭州"}})
        print(f"  ← {call['content'][0]['text']}")
        call = request(proc, 4, "tools/call", {"name": "get_weather", "arguments": {"city": "成都"}})
        print(f"  ← {call['content'][0]['text']}（边界生效：错误也是观察，D12 回响）")

        print("\n=== 3. resources/read：应用可读取的只读数据 ===")
        content = request(proc, 5, "resources/read", {"uri": "weather://cities"})["contents"][0]
        print(f"  ← {content['text']}（决策权在应用：给 UI 或按需进上下文，模型不直接调）")

        print("\n=== 4. prompts/get：用户可选用的模板 ===")
        msg = request(proc, 6, "prompts/get",
                      {"name": "weather_report", "arguments": {"city": "杭州"}})["messages"][0]
        print(f"  ← {msg['content']['text']}（决策权在用户：模板渲染成消息，等 LLM 执行）")

        print("\n停一停：为什么必须先 initialize 握手，而不是上来就 tools/call？")
        print("  模型 API 无状态（D09），MCP 会话同样从零开始——握手协商的不只是版本：")
        print("  server 声明支持哪些原语（tools/resources/prompts 各自独立），客户端据此")
        print("  决定能调什么、怎么渲染——相当于 HTTP 的 Content-Type 协商先于正文。")
    finally:
        proc.terminate()  # 客户端断开即子进程收工（stdio 连接的生命周期）

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（纯本地跨进程，不烧 API）：
#   1. Popen 拉起 23.1_mcp_server.py 子进程（stdin/stdout 管道，UTF-8）
#   2. initialize 握手 → initialized 通知 → tools/list → tools/call×2
#      （杭州 + 不支持的成都）→ resources/read → prompts/get → terminate
# 验证内容：
#   - 握手：serverInfo.name=weather，capabilities 里能看出 tools/resources/
#     prompts 三类原语各自声明——「原语分离」在协议上的实证
#   - tools/list：一个工具，description=ACI 三件套，inputSchema=JSON Schema
#   - tools/call：杭州返回「晴 26°C」；成都返回错误文本（边界生效）
#   - resources/read 读到城市列表；prompts/get 拿到参数化模板——三类原语
#     「谁使用谁」不同，但走同一协议同一连接
#   - 检查点：无状态协议靠握手建立「会话凭证」（版本 + 能力），先协商后干活
