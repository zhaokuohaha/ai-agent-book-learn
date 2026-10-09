"""chapter4/23.1_mcp_server.py — MCP 服务器：工具生态的 JDBC（服务端侧）

本实验新增的主概念是「MCP（Model Context Protocol）」：统一 AI 模型与外部
工具/数据源通信的开放标准。承接 D22——昨天讲「怎么设计好一个工具」，
今天讲「工具怎么跨框架分发」。

痛点（为什么需要协议）：每家 Agent 框架定义工具的方式都不一样——OpenAI
function calling、Anthropic tool use、LangChain Tool 抽象——工具开发者要
为每个框架重复适配。这正是当年数据库驱动百花齐放的老问题，JDBC 用一套
统一接口终结了它；MCP 就是工具生态的 JDBC。

生态价值：一次开发处处可用——本文件这一个 server，Cursor / Claude
Desktop / 任何 MCP 客户端都能直接用，不用为本仓库改一行代码。

三类原语分离（「谁使用」不同，D09 信任分层的回响）：
    tools     模型可执行的操作——决策权在模型（LLM 看 description 选）
    resources 应用可读取的只读数据——决策权在应用（进 UI / 按需进上下文）
    prompts   用户可选用的模板——决策权在用户（显式挑选，参数化填充）

本文件也是作业③的示范：想给玩具 Agent 加的第一个 MCP 工具——天气查询，
description 按 D22 ACI 三件套写（何时用 + 边界 + 参数示例），描述是写给
模型看的使用说明书。

课件差异注：课件用 v1 写法 from mcp.server.fastmcp import FastMCP——
mcp 2.x 已更名 MCPServer（from mcp.server.mcpserver import MCPServer），
装饰器用法不变。

运行说明：直接运行 = server 以 stdio 模式启动并等待客户端连接（阻塞是
正常行为——它就是个服务端）；常规验证由 23.2 作为客户端拉起本文件，
交互式调试可用官方 Inspector（mcp dev chapter4/23.1_mcp_server.py）。
"""

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("weather")

CITIES = ("北京", "上海", "杭州", "深圳")


@mcp.tool()
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。何时用：用户问某城市现在的天气时。
    边界：只支持北京/上海/杭州/深圳四城，其他城市返回错误；只报实时，
    不含预报。参数示例：city='杭州'"""
    if city not in CITIES:
        return f"错误：不支持的城市「{city}」（支持：{'、'.join(CITIES)}）"
    return f"{city}：晴，26°C"


@mcp.resource("weather://cities")
def supported_cities() -> str:
    """支持城市列表（resource：应用读取的只读数据，模型不直接调用）"""
    return "支持城市：" + "、".join(CITIES)


@mcp.prompt("weather_report")
def weather_report(city: str) -> str:
    """天气报告模板（prompt：用户选用的模板，参数化填充）"""
    return f"请为 {city} 写一份简短天气播报：先调用 get_weather，再用一句话轻松点评。"


if __name__ == "__main__":
    mcp.run()  # 默认 stdio 传输：本地进程拉起；远程场景换 Streamable HTTP

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑：
#   1. 本文件是标准 MCP server：MCPServer("weather") 实例 + 三个原语各一个
#      最小示例（@mcp.tool / @mcp.resource / @mcp.prompt）
#   2. 直接 uv run 会以 stdio 模式阻塞等待客户端——服务端的本分
# 验证方式：
#   - 常规：uv run python -X utf8 chapter4/23.2_mcp_client.py（客户端拉起
#     本文件，跨进程走完整 JSON-RPC：握手 → tools → resources → prompts）
#   - 交互：uv run mcp dev chapter4/23.1_mcp_server.py 开官方 Inspector 在
#     浏览器里点调（作业②）
#   - 验收点：tools/call get_weather('杭州') 返回 26°C；不支持城市返回
#     错误文本（D12「错误也是观察」在协议层的形状）
