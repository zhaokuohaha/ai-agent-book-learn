"""chapter7/43.1_eval_task_anatomy.py — 解剖一条 τ²-bench 评估任务：四要素 + 双控现场

本实验新增的主概念是 **评估任务的四要素**：一条评估任务 = ticket（模糊工单）
+ user_scenario（用户模拟器的认知边界与验收口径）+ initial_state（重置脚本）
+ evaluation_criteria（评分标准）——后端类比就是一次可重复的集成测试
fixture：setUp 重置环境、输入用例、断言，一个不少。

三个设计动机在字段里各有落点：
① 渐进式信息透露——known_info 只有姓名/号码/所在国，根因（飞行模式开着、
   漫游关了）**不在里面**：用户"不知道自己不知道"，只能被 Agent 问出来。
   常见误区：靠提示词"别一次说完"约束模拟器——正解是知识范围建模成独立
   字段，从数据层面保证（下面有提示词版 vs 字段版的现场对照）；
② 事实锚定（Grounding）——设备状态回答必须来自工具结果，否则模拟用户会
   顺着 Agent 话头确认"修好了"，评估退化成两个模型互相点头；
③ 双控（Dual-Control）——故障劈两半：运营商侧配置 Agent 能查能改，设备侧
   开关只有用户能操作（客服改不了你手机设置）。本实验用脚本化 Agent 演示
   "查遍运营商侧一切正常"的困境；真 LLM 双侧对话与四维验证器在 43.2。

运行：uv run python -X utf8 chapter7/43.1_eval_task_anatomy.py（无需 .env）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.tools import tool

# 要素① ticket：给 Agent 的工单——只有"我上不了网"级别的模糊描述
TICKET = "用户报告手机无法上网，正在法国出差。请诊断并解决。"

# 要素② user_scenario：用户模拟器是谁、知道什么、怎么验收、靠什么说话
USER_SCENARIO = {
    "known_info": ["姓名: 张三", "号码: +33612345678", "所在国家: 法国"],  # 根因不在其中——渐进式透露
    "grounding": "设备状态回答必须来自 check_status_bar 的工具结果，绝不编造",
    "acceptance": "run_speed_test 返回 excellent 才算解决",
}

# 要素④ evaluation_criteria：四维校验 + 口径（reward_basis 决定一切）
CRITERIA = {
    "env_assertions": ["airplane_mode == False", "roaming == True", "last_speed == 'excellent'"],
    "actions": ["check_status_bar"],           # 关键动作必须发生过
    "communicate_info": ["根因已告知用户"],     # 信息告知
    "nl_assertions": ["Agent 未编造设备状态"],  # 自然语言断言
    "reward_basis": "ENV_ASSERTION",           # 二元奖励：0 或 1，跨模型可比
}


def make_tools(env: dict):
    """双控的两套工具集：运营商侧给 Agent，设备侧只给用户——权限边界在接口层"""

    @tool
    def get_line_status(phone: str) -> str:
        """（运营商侧）查询号码的线路状态与配置"""
        d = env["carrier"][phone]
        return f"线路正常·已开通国际漫游·流量套餐有效·欠费=无（所在国: {d['country']}）"

    @tool
    def check_status_bar() -> str:
        """（设备侧）查看手机状态栏：飞行模式/漫游开关/信号"""
        d = env["device"]
        return f"飞行模式={'开' if d['airplane'] else '关'} 漫游={'开' if d['roaming'] else '关'} 信号={d['signal']}"

    @tool
    def toggle_airplane() -> str:
        """（设备侧）开/关飞行模式"""
        env["device"]["airplane"] = not env["device"]["airplane"]
        return f"飞行模式已切换为 {'开' if env['device']['airplane'] else '关'}"

    @tool
    def run_speed_test() -> str:
        """（设备侧）测速——验收口径的唯一裁判"""
        d = env["device"]
        d["last_speed"] = "excellent" if (not d["airplane"] and d["roaming"]) else "poor"
        return f"测速结果: {d['last_speed']}"

    return get_line_status, check_status_bar, toggle_airplane, run_speed_test


def reset(env: dict):
    """要素③ initial_state：重置脚本——把两侧状态拉回同一起点（可重复的前提）"""
    env["device"] = {"airplane": True, "roaming": False, "signal": "无", "last_speed": None}  # 故障注入
    env["carrier"] = {"+33612345678": {"country": "法国", "line": "正常", "roaming_plan": True}}


if __name__ == "__main__":
    env: dict = {}
    reset(env)
    line_status, status_bar, toggle, speed = make_tools(env)

    print("═" * 62)
    print("一条评估任务的四个组成部分（τ²-bench telecom 简化版）")
    print("═" * 62)
    for name, val in [("① ticket", TICKET), ("② user_scenario", USER_SCENARIO),
                      ("③ initial_state", "reset(env)——飞行模式=开、漫游=关（故障注入）"),
                      ("④ evaluation_criteria", CRITERIA)]:
        print(f"\n【{name}】\n  {val}")

    print("\n" + "═" * 62 + "\n双控现场：脚本化 Agent 只有运营商侧工具\n" + "═" * 62)
    print(f"Agent → get_line_status: {line_status.invoke({'phone': '+33612345678'})}")
    print("Agent 的数据库视角：一切正常——但用户确实上不了网。为什么查不出来？")
    print("  ↓ 因为故障的另一半在设备侧，Agent 的工具集里根本没有那把钥匙：")
    print("    agent_tools = ['get_line_status']（查不到 airplane/roaming 字段）")
    print(f"User  → check_status_bar: {status_bar.invoke({})}   ← 真相在用户这一侧")
    print("User  → toggle_airplane（引导后关飞行模式）:", toggle.invoke({}))
    env["device"]["roaming"] = True  # 引导用户开漫游（设备侧动作只能由用户执行）
    print("User  → 开漫游后测速:", speed.invoke({}))

    print("\n" + "═" * 62 + "\n渐进式透露：提示词版 vs 字段版\n" + "═" * 62)
    print("提示词版：system prompt 写「请不要一次说出全部信息」")
    print("  → 软约束：模型可能顺嘴漏底（37.1 刚证过关键词/指令约束不可靠），也可能过度隐瞒连姓名都不给")
    print(f"字段版：known_info = {USER_SCENARIO['known_info']}")
    print("  → 硬边界：根因物理上不在模拟用户的知识里，想漏也漏不出来——从数据层面保证")

    print("\n【停一停】reset 之后 env 里飞行模式是开的——user_scenario 的 known_info 里为什么不能有它？")
    print("  → 环境状态 ≠ 用户认知：用户「不知道自己不知道」。若写进 known_info，模拟器开场就会说")
    print("    「我飞行模式开着」——需求澄清这个核心能力就测不到了（ticket 保持模糊的意义所在）。")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（纯确定性，无 LLM）：
#   1. 打印四要素——对照课件骨架逐项认字段（②③④各自服务哪个设计动机）
#   2. reset(env) 注入故障 → 脚本化 Agent 调运营商侧工具（一切正常）→
#      用户侧工具揭示真相 → 修好 → 测速 excellent（双控全流程）
#   3. 提示词版 vs 字段版对照 + 一个「停一停」检查点（环境状态≠用户认知）
# 验证内容：
#   - get_line_status 返回全「正常」而 check_status_bar 返回「飞行模式=开
#     漫游=关」——同一世界两个视角，Agent 侧查不出根因（双控成立）
#   - toggle_airplane + 开漫游后测速 excellent——设备侧动作只能由用户执行
#   - known_info 三项里没有故障原因——渐进式透露靠字段不靠提示词
#   - 简化注：真实 τ²-bench 双侧都是 LLM 驱动、用户模拟器带独立工具集，
#     完整对话循环与四维验证器见 43.2；此处脚本化只为解剖结构
