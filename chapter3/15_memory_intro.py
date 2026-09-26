"""chapter3/15_memory_intro.py — 用户记忆系统（上）：会话结束，记忆开始

核心概念（原书第 3 章 · W3 Day 15）：

上下文 vs 记忆：一次请求 vs 一个数据库
    上一章解决"这一次对话怎么装得下"（压缩），记忆解决"下次对话还记得我吗"。
    上下文像 request-scoped 变量，请求结束即释放；记忆像持久层，跨请求存活。
    "会话结束 → LLM 提取事实 → 写入记忆"本质是消费会话日志的后台 ETL job：
    不保存每句话，只提炼对未来有用的事实。

提取三规则（决定 ETL 的质量）
    选择性：丢弃"搜索返回 3 个选项"这类短期细节（本次任务的脚手架，不是用户画像）
    抽象化：把本次"要靠窗座位"归纳为长期偏好（单次行为 → 稳定属性）
    结构化：用可检索字段保存（key-value/分类），而非一段流水账

三层次评估框架（什么算"好记忆"）
    第一层 基础回忆：存得准取得回——"会员号 12345"下次精确复述
    第二层 多会话检索：跨时间跨对象找全并推理——两辆车的用户说"给我的车
    预约保养"，该问哪辆而不是瞎猜
    第三层 主动服务：综合久远记忆主动帮忙——订国际航班时发现护照快过期，
    主动预警（"助理"水准）

轨迹 vs 长期记忆：流水账 vs 档案
    轨迹 = 一次运行的完整原始记录，append-only 只增不改（WAL/审计日志，
    用于追溯调试）；长期记忆 = 跨会话提炼的稳定档案，被反复改写、合并、
    淘汰（ETL 后的聚合表）。流水账不可变，档案随新事实演进。
    作业 3 思考题答案：轨迹必须 append-only，因为①审计追溯——改写历史
    = 篡改账本，事故回放失去依据；②KV Cache 前缀稳定——回头改轨迹 =
    从改动点起缓存全灭（D10 同一机制）；③tool_call_id 配对链——改写/
    删除中间消息会破坏调用-回执配对（D09 实测 400）。要"改"只能靠压缩
    （原地替换 tool 内容、条数不变）或把结论写进新消息——前者也只动轨迹
    中后段，且是框架的有意识权衡。

作业落点：
    1. MemoryStore 扩展冲突处理与 delete → update 保留带时间戳的历史（新
       旧都留，见类 docstring 的取舍理由）；delete 软删除可恢复
    2. 三规则手动过滤 → SESSION_1 对话让 LLM 提取，打印后人工点评哪几条
       该进、哪条看似重要其实该丢
    3. 思考题 → 上文"轨迹 vs 长期记忆"段

运行：uv run python -X utf8 chapter3/15_memory_intro.py（需 .env，约 5 次真实调用）
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from common.config import API_KEY, BASE_URL, MODEL

NOW = "2026-09-24"  # 模拟当前日期，写死保证实验可复现

# ---------- 会话 1 的原始轨迹（模拟：含该丢的短期细节 + 一个偏好冲突） ----------

SESSION_1_DIALOGUE = """用户：帮我订下周五飞东京的航班，我偏好靠窗座位，另外我是素食者，需要特殊餐食。
助手：好的，我来搜索下周五飞东京的航班……
[调用 flight_search，返回 3 个选项：NH960 直飞 ¥3200、MU523 中转 ¥2400、JL802 直飞 ¥3500]
助手：根据您靠窗的偏好，我筛出了带靠窗座位的选项。推荐全日空 NH960 直飞，要订吗？
用户：等一下，我最近晕机，坐靠窗反而难受——还是给我过道座位吧。
助手：好的，已为您选择过道座位。请确认素食餐已备注。
用户：确认。里程累计用我的联合航空 MileagePlus 会员号 12345678。
助手：已为您预订 NH960，过道座位，素食餐，里程累计至 MileagePlus 12345678。祝旅途愉快。"""


# ---------- MemoryStore：独立于对话的持久化层（作业 1：冲突处理与 delete） ----------


class MemoryStore:
    """key-value 记忆库，JSON 文件持久化（独立于任何对话进程）。

    冲突处理取舍（作业 1）：update 不直接覆盖——旧值带时间戳进 history。
    保留理由：①第二层评估的"时间推理"需要历史（用户去年住北京今年住上海，
    "你以前是不是住北京？"要能答上）；②审计与误更新回滚（LLM 提取可能提错，
    历史在就能查"这个偏好是什么时候、被哪次会话改的"）。代价：存储渐长、
    注入上下文时须只取当前值——生产可用"history 超过 N 条归档冷存"控制。
    delete 用软删除（记录时间不物理删）：误删可恢复，且"用户曾要求删除某
    记忆"本身也是应被尊重的事实（隐私审计）。均不做物理清理——教学最小实现。
    """

    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def update(self, key, value, category="fact"):
        entry = self.data.get(key, {"history": []})
        if entry.get("value") is not None:
            entry["history"].append({"value": entry["value"], "until": NOW})
        entry.update({"value": value, "category": category, "updated_at": NOW, "deleted_at": None})
        self.data[key] = entry
        self._save()

    def delete(self, key):
        if key in self.data:  # 软删除：可恢复，且"删除"本身留痕
            self.data[key]["deleted_at"] = NOW
            self._save()

    def active(self):
        """注入上下文用的视图：只取未删除条目的当前值（history 不进上下文）"""
        return {k: e["value"] for k, e in self.data.items() if not e.get("deleted_at")}

    def render(self):
        lines = [f"{k} = {v}" for k, v in sorted(self.active().items())]
        return "用户长期记忆（跨会话档案，非本次对话内容）：\n" + "\n".join(lines)

    def _save(self):
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_memories(model, dialogue):
    """会话结束后的 ETL：一次专门调用，按三规则提取结构化事实"""
    response = model.invoke([
        SystemMessage(content=(
            "你是记忆提取器。分析对话，提取值得跨会话长期记住的用户事实，满足三规则："
            "选择性（丢弃搜索结果、航班号、价格等本次任务的短期细节）；"
            "抽象化（把单次行为归纳为长期偏好/属性，注意用户中途的偏好修正以最新为准）；"
            "结构化（key 用 snake_case 语义字段，value 用简洁中文短语——实测不约束能跑出"
            "英文值 aisle，同义不同形会变成检索噪音）"
            '只输出 JSON：{"facts": [{"key": "...", "value": "...", "category": "preference|fact|activity"}]}'
        )),
        HumanMessage(content=dialogue),
    ])
    text = response.content.strip()
    text = text[text.index("{"): text.rindex("}") + 1]  # 剥掉可能的 markdown 代码围栏
    return json.loads(text)["facts"]


def ask_with_memory(model, store, question):
    """会话 2：新轨迹 + 记忆注入 system（D11：系统提示词可含跨会话用户记忆）"""
    response = model.invoke([
        SystemMessage(content=(
            f"你是私人助理。今天是 {NOW}。\n\n{store.render()}\n\n"
            "以上记忆来自过往会话；当前对话尚未开始，请基于记忆回答。需要澄清时主动询问，"
            "不要瞎猜；发现记忆中的潜在问题（如证件过期）要主动预警。"
        )),
        HumanMessage(content=question),
    ])
    return response.content


if __name__ == "__main__":
    model = ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL, timeout=30, max_retries=0,
    )
    workdir = Path(tempfile.mkdtemp(prefix="agblearn_d15_"))
    store = MemoryStore(workdir / "user_memory.json")

    # 预置"更早会话"的记忆：L3 主动服务的原料（数月前存下的护照/两辆车），
    # 以及旧版座位偏好——会话 1 提取的"过道"更新时，旧值"靠窗"应进 history
    store.update("passport_expiry", "2026-10-05", "fact")
    store.update("vehicle_1", "蓝色特斯拉 Model 3（沪A·D8888）", "fact")
    store.update("vehicle_2", "白色丰田凯美瑞（沪B·F6666）", "fact")
    store.update("seat_preference", "靠窗座位", "preference")

    print("=== 会话 1：订票对话（含偏好冲突：靠窗 → 晕机改过道） ===")
    print(f"对话 {len(SESSION_1_DIALOGUE)} 字（含该丢的短期细节：3 个航班选项/价格/航班号）")

    print("\n=== 会话结束：ETL 提取（三规则） ===")
    facts = extract_memories(model, SESSION_1_DIALOGUE)
    for f in facts:
        store.update(f["key"], f["value"], f["category"])
        print(f"  [入库] {f['key']} = {f['value']}（{f['category']}）")

    print("\n[作业 2 点评] 该进的：靠窗→过道的偏好修正（抽象化，且以最新为准）、素食、"
          "MileagePlus 号（第一层基础回忆的原料）；该丢的：3 个航班选项、NH960、"
          "价格——本次任务的脚手架，对下次会话毫无价值；'东京行程'可存为 activity"
          "（第三层主动服务要用它关联护照）。")
    print("\n[作业 1 验证] 偏好冲突的历史保留：")
    pref = store.data.get("seat_preference", {})
    for h in pref.get("history", []):
        print(f"  历史：{h['value']}（截至 {h['until']}）")
    print(f"  当前：{pref.get('value')}（{pref.get('updated_at')} 更新）——新旧都留带时间戳，"
          "时间推理可答'以前是不是靠窗'")
    store.update("marketing_opt_in", "yes")  # 软删除演示条目（与业务记忆无关）
    store.delete("marketing_opt_in")
    print(f"  [软删除] marketing_opt_in deleted_at={store.data['marketing_opt_in']['deleted_at']}，"
          f"active() 已不含它——可恢复、留审计痕，不物理清除")

    print("\n=== 会话 2（新会话，只带记忆不带旧轨迹）：三层次验证 ===")
    print("[L1 基础回忆] 会员号")
    print(f"  答：{ask_with_memory(model, store, '我的联合航空会员号是多少？')[:80]}")
    print("\n[L2 多会话检索] 两辆车该问哪辆，而不是瞎猜")
    print(f"  答：{ask_with_memory(model, store, '帮我给车预约下周保养。')[:150]}")
    print("\n[L3 主动服务] 订航班应关联护照记忆主动预警")
    answer = ask_with_memory(model, store, "帮我订 10 月 20 日飞东京的航班。")
    print(f"  答：{answer[:200]}")
    warned = "护照" in answer and ("过期" in answer or "有效" in answer)
    print(f"  [L3 验收] 主动预警护照问题：{'是——达到助理水准' if warned else '否——记忆在但未主动关联（可再跑或加强提示词）'}")

    print(f"\n[持久化] 记忆文件：{store.path}（独立于对话进程，下次运行仍可读取）")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑：
#   1. MemoryStore 落在临时目录的 JSON 文件上（"独立于对话的持久化层"的物证），
#      预置三条"更早会话"的记忆：护照有效期、两辆车（L2/L3 的原料）
#   2. 会话 1 是一段写死的订票对话（含偏好冲突与该丢的短期细节）→
#      extract_memories 一次专门 LLM 调用按三规则提取 → 逐条 store.update 入库
#   3. 冲突验证：seat_preference 的 history 里应留着旧值（靠窗），当前值是过道
#   4. 会话 2：ask_with_memory 每问新建轨迹，记忆渲染进 system——
#      L1 问会员号 / L2 问"给车预约保养" / L3 问"订 10 月 20 日飞东京航班"
# 验证内容：
#   - 提取结果不含航班号/价格/选项数（选择性），偏好取最新过道（抽象化+冲突），
#     key 是 snake_case 字段（结构化）
#   - 偏好 history：旧值"靠窗"带时间戳保留（作业 1 取舍的落地）
#   - L1 精确复述 12345678；L2 反问哪辆车（不是瞎猜）；L3 回答中提到护照
#     （10-05 过期 < 10-20 出行，预警与否看模型主动性——动态输出，验收看关键词）
