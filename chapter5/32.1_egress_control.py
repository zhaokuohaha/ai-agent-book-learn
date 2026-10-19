"""chapter5/32.1_egress_control.py — 网络出口控制：掐断攻击链的收益端

本实验新增的主概念是「致命三要素 + 网络出口控制」（沙箱第一、二道防线）。

致命三要素（Simon Willison）：①访问私有数据 ②暴露于不受信内容 ③具备外部
通信能力——三者单独都正常，同时具备攻击链就闭合：恶意指令藏在网页/邮件/
工具输出里 → 驱使 Agent 读私密文件 → 经对外通道传出去。原书补第四维：
持久记忆是放大器（恶意指令写进 MEMORY.md 跨会话潜伏）。后端类比：同时有
「数据库读权限 + 用户任意输入 + 公网出网」却没有 WAF 和出口防火墙。

网络出口控制为什么是第一道防线：回看第③条——即使注入成功、恶意代码在
沙盒里读到了敏感数据，没有出口就传不出去。数据泄露必须有一条外传通道，
掐断通道 = 掐断攻击收益（与数据库最小权限同构：不是防住所有注入，而是
让注入拿不到东西）。

本实验演练（零 API，纯确定性）：
    ① 三要素盘点——我们的玩具 Agent 三要素各占几条（作业③落点）
    ② egress 白名单攻击演练——域名仿冒变体 vs urlparse 正确解析；
       对照「字符串 contains」这个最常见的错误实现怎么被骗
    ③ 凭证隔离——凭证类文件根本不进沙盒（生产：容器不挂载 ~/.ssh）；
       教学版用路径锁近似：read_file 的 ../ 逃逸一律拒绝

运行：uv run python -X utf8 chapter5/32.1_egress_control.py（无需 .env）
"""

import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.coding_tools import make_tools

WS = Path(__file__).parent / ".agent_ws5"
SECRETS = Path(__file__).parent / ".secrets"  # 工作区外的「凭证目录」（假钥，不进 git）

ALLOWED_HOSTS = {"pypi.org", "files.pythonhosted.org", "hf-mirror.com"}


def egress_allowed(url: str) -> str:
    """白名单代理判定：urlparse 解析出真 hostname（自动小写化）再全等比对"""
    host = urlparse(url).hostname or ""
    return f"放行（{host} 在白名单）" if host in ALLOWED_HOSTS else f"拒绝（{host} 不在白名单）"


def naive_allowed(url: str) -> bool:
    """错误示范：字符串 contains——白名单实现最常见的坑，被仿冒域名骗"""
    return "pypi.org" in url


if __name__ == "__main__":
    print("=== 作业③：我们的玩具 Agent 三要素盘点 ===")
    print("  ① 访问私有数据：是——read_file 可读工作区（加凭证隔离前 ../ 逃逸也在尝试范围内）")
    print("  ② 暴露于不受信内容：弱——无外部抓取工具，但任务文本/工具输出本身不可信（D12）")
    print("  ③ 外部通信能力：是——py_exec 可 requests 任意 URL（出口控制前无防线）")
    print("  → 三要素基本齐备；下一步砍第③条：执行环境默认断网 + 白名单代理")

    print("\n=== 演练 1：egress 白名单 vs 域名仿冒 ===")
    for url in ["https://pypi.org/simple/", "https://evil.com/payload"]:
        print(f"  {url:<42} {egress_allowed(url)}")
    for url in ["https://pypi.org.evil.com/simple", "https://pypi.org@evil.com/simple"]:
        fooled = "字符串 contains 版被骗放行" if naive_allowed(url) else "contains 版也拒了"
        print(f"  {url:<42} {egress_allowed(url)}｜{fooled}")
    for url in ["https://Pypi.Org/simple", "https://192.168.1.5:8080/x"]:
        print(f"  {url:<42} {egress_allowed(url)}")
    print("  → 子域名仿冒（pypi.org.evil.com）与 userinfo 仿冒（pypi.org@evil.com）")
    print("    是 contains 实现的经典翻车点：hostname 必须 urlparse 解析后全等比对")

    print("\n=== 演练 2：凭证隔离（读不到的文件最安全） ===")
    shutil.rmtree(WS, ignore_errors=True)
    WS.mkdir(parents=True)
    SECRETS.mkdir(exist_ok=True)
    (SECRETS / "id_rsa").write_text("FAKE-PRIVATE-KEY（教学演示用假钥）", encoding="utf-8")
    by = {t.name: t for t in make_tools(WS)}
    for path in ["../.secrets/id_rsa", ".secrets/id_rsa"]:
        r = by["read_file"].invoke({"path": path})
        print(f"  read_file({path!r}) → {str(r).splitlines()[0]}")
    print("  → 路径锁把逃逸一律拒绝；生产更强：凭证根本不挂载进沙盒（容器不")
    print("    volume ~/.ssh），读不到的文件不需要防御——挂载层即信任边界")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（零 API 调用）：
#   1. 三要素盘点（纯打印——作业③思考题的落点答案）
#   2. egress 演练：6 个 URL 变体过白名单；其中仿冒变体同时过 contains 错误版
#   3. 凭证演练：工作区外预置假 id_rsa → read_file 两条逃逸路径都被路径锁拒绝
# 验证内容：
#   - pypi.org 放行、evil.com 拒绝；Pypi.Org 放行（urlparse hostname 自动小写）
#   - pypi.org.evil.com / pypi.org@evil.com 拒绝——且 contains 版均被骗放行
#     （@ 前是 userinfo，真 host 在 @ 后——urlparse 剥离后才可比对）
#   - 192.168.1.5:8080 拒绝（IP 直连同样不在白名单）
#   - read_file 的两条 ../ 逃逸均返回 [错误] 路径必须在工作区内
#   - 注：egress 判定是纯函数演示；接进 py_exec 需在代理层拦 requests/urllib
#     的出口（教学版无真实出网需求，注明差距）
