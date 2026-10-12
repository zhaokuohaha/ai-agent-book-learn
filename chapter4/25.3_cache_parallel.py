"""chapter4/25.3_cache_parallel.py — 只读红利：可缓存 + 可并行（作业②）

本实验新增的主概念是「只读的两大工程红利」。承接 22.1（感知 = GET
只读）——只读无副作用，天然带来两个自由：
    可缓存：相同查询直接复用上次结果，省时省钱——相当于给 GET 接口
    套层 Redis（课件原版就是 @lru_cache 一行）
    可并行：同时读 5 个文件、并发 3 个搜索互不干扰，无需操心顺序与
    副作用——执行工具（POST）没这种自由，那是明天的主角

实证设计：本地起一个「慢」HTTP 服务（每请求固定 sleep 0.4s + 服务端
命中计数）——缓存是否生效，证据取服务端被打了几次，比「客户端变快」
更硬（快可能是巧合，后端少收一次请求不会骗人）：
    缓存：同一 URL 连抓两次——无缓存版服务端 2 次、第二次照旧 ~400ms；
    lru_cache 版服务端仅 1 次、第二次 ~0ms
    并行：3 个不同 URL——串行 ~1.2s vs ThreadPoolExecutor 并行 ~0.4s
    （I/O 等待释放 GIL，三路往返叠着走，墙钟砍到 1/3）

运行：uv run python -X utf8 chapter4/25.3_cache_parallel.py（纯本地，不烧 API）
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

HITS = 0  # 服务端命中计数：缓存生效的证据在服务端，不在客户端


class SlowHandler(BaseHTTPRequestHandler):
    """慢端点：固定 0.4s 延迟 + 命中计数——模拟真实网络往返的后端"""

    def do_GET(self):
        global HITS
        HITS += 1
        time.sleep(0.4)
        body = f"（{self.path} 的正文，约 3KB）".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # 静默访问日志，别污染实验输出
        pass


SESSION = requests.Session()
SESSION.trust_env = False  # 本地回环不走系统代理（本机常挂代理会拦截 127.0.0.1）


def fetch(url: str) -> str:
    """感知工具原型：真 requests GET，每次都打服务端"""
    return SESSION.get(url, timeout=5).text


@lru_cache(maxsize=128)  # 只读结果可安全缓存——课件原版同款
def fetch_cached(url: str) -> str:
    """套上缓存的同一个工具：相同参数直接复用，不再打服务端"""
    return SESSION.get(url, timeout=5).text


if __name__ == "__main__":
    # ThreadingHTTPServer：每请求一线程——串行服务器会让「并行」演示假摔
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    print("=== 红利一·可缓存：相同查询直接复用（GET 套 Redis 的进程内版） ===")
    for label, fn in [("无缓存", fetch), ("lru_cache", fetch_cached)]:
        before = HITS
        t0 = time.perf_counter()
        fn(f"{base}/a")
        t1 = time.perf_counter()
        fn(f"{base}/a")  # 第二次：缓存版应直接复用
        t2 = time.perf_counter()
        print(f"  {label:<8} 首次 {1000 * (t1 - t0):>4.0f}ms｜第二次 {1000 * (t2 - t1):>6.1f}ms｜服务端被打 {HITS - before} 次")

    print("\n=== 红利二·可并行：同时读互不干扰（I/O 等待释放 GIL） ===")
    urls = [f"{base}/x", f"{base}/y", f"{base}/z"]
    t0 = time.perf_counter()
    for u in urls:  # 串行：一路往返走完再走下一路
        fetch(u)
    serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(fetch, urls))  # 并行：三路往返叠着走
    parallel = time.perf_counter() - t0
    print(f"  串行 {serial:.2f}s → 并行 {parallel:.2f}s（墙钟约 {serial / parallel:.0f} 倍）")
    print("  → 这自由只属于只读：执行工具（明天主角）并发要操心顺序/副作用/")
    print("    补偿；lru_cache 套在写接口上更是灾难——缓存=重放副作用")

# ---- 执行逻辑与验证（看完代码再回头看这里） ----
# 执行逻辑（纯本地，ThreadingHTTPServer 托管 0.4s 慢端点）：
#   1. 缓存对照：无缓存与 lru_cache 版各连抓同一 URL 两次，记录两次
#      耗时与服务端命中数
#   2. 并行对照：3 个不同 URL 串行 vs ThreadPoolExecutor(3) 并行，比墙钟
# 验证内容：
#   - 无缓存：两次均 ~400ms、服务端 2 次；lru_cache：第二次 ~0ms、
#     服务端仅 1 次——缓存生效的铁证是服务端命中计数，不是客户端耗时
#   - 并行墙钟约为串行 1/3（0.4s vs 1.2s 量级，允许调度抖动）
#   - 彩蛋呼应：25.1 的 fetch 每次 offset 续读都重新 GET 全文——套上
#     lru_cache(url, offset) 后同页复抓零成本（缓存键含全部参数）
