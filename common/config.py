"""common/config.py — 全项目共用的 API 配置（所有章节实验通用）

自动读取项目根目录的 .env 文件（已被 git 忽略），也可用同名环境变量覆盖。
适用于任何 OpenAI 兼容接口（DeepSeek、通义、Kimi、OpenAI 等）。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.environ.get("LLM_API_KEY", "")    # token，DeepSeek 在 https://platform.deepseek.com/api_keys 创建
BASE_URL = os.environ.get("LLM_BASE_URL", "")  # 接口地址，DeepSeek 为 https://api.deepseek.com
MODEL = os.environ.get("LLM_MODEL", "")        # 模型名，DeepSeek 为 deepseek-chat 或 deepseek-reasoner

if not (API_KEY and BASE_URL and MODEL):
    raise SystemExit("请先在项目根目录 .env 中填写 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL")
