# -*- coding: utf-8 -*-
import os
from pathlib import Path
from dotenv import load_dotenv

# 自动寻找项目根目录下的 .env 文件
# 当前文件在 utils/env_utils.py，所以根目录是父目录的父目录
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
env_path = project_root / ".env" # <-- 提前定义 env_path

if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    # 如果没找到指定路径，尝试默认加载 (兼容性)
    load_dotenv(override=True)

API_KEY = os.environ.get("BAILIAN_API_KEY")
BASE_URL = os.environ.get("BAILIAN_BASE_URL")

if not API_KEY:
    raise ValueError(f"❌ 错误: 未找到 API Key。请确保 {env_path} 文件存在且包含 BAILIAN_API_KEY 配置。")