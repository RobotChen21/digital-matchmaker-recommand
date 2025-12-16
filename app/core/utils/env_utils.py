# -*- coding: utf-8 -*-
import os
from pathlib import Path
from dotenv import load_dotenv

# 自动寻找项目根目录下的 .env 文件
# 当前文件在 utils/env_utils.py，所以根目录是父目录的父目录
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent # app/core/env_utils.py -> app/core -> app -> root
env_path = project_root / ".env" 

if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv(override=True)

# LLM API Keys and Base URLs
API_KEY = os.environ.get("BAILIAN_API_KEY")
BASE_URL = os.environ.get("BAILIAN_BASE_URL")

# JWT Security Settings
SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256") # 默认 HS256
# 检查关键环境变量
if not API_KEY:
    raise ValueError(f"❌ 错误: 未找到 LLM API Key。请确保 {env_path} 文件或环境变量包含 BAILIAN_API_KEY 配置。")
if not SECRET_KEY:
    raise ValueError(f"❌ 错误: 未找到 JWT Secret Key。请确保 {env_path} 文件或环境变量包含 JWT_SECRET_KEY 配置。")
