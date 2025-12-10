import os

from dotenv import load_dotenv

load_dotenv(override=True)

API_KEY = os.environ.get("BAILIAN_API_KEY")
BASE_URL = os.environ.get("BAILIAN_BASE_URL")