import os
from dotenv import load_dotenv
import logging

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 標準出力へのハンドラ
    ],
)
logger = logging.getLogger("env_loader")


def load_env_variables():
    """環境変数を.envファイルから読み込む"""
    load_dotenv()

    # 必要な環境変数を確認
    required_vars = [
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_CSE_ID",
        "SERPAPI_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_TRACING_V2",
        "OPENAI_MODEL",
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.warning(
            f"警告: 以下の環境変数が設定されていません: {', '.join(missing_vars)}"
        )
        logger.warning("一部の機能が制限される可能性があります。")

    return {var: os.getenv(var) for var in required_vars}
