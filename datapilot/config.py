# datapilot/config.py

import os
from pathlib import Path

from dotenv import load_dotenv


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


HF_TOKEN = os.getenv("HF_TOKEN")


if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is not configured. "
        "Add HF_TOKEN=your_token to the project root .env file."
    )