import os
from dotenv import load_dotenv

# load_dotenv() ต้องอยู่ก่อน langsmith import ทุกตัว —
# lru_cache ของ langsmith.utils.get_env_var cache ค่า env ตอน import ครั้งแรก
# ถ้า import langsmith ก่อน load_dotenv() ค่าจาก .env จะไม่ถูกเห็น
load_dotenv()
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

from langsmith import Client
from langchain_core.tracers import LangChainTracer

PROJECT_NAME = "financial-analyst-agent"

_langchain_key = os.environ.get("LANGCHAIN_API_KEY", "")
_groq_key = os.environ.get("GROQ_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if not _langchain_key:
    raise EnvironmentError("LANGCHAIN_API_KEY not set — add it to .env")
if not _groq_key:
    raise EnvironmentError("GROQ_API_KEY not set — add it to .env")

GROQ_API_KEY = _groq_key

DB_PATH = os.environ.get("DB_PATH", "portfolio.db")

# LangGraph checkpointer (conversation memory) — kept in a SEPARATE sqlite file so the
# checkpoint schema never mixes with the SQLAlchemy portfolio models in DB_PATH.
CHECKPOINT_DB_PATH = os.environ.get("CHECKPOINT_DB_PATH", "checkpoints.db")

ls_client = Client(
    api_key=_langchain_key,
    api_url="https://api.smith.langchain.com",
)

tracer = LangChainTracer(
    project_name=PROJECT_NAME,
    client=ls_client,
)
