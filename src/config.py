import os
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.tracers import LangChainTracer

load_dotenv()

PROJECT_NAME = "financial-analyst-agent"

_langchain_key = os.environ.get("LANGCHAIN_API_KEY", "")
_groq_key = os.environ.get("GROQ_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if not _langchain_key:
    raise EnvironmentError("LANGCHAIN_API_KEY not set — add it to .env")
if not _groq_key:
    raise EnvironmentError("GROQ_API_KEY not set — add it to .env")

GROQ_API_KEY = _groq_key

ls_client = Client(
    api_key=_langchain_key,
    api_url="https://api.smith.langchain.com",
)

tracer = LangChainTracer(
    project_name=PROJECT_NAME,
    client=ls_client,
)
