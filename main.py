import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

# Windows terminal ใช้ cp1252/cp874 โดย default — Thai chars จะออก ? ถ้าไม่ reconfigure
# reconfigure ต้องทำก่อน uvicorn เริ่ม log ใดๆ
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.api.routes import router
from src.config import ls_client
from src.database.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    ls_client.flush()  # flush pending LangSmith traces ก่อน shutdown


app = FastAPI(
    title="Financial Analyst Agent",
    description="Explainable Quant Analytics + Agent Orchestration",
    version="1.5.0",
    lifespan=lifespan,
)
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
