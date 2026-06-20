from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from src.config import DB_PATH
from src.database.models import Base

DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine, expire_on_commit=False
)


async def init_db() -> None:
    """สร้าง tables ทั้งหมด (idempotent — รันซ้ำได้)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
