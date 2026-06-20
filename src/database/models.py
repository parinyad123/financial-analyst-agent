import uuid
from datetime import datetime

from sqlalchemy import String, Float, ForeignKey, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Portfolio(Base):
    __tablename__ = "portfolios"

    portfolio_id: Mapped[str] = mapped_column(String, primary_key=True,
                                               default=lambda: str(uuid.uuid4()))
    name:         Mapped[str] = mapped_column(Text, nullable=False)
    created_at:   Mapped[str] = mapped_column(Text,
                                               default=lambda: datetime.utcnow().isoformat())
    positions: Mapped[list["Position"]] = relationship(
        "Position", back_populates="portfolio", cascade="all, delete-orphan"
    )


class Position(Base):
    __tablename__ = "positions"

    position_id:  Mapped[str]   = mapped_column(String, primary_key=True,
                                                  default=lambda: str(uuid.uuid4()))
    portfolio_id: Mapped[str]   = mapped_column(String,
                                                  ForeignKey("portfolios.portfolio_id",
                                                             ondelete="CASCADE"))
    ticker:       Mapped[str]   = mapped_column(String, nullable=False)
    shares:       Mapped[float] = mapped_column(Float, nullable=False)
    avg_cost:     Mapped[float] = mapped_column(Float, nullable=False)
    created_at:   Mapped[str]   = mapped_column(Text,
                                                  default=lambda: datetime.utcnow().isoformat())
    updated_at:   Mapped[str]   = mapped_column(Text,
                                                  default=lambda: datetime.utcnow().isoformat())
    portfolio: Mapped["Portfolio"] = relationship("Portfolio", back_populates="positions")
