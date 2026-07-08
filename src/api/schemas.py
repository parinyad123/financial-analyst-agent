import re

from pydantic import BaseModel, field_validator

_PORTFOLIO_ID_RE = re.compile(r"^[a-z0-9-]+$")
_GENERIC_PORTFOLIO_IDS = frozenset(["demo", "test", "port", "portfolio"])


def validate_new_portfolio_id(portfolio_id: str) -> None:
    """Naming rule for NEWLY created portfolios only (case 12 fix: a real,
    specific id is what makes _plan_override in src/agent/core.py reliable —
    ids that are too short/generic risk accidental substring matches or
    collide with common words in free-text queries). Existing ids already in
    the DB (e.g. "demo", "test_api" test fixtures) are grandfathered — this is
    called only from the create branch in routes.save_positions, never as an
    automatic Pydantic field_validator, so it can't be applied retroactively
    to portfolios that already exist. See docs/DECISIONS.md#routing."""
    if len(portfolio_id) < 5:
        raise ValueError("portfolio_id ต้องมีความยาวอย่างน้อย 5 ตัวอักษร")
    if not _PORTFOLIO_ID_RE.match(portfolio_id):
        raise ValueError("portfolio_id ต้องเป็นตัวพิมพ์เล็ก ตัวเลข และ '-' เท่านั้น")
    if portfolio_id in _GENERIC_PORTFOLIO_IDS:
        raise ValueError(
            f"portfolio_id '{portfolio_id}' เป็นคำทั่วไปเกินไป — ใช้ชื่อที่เจาะจงกว่านี้ "
            "(เช่น 'my-tech-2025')"
        )


class PositionIn(BaseModel):
    ticker: str
    shares: float
    avg_cost: float

    @field_validator("ticker")
    @classmethod
    def upper_ticker(cls, v: str) -> str:
        return v.upper().strip()


class StockAnalysisRequest(BaseModel):
    query: str
    ticker: str | None = None


class PortfolioAnalysisRequest(BaseModel):
    portfolio: dict[str, float]
    query: str = ""


class SavePositionsRequest(BaseModel):
    portfolio_id: str
    name: str
    positions: list[PositionIn]


class HealthResponse(BaseModel):
    status: str


class StockAnalysisResponse(BaseModel):
    query: str
    response: str
    ticker: str | None
    trace_id: str | None


class PortfolioAnalysisResponse(BaseModel):
    portfolio: dict[str, float]
    query: str
    response: str
    trace_id: str | None


class SavePositionsResponse(BaseModel):
    portfolio_id: str
    name: str
    positions_saved: int


class TrackPortfolioResponse(BaseModel):
    portfolio_id: str
    response: str
    trace_id: str | None
