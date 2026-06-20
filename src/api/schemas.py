from pydantic import BaseModel, field_validator


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
