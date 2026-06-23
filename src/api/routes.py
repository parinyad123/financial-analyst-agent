import asyncio
import json

from fastapi import APIRouter

from src.agent.core import run_financial_agent
from src.api.schemas import (
    HealthResponse,
    PortfolioAnalysisRequest,
    PortfolioAnalysisResponse,
    SavePositionsRequest,
    SavePositionsResponse,
    StockAnalysisRequest,
    StockAnalysisResponse,
    TrackPortfolioResponse,
)
from src.database.models import Portfolio, Position
from src.database.session import AsyncSessionLocal

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@router.post("/analyze/stock", response_model=StockAnalysisResponse)
async def analyze_stock(req: StockAnalysisRequest):
    tickers = [req.ticker.upper()] if req.ticker else None
    # run_financial_agent เป็น sync — ใช้ to_thread เพื่อไม่ block event loop
    # และให้ asyncio.run() ใน portfolio_track._load_positions ทำงานใน thread ได้
    result = await asyncio.to_thread(
        run_financial_agent,
        req.query,
        tickers,
        "stock_analysis",
    )
    return StockAnalysisResponse(
        query=result["query"],
        response=result["response"],
        ticker=req.ticker,
        trace_id=result["run_id"],
    )


@router.post("/analyze/portfolio", response_model=PortfolioAnalysisResponse)
async def analyze_portfolio(req: PortfolioAnalysisRequest):
    portfolio_upper = {t.upper(): v for t, v in req.portfolio.items()}
    # Normalize float amounts to int where lossless — matches tool docstring examples
    portfolio_clean = {
        t: int(v) if float(v) == int(v) else v for t, v in portfolio_upper.items()
    }
    portfolio_json = json.dumps(portfolio_clean)
    user_q = req.query.strip() if req.query else ""
    # Single-quoted JSON matches tool docstring format; explicit label prevents
    # the model from treating the portfolio line as background context and asking
    # the user to re-supply data that is already present.
    if user_q:
        query = (
            f"วิเคราะห์ความเสี่ยงพอร์ต\n"
            f"Portfolio: '{portfolio_json}'\n"
            f"คำถาม: {user_q}"
        )
    else:
        query = f"วิเคราะห์ความเสี่ยงพอร์ต: '{portfolio_json}'"
    result = await asyncio.to_thread(
        run_financial_agent,
        query,
        list(portfolio_upper.keys()),
        "portfolio_risk",
    )
    return PortfolioAnalysisResponse(
        portfolio=req.portfolio,
        query=query,
        response=result["response"],
        trace_id=result["run_id"],
    )


@router.post("/portfolio/positions", response_model=SavePositionsResponse)
async def save_positions(req: SavePositionsRequest):
    async with AsyncSessionLocal() as session:
        pf = await session.get(Portfolio, req.portfolio_id)
        if pf is None:
            pf = Portfolio(portfolio_id=req.portfolio_id, name=req.name)
            session.add(pf)

        for pos in req.positions:
            session.add(Position(
                portfolio_id=req.portfolio_id,
                ticker=pos.ticker,
                shares=pos.shares,
                avg_cost=pos.avg_cost,
            ))

        await session.commit()

    return SavePositionsResponse(
        portfolio_id=req.portfolio_id,
        name=req.name,
        positions_saved=len(req.positions),
    )


@router.get("/portfolio/{portfolio_id}", response_model=TrackPortfolioResponse)
async def get_portfolio(portfolio_id: str):
    result = await asyncio.to_thread(
        run_financial_agent,
        f"ติดตาม portfolio id: {portfolio_id}",
        None,
        "portfolio_tracking",
    )
    return TrackPortfolioResponse(
        portfolio_id=portfolio_id,
        response=result["response"],
        trace_id=result["run_id"],
    )
