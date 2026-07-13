import asyncio
import json

from fastapi import APIRouter, HTTPException

from src.agent.core import run_financial_agent, thread_has_history
from src.api.schemas import (
    AskPortfolioRequest,
    AskPortfolioResponse,
    HealthResponse,
    PortfolioAnalysisRequest,
    PortfolioAnalysisResponse,
    PortfolioListItem,
    PortfolioListResponse,
    SavePositionsRequest,
    SavePositionsResponse,
    StockAnalysisRequest,
    StockAnalysisResponse,
    TrackPortfolioResponse,
    validate_new_portfolio_id,
)
from src.database.models import Portfolio, Position
from src.database.session import AsyncSessionLocal
from src.tools.portfolio_track import (
    _list_portfolios,
    _load_positions,
    _track_portfolio_logic,
)

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
        req.session_id,
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
        req.session_id,
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
            try:
                validate_new_portfolio_id(req.portfolio_id)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
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


@router.get("/portfolios", response_model=PortfolioListResponse)
async def list_portfolios():
    """List saved portfolios (id, name, tickers) — powers the tracking-tab
    dropdown that lets users pick by name while the system uses portfolio_id."""
    rows = await asyncio.to_thread(_list_portfolios)
    return PortfolioListResponse(
        portfolios=[
            PortfolioListItem(portfolio_id=pid, name=name, tickers=tickers)
            for pid, name, tickers in rows
        ]
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


@router.post("/portfolio/{portfolio_id}/ask", response_model=AskPortfolioResponse)
async def ask_portfolio(portfolio_id: str, req: AskPortfolioRequest):
    """Ask a free-text question about a saved portfolio, in the same page as the
    tracking view. Routes through the GENERAL agent (not track_portfolio) so the
    planner can freely pick news/hurst/price tools — the current track report is
    injected as context so portfolio-level questions ("which is riskiest") are
    answerable without re-calling a tool.

    CRITICAL: the injected context must NOT contain the portfolio_id as a token,
    or _plan_override (src/agent/core.py) would match it and force track_portfolio,
    defeating the free routing. We build the context as a natural-language
    description of the holdings and drop the report's header line (the only place
    the id appears) so the id never enters the context in the first place — with a
    final .replace() as a defense-in-depth safety net."""
    user_q = (req.query or "").strip()
    if not user_q:
        raise HTTPException(status_code=422, detail="query ว่าง")

    # 1) Load positions → 404 if unknown + get tickers for the trace/context
    try:
        _pf_name, positions = await asyncio.to_thread(_load_positions, portfolio_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"ไม่พบ portfolio_id '{portfolio_id}'"
        ) from exc
    if not positions:
        raise HTTPException(
            status_code=404, detail=f"พอร์ต '{portfolio_id}' ยังไม่มี position"
        )
    tickers = sorted({p["ticker"].upper() for p in positions})

    # 2) Inject the report ONCE per conversation thread. On follow-up turns it is already
    #    in the thread's history, so re-injecting would duplicate a multi-KB block every
    #    turn and leave several conflicting snapshots in context.
    if await asyncio.to_thread(thread_has_history, req.session_id):
        query = user_q
    else:
        report = await asyncio.to_thread(_track_portfolio_logic, portfolio_id)

        # 3) Frame as natural language — the id lives ONLY in the report's header line
        #    ("Portfolio: {name} (id: {id})"), so drop it; the body is tickers + numbers.
        #    positions guaranteed non-empty above → report is always multi-line here.
        report_body = report.split("\n", 1)[1] if "\n" in report else report
        # Frame as the system's OWN tool-computed analysis (single voice) — otherwise the
        # model treats the injected block as third-party "given data" and hedges ("ข้อมูลที่
        # ให้มา"), which also collides with the system prompt's "numbers must come from a
        # tool" rule. This legitimizes the figures as real tool output without inviting the
        # model to invent numbers the report does not contain.
        context = (
            f"ต่อไปนี้คือผลวิเคราะห์พอร์ต (หุ้น: {', '.join(tickers)}) ที่ระบบคำนวณจากเครื่องมือ "
            f"risk analysis แล้ว — ใช้ตัวเลขเหล่านี้ตอบได้เต็มที่เสมือนเป็นผลวิเคราะห์ของคุณเอง "
            f"ห้ามพูดทำนอง 'ข้อมูลที่ให้มา' หรือ 'เท่าที่มีข้อมูล':\n{report_body}"
        )
        # safety net (defense-in-depth) — never let a stray id reach _plan_override
        context = context.replace(portfolio_id, "<พอร์ตนี้>")
        query = f"{context}\n\n[คำถาม] {user_q}"

    result = await asyncio.to_thread(
        run_financial_agent,
        query,
        tickers or None,
        "stock_analysis",
        req.session_id,
    )
    return AskPortfolioResponse(
        portfolio_id=portfolio_id,
        query=user_q,
        response=result["response"],
        trace_id=result["run_id"],
        session_id=result["session_id"],
    )
