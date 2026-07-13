"""
Streamlit UI — Financial Analyst Agent
Calls FastAPI at http://localhost:8000. Start uvicorn main:app first.
"""

import re
import uuid

import requests
import streamlit as st

API_BASE = "http://localhost:8000"
TIMEOUT = 180  # agent calls can take ~60-90s for multi-tool queries


# ─── helpers ─────────────────────────────────────────────────────────────────

def _call(method: str, path: str, **kwargs) -> dict:
    try:
        r = requests.request(method, f"{API_BASE}{path}", timeout=TIMEOUT, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "**ไม่สามารถเชื่อมต่อ API ได้** — กรุณาเปิด FastAPI server ก่อนใช้งาน:\n\n"
            "```\nuvicorn main:app --reload\n```"
        )
        st.stop()
    except requests.exceptions.Timeout:
        st.error("Request timeout — agent ใช้เวลานานเกินไป ลองใหม่อีกครั้ง")
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text[:300]}")
        st.stop()


def _split_response(text: str) -> tuple[str, str]:
    """Return (summary, full_text).
    Summary = last chunk that is not a pure markdown table, not a bare
    single-line header, and not a short disclaimer (< 60 chars).
    Agent responses wrap section headers + content in one chunk (### + bullets),
    so we only filter bare standalone headers, not header-prefixed content blocks.
    """
    chunks = [c.strip() for c in re.split(r"\n{2,}", text) if c.strip()]
    summary = ""
    for chunk in reversed(chunks):
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        is_pure_table = lines and all(ln.strip().startswith("|") for ln in lines)
        is_bare_header = len(lines) == 1 and chunk.startswith("#")
        is_too_short = len(chunk) < 60
        if not is_pure_table and not is_bare_header and not is_too_short:
            summary = chunk
            break
    if not summary:
        summary = chunks[-1] if chunks else text
    return summary, text


def _show_result(data: dict) -> None:
    """Two-tier display: summary (prominent) + full technical detail in expander."""
    response = data.get("response", "")
    trace_id = data.get("trace_id")
    summary, full = _split_response(response)

    st.success(summary)
    with st.expander("รายละเอียดเชิงเทคนิค"):
        st.markdown(full)
        if trace_id:
            st.caption(f"trace_id: `{trace_id}`")


# ─── page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Financial Analyst Agent",
    page_icon="📊",
    layout="centered",
)
st.title("📊 Financial Analyst Agent")
st.caption(
    "Powered by LangGraph StateGraph + Groq · ข้อมูลจาก yfinance · "
    "**ไม่ใช่คำแนะนำการลงทุน**"
)

# ─── session_state init ───────────────────────────────────────────────────────

for key, default in [
    ("tab1_query_area", ""),
    ("tab2_row_ids", [0, 1]),
    ("tab2_next_id", 2),
    ("tab2_query_area", ""),
    ("tab3_row_ids", [0]),
    ("tab3_next_id", 1),
    ("tab3_ask_area", ""),
    ("tab3_view_result", None),   # persisted GET /portfolio/{id} result
    ("tab3_ask_result", None),    # persisted POST /portfolio/{id}/ask result
    ("tab3_active_id", None),     # portfolio_id currently viewed/asked
    # ── conversation-memory threads (see docs: thread scoping) ────────────────
    # Each tab owns a checkpointer thread. Scoping differs on purpose:
    #   tab1 — per session, reset by the "เริ่มบทสนทนาใหม่" button
    #   tab2 — per portfolio composition: changing the form MUST start a new thread,
    #          or the previous allocation's risk figures linger and get misattributed
    #   tab3 — per tracking report load: "ดูพอร์ต" mints a new thread so the injected
    #          snapshot is fresh and injected exactly once
    ("tab1_thread", str(uuid.uuid4())),
    ("tab1_history", []),         # [(question, result_dict)] for display
    ("tab2_thread", str(uuid.uuid4())),
    ("tab2_pf_hash", None),       # portfolio composition the tab2 thread belongs to
    ("tab3_thread", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(
    ["💬 ถามทั่วไป", "📈 วิเคราะห์ Risk พอร์ต", "📋 ติดตามพอร์ต"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ถามทั่วไป  →  POST /analyze/stock
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("ถามเรื่องหุ้นรายตัว")

    ticker1 = st.text_input(
        "Ticker",
        placeholder="เช่น NVDA, TSLA, AMD",
        key="tab1_ticker",
    ).strip().upper()

    st.write("**คำถามด่วน**")
    q1a, q1b, q1c, q1d = st.columns(4)
    with q1a:
        if st.button("🔍 วิเคราะห์แบบเต็ม", use_container_width=True, key="qb1_full"):
            st.session_state["tab1_query_area"] = (
                f"วิเคราะห์ {ticker1} ให้หน่อย: ราคา, fundamentals, Hurst"
            )
    with q1b:
        if st.button("💰 ราคาปัจจุบัน", use_container_width=True, key="qb1_price"):
            st.session_state["tab1_query_area"] = f"ราคา {ticker1} ตอนนี้เท่าไหร่"
    with q1c:
        if st.button("📐 Trending/Mean-Rev", use_container_width=True, key="qb1_hurst"):
            st.session_state["tab1_query_area"] = (
                f"ตอนนี้ {ticker1} เป็น trending หรือ mean-reverting"
            )
    with q1d:
        if st.button("📰 ทำไมขึ้น/ลง", use_container_width=True, key="qb1_news"):
            st.session_state["tab1_query_area"] = (
                f"ทำไมหุ้น {ticker1} ขึ้นหรือลงช่วงนี้"
            )

    query1 = st.text_area(
        "หรือพิมพ์คำถามเองได้เลย",
        key="tab1_query_area",
        height=80,
        placeholder="เช่น วิเคราะห์ NVDA ให้หน่อย — ราคา, fundamentals, และ regime",
    )

    st.caption(
        "ℹ️ กรอก Ticker แล้วกดปุ่มด่วน หรือพิมพ์คำถามเองได้เลย — "
        "agent **จำบทสนทนาก่อนหน้าได้** ถามต่อเนื่องได้เลย เช่น 'แล้วข่าวล่ะ'"
    )

    c_ask, c_new = st.columns([3, 1])
    with c_ask:
        submitted = st.button("วิเคราะห์", type="primary", key="tab1_submit")
    with c_new:
        if st.button("🔄 เริ่มบทสนทนาใหม่", use_container_width=True, key="tab1_reset"):
            st.session_state["tab1_thread"] = str(uuid.uuid4())
            st.session_state["tab1_history"] = []
            st.rerun()

    if submitted:
        if not query1.strip():
            st.warning("กรุณากรอกคำถามหรือกดปุ่มด่วนด้านบน")
        else:
            # inject ticker into query if user filled the field but query doesn't mention it
            effective_query = query1
            if ticker1 and ticker1 not in query1.upper():
                effective_query = f"[Ticker: {ticker1}] {query1}"
            with st.spinner("กำลังวิเคราะห์ (อาจใช้เวลา 30–60 วินาที)..."):
                data = _call(
                    "POST",
                    "/analyze/stock",
                    json={
                        "query": effective_query,
                        "ticker": ticker1 or None,
                        "session_id": st.session_state["tab1_thread"],
                    },
                )
            st.session_state["tab1_history"].append((query1, data))

    # newest first — the agent keeps the real memory server-side; this is just the transcript
    history = st.session_state["tab1_history"]
    if history:
        st.divider()
        st.caption(f"บทสนทนานี้ ({len(history)} คำถาม) — กด 'เริ่มบทสนทนาใหม่' เพื่อล้าง")
        for q, data in reversed(history):
            st.markdown(f"**❓ {q}**")
            _show_result(data)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — วิเคราะห์ Risk พอร์ต  →  POST /analyze/portfolio
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("วิเคราะห์ความเสี่ยงพอร์ต (What-If)")
    st.caption(
        "ใส่จำนวนเงิน USD ต่อ ticker — "
        "ระบบ normalize เป็น weight ให้อัตโนมัติ (amount ÷ total)"
    )

    # Dynamic ticker rows
    rows_to_del = []
    for rid in list(st.session_state.tab2_row_ids):
        c1, c2, c3 = st.columns([3, 3, 1])
        with c1:
            st.text_input("Ticker", placeholder="NVDA", key=f"t2_tk_{rid}")
        with c2:
            st.number_input(
                "Amount (USD)", min_value=0.0, step=500.0, key=f"t2_amt_{rid}"
            )
        with c3:
            st.write("")  # vertical pad to align button
            if len(st.session_state.tab2_row_ids) > 1:
                if st.button("−", key=f"t2_del_{rid}"):
                    rows_to_del.append(rid)

    if rows_to_del:
        for rid in rows_to_del:
            st.session_state.tab2_row_ids.remove(rid)
        st.rerun()

    if st.button("+ เพิ่ม Ticker", key="tab2_add"):
        st.session_state.tab2_row_ids.append(st.session_state.tab2_next_id)
        st.session_state.tab2_next_id += 1
        st.rerun()

    st.divider()

    # Optional query with quick-question buttons
    st.write("**คำถามเสริม** (ไม่บังคับ — ถ้าว่างระบบวิเคราะห์ครบทุก metric)")
    q2a, q2b, q2c = st.columns(3)
    with q2a:
        if st.button("🔗 อธิบาย Correlation", use_container_width=True, key="qb2_corr"):
            st.session_state["tab2_query_area"] = (
                "เน้นอธิบาย correlation matrix และ rolling correlation — "
                "ticker ไหนเคลื่อนไหวพร้อมกัน ส่งผลต่อการกระจายความเสี่ยงยังไง"
            )
    with q2b:
        if st.button("📉 อธิบาย Drawdown", use_container_width=True, key="qb2_dd"):
            st.session_state["tab2_query_area"] = (
                "เน้นอธิบาย max drawdown, Ulcer Index และ drawdown duration — "
                "พอร์ตนี้ถ้าตลาดลง pain level เป็นยังไง"
            )
    with q2c:
        if st.button("⚖️ Risk-Return Tradeoff", use_container_width=True, key="qb2_rr"):
            st.session_state["tab2_query_area"] = (
                "เน้นอธิบาย Sharpe, Calmar, Sortino — "
                "ผลตอบแทนที่ได้คุ้มกับความเสี่ยงที่รับไหม"
            )

    query2 = st.text_area(
        "คำถามเสริม",
        key="tab2_query_area",
        height=80,
        placeholder="(ว่างได้ — ระบบวิเคราะห์ครบ metrics อัตโนมัติ)",
    )

    st.caption(
        "⚠️ **ระบบนี้ไม่รองรับ:** stress test ('ถ้าราคาตก 20%'), "
        "causal cross-asset ('ถ้า Intel ฟื้นจะกระทบยังไง') — "
        "วิเคราะห์ได้เฉพาะ correlation/risk metrics จากข้อมูล historical ที่มี"
    )

    if st.button("วิเคราะห์ Risk", type="primary", key="tab2_submit"):
        portfolio = {
            st.session_state[f"t2_tk_{rid}"].strip().upper(): float(
                st.session_state.get(f"t2_amt_{rid}", 0.0)
            )
            for rid in st.session_state.tab2_row_ids
            if st.session_state.get(f"t2_tk_{rid}", "").strip()
            and st.session_state.get(f"t2_amt_{rid}", 0.0) > 0
        }
        if not portfolio:
            st.warning("กรุณากรอก Ticker และ Amount (USD > 0) อย่างน้อย 1 ตัว")
        else:
            # A changed allocation is a different portfolio — start a new memory thread so
            # the previous one's risk figures cannot be carried over and misattributed.
            pf_hash = str(sorted(portfolio.items()))
            if pf_hash != st.session_state["tab2_pf_hash"]:
                st.session_state["tab2_thread"] = str(uuid.uuid4())
                st.session_state["tab2_pf_hash"] = pf_hash
            with st.spinner(f"กำลังวิเคราะห์พอร์ต {list(portfolio.keys())} ..."):
                data = _call(
                    "POST",
                    "/analyze/portfolio",
                    json={
                        "portfolio": portfolio,
                        "query": query2,
                        "session_id": st.session_state["tab2_thread"],
                    },
                )
            _show_result(data)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ติดตามพอร์ต
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("ติดตามพอร์ตที่ถืออยู่")

    # ── Section A: บันทึกพอร์ตใหม่  →  POST /portfolio/positions ────────────
    st.markdown("### บันทึกพอร์ตใหม่")
    st.caption(
        "ระบุจำนวนหุ้นจริงและราคาเฉลี่ยที่ซื้อ — "
        "ระบบดึงราคาปัจจุบันและคำนวณ unrealized P&L ให้"
    )

    pf_id = st.text_input(
        "Portfolio ID", placeholder="เช่น my-tech-2025", key="tab3_pf_id"
    ).strip()
    st.caption(
        "ใช้ชื่อนี้ถามใน Tab \"ถามทั่วไป\" ได้เลย เช่น 'พอร์ต my-tech-2025 เสี่ยงแค่ไหน' "
        "— ระบบจะจำพอร์ตนี้ได้จากชื่อโดยตรง"
    )
    pf_name = st.text_input(
        "ชื่อพอร์ต", placeholder="เช่น Tech Portfolio 2025", key="tab3_pf_name"
    ).strip()

    # Dynamic position rows
    pos_to_del = []
    for rid in list(st.session_state.tab3_row_ids):
        c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
        with c1:
            st.text_input("Ticker", placeholder="NVDA", key=f"t3_tk_{rid}")
        with c2:
            st.number_input("Shares", min_value=0.0, step=1.0, key=f"t3_sh_{rid}")
        with c3:
            st.number_input(
                "Avg Cost (USD/share)", min_value=0.0, step=1.0, key=f"t3_ac_{rid}"
            )
        with c4:
            st.write("")
            if len(st.session_state.tab3_row_ids) > 1:
                if st.button("−", key=f"t3_del_{rid}"):
                    pos_to_del.append(rid)

    if pos_to_del:
        for rid in pos_to_del:
            st.session_state.tab3_row_ids.remove(rid)
        st.rerun()

    if st.button("+ เพิ่ม Position", key="tab3_add"):
        st.session_state.tab3_row_ids.append(st.session_state.tab3_next_id)
        st.session_state.tab3_next_id += 1
        st.rerun()

    st.caption(
        "ℹ️ ถ้า portfolio_id ซ้ำ ระบบเพิ่ม position เข้าพอร์ตเดิม — "
        "ใช้ ID ใหม่ถ้าต้องการพอร์ตแยก"
    )

    if st.button("บันทึกพอร์ต", type="primary", key="tab3_save"):
        positions = [
            {
                "ticker": st.session_state.get(f"t3_tk_{rid}", "").strip().upper(),
                "shares": float(st.session_state.get(f"t3_sh_{rid}", 0.0)),
                "avg_cost": float(st.session_state.get(f"t3_ac_{rid}", 0.0)),
            }
            for rid in st.session_state.tab3_row_ids
            if st.session_state.get(f"t3_tk_{rid}", "").strip()
            and st.session_state.get(f"t3_sh_{rid}", 0.0) > 0
            and st.session_state.get(f"t3_ac_{rid}", 0.0) > 0
        ]
        if not pf_id:
            st.warning("กรุณากรอก Portfolio ID")
        elif not pf_name:
            st.warning("กรุณากรอกชื่อพอร์ต")
        elif not positions:
            st.warning("กรุณากรอก Position อย่างน้อย 1 รายการ (Ticker + Shares + Avg Cost > 0)")
        else:
            with st.spinner("กำลังบันทึก..."):
                data = _call(
                    "POST",
                    "/portfolio/positions",
                    json={"portfolio_id": pf_id, "name": pf_name, "positions": positions},
                )
            st.success(
                f"บันทึกพอร์ต **{data['name']}** (`{data['portfolio_id']}`) สำเร็จ — "
                f"{data['positions_saved']} position(s)"
            )

    st.divider()

    # ── Section B: ดูพอร์ต + ถามต่อ  →  GET /portfolio/{id} + POST /portfolio/{id}/ask ──
    st.markdown("### ดูพอร์ตที่บันทึกไว้")
    st.caption(
        "เลือกพอร์ตจากชื่อ — agent ดึงราคาปัจจุบัน รายงาน P&L, market value, "
        "current weight แล้วถามต่อเจาะจงได้ในหน้านี้เลย"
    )

    pf_list = _call("GET", "/portfolios").get("portfolios", [])
    if not pf_list:
        st.info("ยังไม่มีพอร์ตที่บันทึกไว้ — สร้างพอร์ตใหม่ในส่วนด้านบนก่อน")
    else:
        # label แสดง "ชื่อ (id)" แต่ระบบใช้ portfolio_id เบื้องหลัง (ชื่อใน DB ไม่ unique)
        options = {f"{p['name']} ({p['portfolio_id']})": p for p in pf_list}
        picked = options[
            st.selectbox("เลือกพอร์ต", list(options.keys()), key="tab3_pick")
        ]
        view_id = picked["portfolio_id"]

        if st.button("ดูพอร์ต", type="primary", key="tab3_view"):
            with st.spinner(f"กำลังโหลดพอร์ต '{picked['name']}' ..."):
                data = _call("GET", f"/portfolio/{view_id}")
            st.session_state["tab3_view_result"] = data
            st.session_state["tab3_ask_result"] = None  # reset Q&A เก่าของพอร์ตก่อนหน้า
            st.session_state["tab3_active_id"] = view_id
            # New report load = new memory thread: the backend injects this snapshot once
            # per thread, so a fresh thread guarantees the Q&A is about the report shown.
            st.session_state["tab3_thread"] = str(uuid.uuid4())

        # persist report ข้าม rerun (กด "ถาม" ทำให้ rerun — report ต้องไม่หาย)
        if (
            st.session_state["tab3_view_result"]
            and st.session_state["tab3_active_id"] == view_id
        ):
            _show_result(st.session_state["tab3_view_result"])

            st.divider()
            st.write("**ถามเจาะจงพอร์ตนี้**")
            tickers = picked.get("tickers", [])
            first_tk = tickers[0] if tickers else "หุ้น"
            a1, a2, a3 = st.columns(3)
            with a1:
                if st.button("⚠️ ตัวไหนเสี่ยงสุด", use_container_width=True, key="qb3_risk"):
                    st.session_state["tab3_ask_area"] = (
                        "ในพอร์ตนี้ หุ้นตัวไหน contribute ความเสี่ยง (variance) มากสุด เพราะอะไร"
                    )
            with a2:
                if st.button(f"📰 ข่าว {first_tk}", use_container_width=True, key="qb3_news"):
                    st.session_state["tab3_ask_area"] = f"{first_tk} มีข่าวอะไรล่าสุดที่ส่งผลต่อราคา"
            with a3:
                if st.button(f"📐 {first_tk} trending?", use_container_width=True, key="qb3_hurst"):
                    st.session_state["tab3_ask_area"] = (
                        f"ตอนนี้ {first_tk} เป็น trending หรือ mean-reverting"
                    )

            ask_q = st.text_area(
                "คำถาม",
                key="tab3_ask_area",
                height=80,
                placeholder="เช่น ตัวไหนขาดทุนอยู่ / NVDA มีข่าวอะไร / TSLA trending ไหม",
            )
            st.caption(
                "ℹ️ ถามเรื่องหุ้นในพอร์ตได้อิสระ (ข่าว/regime/ราคา) และคำถามระดับพอร์ต "
                "('ตัวไหนเสี่ยงสุด') — **ถามต่อเนื่องได้** agent จำบทสนทนาในพอร์ตนี้ "
                "(กด 'ดูพอร์ต' ใหม่ = เริ่มบทสนทนาใหม่)"
            )

            if st.button("ถาม", type="primary", key="tab3_ask"):
                if not ask_q.strip():
                    st.warning("กรุณาพิมพ์คำถามหรือกดปุ่มด่วน")
                else:
                    with st.spinner("กำลังตอบ (อาจใช้เวลา 30–60 วินาที)..."):
                        data = _call(
                            "POST",
                            f"/portfolio/{view_id}/ask",
                            json={
                                "query": ask_q,
                                "session_id": st.session_state["tab3_thread"],
                            },
                        )
                    st.session_state["tab3_ask_result"] = data

        if (
            st.session_state["tab3_ask_result"]
            and st.session_state["tab3_active_id"] == view_id
        ):
            st.markdown("#### คำตอบ")
            _show_result(st.session_state["tab3_ask_result"])
