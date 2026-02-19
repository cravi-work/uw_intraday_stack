from __future__ import annotations

from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import streamlit as st

from src.config_loader import load_yaml

st.set_page_config(layout="wide", page_title="Institutional Dashboard")

ET = ZoneInfo("America/New_York")


def get_db():
    return duckdb.connect(load_yaml("src/config/config.yaml").raw["storage"]["duckdb_path"], read_only=True)


def _has_column(con, table: str, col: str) -> bool:
    cols = [r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
    return col in cols


def render_scorecard(con):
    st.subheader("📊 Performance Scorecard (ET)")

    # Guard: older DBs may not have these columns if migrations were skipped.
    for tbl, col in [("predictions", "is_mock"), ("predictions", "outcome_realized"), ("predictions", "brier_score"), ("snapshots", "session_label")]:
        if not _has_column(con, tbl, col):
            st.warning(f"Missing {tbl}.{col}. Run one ingestion cycle to apply schema migrations.")
            return

    df = con.execute(
        """
        SELECT
          p.realized_at_utc,
          p.brier_score,
          s.session_label
        FROM predictions p
        JOIN snapshots s ON s.snapshot_id = p.snapshot_id
        WHERE p.outcome_realized = TRUE AND p.is_mock = FALSE
        """
    ).fetchdf()

    if df.empty:
        st.info("No realized (non-mock) predictions yet.")
        return

    # Robust timezone handling: treat DB timestamps as UTC.
    df["realized_at_et"] = pd.to_datetime(df["realized_at_utc"], utc=True).dt.tz_convert(ET)
    agg = (
        df.groupby("session_label")
        .agg(preds=("brier_score", "count"), mean_brier=("brier_score", "mean"))
        .sort_index()
    )
    st.dataframe(agg)


def main():
    st.title("UW Intraday Stack")

    con = get_db()
    try:
        tickers = con.execute("SELECT ticker FROM dim_tickers").fetchdf()
        if tickers.empty:
            st.info("No tickers found. Run ingestion first.")
            return

        ticker = st.sidebar.selectbox("Ticker", tickers["ticker"].tolist())

        snap = con.execute(
            """
            SELECT
              snapshot_id, asof_ts_utc, session_label, data_quality_score,
              market_close_utc, post_end_utc, seconds_to_close, is_early_close
            FROM snapshots
            WHERE ticker=?
            ORDER BY asof_ts_utc DESC
            LIMIT 1
            """,
            [ticker],
        ).fetchone()
        if not snap:
            st.info("No snapshots found for selected ticker.")
            return

        cols = st.columns(4)

        ts = pd.to_datetime(snap[1], utc=True).tz_convert(ET)
        cols[0].metric("Session", snap[2], ts.strftime("%H:%M ET"))
        cols[1].metric("Data Quality", f"{float(snap[3]):.2f}")

        with cols[2]:
            close_ts = pd.to_datetime(snap[4], utc=True).tz_convert(ET) if snap[4] else None
            post_ts = pd.to_datetime(snap[5], utc=True).tz_convert(ET) if snap[5] else None
            if bool(snap[7]):
                st.error("⚠️ EARLY CLOSE")
            if close_ts is not None:
                st.caption(f"Close: {close_ts.strftime('%H:%M')} ET")
            if post_ts is not None:
                st.caption(f"Post-Mkt: {post_ts.strftime('%H:%M')} ET")

        with cols[3]:
            if snap[6] is not None:
                st.metric("Countdown", f"{int(int(snap[6]) / 60)} min")
            else:
                st.caption("Market Closed")

        render_scorecard(con)
    finally:
        con.close()


if __name__ == "__main__":
    main()
