"""
QuantDCA & Lump Sum Backtester Pro
==================================
Application d'ingénierie quantitative pour la simulation, le backtesting historique
et la projection probabiliste de stratégies d'investissement (DCA vs Lump Sum).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import optimize
import streamlit as st
import yfinance as yf

# ==============================================================================
# 1. CONFIGURATION DE LA PAGE & CONSTANTES GLOBALES
# ==============================================================================

st.set_page_config(
    page_title="QuantBacktester Pro | DCA vs Lump Sum",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    div[data-testid="metric-container"] {
        background-color: rgba(28, 31, 48, 0.65);
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        padding-left: 18px;
        padding-right: 18px;
        border-radius: 6px 6px 0px 0px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PRECONFIGURED_ASSETS: Dict[str, Tuple[str, str, bool]] = {
    "S&P 500 - Indice Nu (^GSPC)": ("^GSPC", "USD", True),
    "S&P 500 - ETF SPY (Total Return / Div. Réinvestis)": ("SPY", "USD", False),
    "Nasdaq 100 - Indice Nu (^NDX)": ("^NDX", "USD", True),
    "Nasdaq 100 - ETF Invesco QQQ": ("QQQ", "USD", False),
    "MSCI World - iShares Core (IWDA.AS - EUR)": ("IWDA.AS", "EUR", False),
    "MSCI World - iShares ETF (URTH - USD)": ("URTH", "USD", False),
    "Amundi MSCI World PEA (CW8.PA - EUR)": ("CW8.PA", "EUR", False),
    "Euro Stoxx 50 - Indice (^STOXX50E)": ("^STOXX50E", "EUR", True),
    "CAC 40 - Indice (^FCHI)": ("^FCHI", "EUR", True),
    "Or Physique - SPDR Gold Shares (GLD)": ("GLD", "USD", False),
    "Bitcoin (BTC-USD)": ("BTC-USD", "USD", False),
}

COLOR_PALETTE: List[str] = [
    "#00D4FF",
    "#FF6B6B",
    "#10B981",
    "#F59E0B",
    "#A855F7",
    "#EC4899",
    "#3B82F6",
    "#84CC16",
]

# ==============================================================================
# 2. STRUCTURES DE DONNÉES (DATACLASSES)
# ==============================================================================


@dataclass
class SimulationParams:
    start_date: date
    end_date: date
    initial_capital: float
    dca_amount: float
    dca_frequency: str
    dca_day_rule: str
    fee_mode: str
    fee_pct: float
    fee_min_eur: float
    fee_fixed_eur: float
    annual_ter: float
    apply_ter_to_etfs: bool
    tax_regime_name: str
    tax_rate: float
    inflation_rate: float
    risk_free_rate: float
    convert_usd_to_eur: bool


@dataclass
class BacktestMetrics:
    total_invested: float
    final_gross_value: float
    final_net_value: float
    final_real_net_value: float
    gross_pnl: float
    net_pnl: float
    net_return_pct: float
    tax_paid: float
    total_brokerage_fees: float
    xirr: float
    underlying_cagr: float
    portfolio_cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_underwater_days: int
    total_shares: float
    final_pru: float
    final_market_price: float
    num_orders: int


@dataclass
class AssetBacktestResult:
    asset_label: str
    ticker: str
    currency: str
    daily_df: pd.DataFrame
    dca_metrics: BacktestMetrics
    lump_sum_metrics: BacktestMetrics
    cashflows_dca: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)
    cashflows_ls: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)


# ==============================================================================
# 3. COUCHE D'ACQUISITION & PRÉTRAITEMENT DES DONNÉES
# ==============================================================================


def _extract_price_series(raw_df: pd.DataFrame, ticker: str) -> pd.Series:
    if raw_df is None or raw_df.empty:
        return pd.Series(dtype=float)

    df = raw_df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        level_0 = df.columns.get_level_values(0).unique().tolist()
        level_1 = df.columns.get_level_values(1).unique().tolist()

        for col_choice in ["Adj Close", "Close"]:
            if col_choice in level_0:
                sub = df[col_choice]
                if isinstance(sub, pd.DataFrame):
                    if ticker in sub.columns:
                        return sub[ticker].astype(float)
                    return sub.iloc[:, 0].astype(float)
                return sub.astype(float)
            if col_choice in level_1:
                sub = df.xs(col_choice, axis=1, level=1)
                if isinstance(sub, pd.DataFrame):
                    if ticker in sub.columns:
                        return sub[ticker].astype(float)
                    return sub.iloc[:, 0].astype(float)
                return sub.astype(float)

        return df.iloc[:, 0].astype(float)

    for col in ["Adj Close", "Close", "adj close", "close"]:
        if col in df.columns:
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            return series.astype(float)

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        return df[num_cols[0]].astype(float)

    return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_series(
    ticker: str, start_date: date, end_date: date
) -> Tuple[Optional[pd.Series], str, Optional[str]]:
    try:
        dl_start = (pd.Timestamp(start_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        dl_end = (pd.Timestamp(end_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

        raw = yf.download(
            tickers=ticker
