from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import optimize
import streamlit as st
import yfinance as yf

# ==============================================================================
# 1. CONFIGURATION & DESIGN "MOBILE FINTECH" (CSS)
# ==============================================================================

st.set_page_config(
    page_title="Mon Simulateur Bourse",
    page_icon="💎",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Adaptation mobile et respiration visuelle */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 880px !important;
    }
    /* Grande carte principale (Hero Card style Finary / Revolut) */
    .hero-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 18px;
        padding: 22px 20px;
        margin-top: 10px;
        margin-bottom: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        text-align: center;
    }
    .hero-label {
        color: #94A3B8;
        font-size: 0.92rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .hero-value {
        color: #F8FAFC;
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 10px;
    }
    .hero-badge-pos {
        display: inline-block;
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.95rem;
        font-weight: 600;
    }
    .hero-badge-neg {
        display: inline-block;
        background: rgba(239, 68, 68, 0.18);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.95rem;
        font-weight: 600;
    }
    /* Grille de mini-cartes 2x2 sur mobile */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin-bottom: 18px;
    }
    .kpi-box {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 14px 16px;
    }
    .kpi-title {
        color: #94A3B8;
        font-size: 0.80rem;
        margin-bottom: 4px;
    }
    .kpi-num {
        color: #F1F5F9;
        font-size: 1.25rem;
        font-weight: 700;
    }
    .kpi-sub {
        color: #38BDF8;
        font-size: 0.78rem;
        margin-top: 2px;
    }
    /* Boîte de résumé en français clair */
    .insight-box {
        background: rgba(14, 165, 233, 0.10);
        border-left: 4px solid #38BDF8;
        border-radius: 0 12px 12px 0;
        padding: 14px 16px;
        margin-bottom: 20px;
        font-size: 0.95rem;
        line-height: 1.5;
        color: #E2E8F0;
    }
    /* Onglets plus larges et tactiles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding-left: 12px;
        padding-right: 12px;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Catalogue enrichi avec actions PEA / US populaires et ETFs majeurs
ASSET_CATALOG: Dict[str, Tuple[str, str, bool]] = {
    "🌍 MSCI World PEA (Amundi CW8)": ("CW8.PA", "EUR", False),
    "🇺🇸 S&P 500 ETF (SPY - Dividendes inclus)": ("SPY", "USD", False),
    "💻 Nasdaq 100 ETF (Invesco QQQ)": ("QQQ", "USD", False),
    "🌍 MSCI World Core (iShares IWDA)": ("IWDA.AS", "EUR", False),
    "🇫🇷 Schneider Electric (SU.PA - PEA)": ("SU.PA", "EUR", False),
    "🇫🇷 Air Liquide (AI.PA - PEA)": ("AI.PA", "EUR", False),
    "🇳🇱 ASML Holding (ASML.AS - PEA)": ("ASML.AS", "EUR", False),
    "🇫🇷 LVMH (MC.PA - PEA)": ("MC.PA", "EUR", False),
    "🇺🇸 Nvidia (NVDA - CTO)": ("NVDA", "USD", False),
    "🇺🇸 Microsoft (MSFT - CTO)": ("MSFT", "USD", False),
    "🇫🇷 CAC 40 Indice (^FCHI)": ("^FCHI", "EUR", True),
    "🇪🇺 Euro Stoxx 50 Indice (^STOXX50E)": ("^STOXX50E", "EUR", True),
    "🥇 Or Physique (ETF GLD)": ("GLD", "USD", False),
    "₿ Bitcoin (BTC-USD)": ("BTC-USD", "USD", False),
}

COLOR_PALETTE: List[str] = [
    "#38BDF8",
    "#A855F7",
    "#10B981",
    "#F59E0B",
    "#F43F5E",
    "#3B82F6",
    "#EC4899",
    "#84CC16",
]

# ==============================================================================
# 2. STRUCTURES DE DONNÉES
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
# 3. ACQUISITION DES DONNÉES & MOTEUR QUANTITATIF
# ==============================================================================


def _extract_price_series(raw_df: pd.DataFrame, ticker: str) -> pd.Series:
    if raw_df is None or raw_df.empty:
        return pd.Series(dtype=float)
    df = raw_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        l0 = df.columns.get_level_values(0).unique().tolist()
        l1 = df.columns.get_level_values(1).unique().tolist()
        for col_choice in ["Adj Close", "Close"]:
            if col_choice in l0:
                sub = df[col_choice]
                if isinstance(sub, pd.DataFrame):
                    return (sub[ticker] if ticker in sub.columns else sub.iloc[:, 0]).astype(float)
                return sub.astype(float)
            if col_choice in l1:
                sub = df.xs(col_choice, axis=1, level=1)
                if isinstance(sub, pd.DataFrame):
                    return (sub[ticker] if ticker in sub.columns else sub.iloc[:, 0]).astype(float)
                return sub.astype(float)
        return df.iloc[:, 0].astype(float)

    for col in ["Adj Close", "Close", "adj close", "close"]:
        if col in df.columns:
            s = df[col]
            return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s).astype(float)
    num_cols = df.select_dtypes(include=[np.number]).columns
    return df[num_cols[0]].astype(float) if len(num_cols) > 0 else pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_series(
    ticker: str, start_date: date, end_date: date
) -> Tuple[Optional[pd.Series], str, Optional[str]]:
    try:
        dl_start = (pd.Timestamp(start_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        dl_end = (pd.Timestamp(end_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        raw = yf.download(
            tickers=ticker,
            start=dl_start,
            end=dl_end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            return None, "USD", f"Aucune donnée trouvée pour '{ticker}'."
        series = _extract_price_series(raw, ticker).dropna()
        if series.empty:
            return None, "USD", f"Série de prix vide pour '{ticker}'."
        series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
        series = series[~series.index.duplicated(keep="last")].sort_index()

        upper_t = ticker.upper()
        currency = (
            "EUR"
            if upper_t.endswith((".PA", ".AS", ".DE", ".MI", ".MC", ".BR"))
            or upper_t in ["^STOXX50E", "^FCHI", "^GDAXI"]
            else "USD"
        )
        eff_start = max(pd.Timestamp(start_date), series.index.min())
        eff_end = min(pd.Timestamp(end_date), series.index.max())
        if eff_start >= eff_end:
            return None, currency, f"Pas de cotation sur cette période pour '{ticker}'."

        cal = pd.date_range(start=eff_start, end=eff_end, freq="D")
        series = series.reindex(cal.union(series.index)).sort_index().ffill().bfill().reindex(cal)
        return series, currency, None
    except Exception as exc:
        return None, "USD", f"Erreur réseau sur '{ticker}' : {str(exc)}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_eurusd_fx_series(start_date: date, end_date: date) -> pd.Series:
    fx, _, err = fetch_yahoo_series("EURUSD=X", start_date, end_date)
    if fx is None or fx.empty or err is not None:
        return pd.Series(
            1.10, index=pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq="D")
        )
    return fx


def parse_custom_csv(
    uploaded_file: io.BytesIO, start_date: date, end_date: date
) -> Tuple[Optional[pd.Series], str, Optional[str]]:
    try:
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        first_line = content.splitlines()[0] if content else ""
        sep = ";" if ";" in first_line else ("\t" if "\t" in first_line else ",")
        df = pd.read_csv(io.StringIO(content), sep=sep)
        if df.empty or len(df.columns) < 2:
            return None, "EUR", "Le CSV doit contenir une colonne Date et une colonne Prix."
        date_col = next(
            (c for c in df.columns if any(k in str(c).lower() for k in ["date", "time", "jour", "day"])),
            df.columns[0],
        )
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        df = df.dropna(subset=[date_col]).sort_values(by=date_col)
        kws = ["adj close", "close", "clôture", "cloture", "prix", "price", "valeur", "nav", "vl"]
        price_col = next(
            (c for k in kws for c in df.columns if c != date_col and k in str(c).lower()),
            [c for c in df.columns if c != date_col][0],
        )
        if df[price_col].dtype == object:
            df[price_col] = pd.to_numeric(
                df[price_col]
                .astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace("€", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace(",", ".", regex=False),
                errors="coerce",
            )
        else:
            df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        df = df.dropna(subset=[price_col])
        df = df[df[price_col] > 0]
        if df.empty:
            return None, "EUR", "Aucun prix valide trouvé dans le CSV."
        s = pd.Series(df[price_col].values, index=df[date_col])
        s = s[~s.index.duplicated(keep="last")].sort_index()
        eff_s, eff_e = max(pd.Timestamp(start_date), s.index.min()), min(pd.Timestamp(end_date), s.index.max())
        if eff_s >= eff_e:
            return None, "EUR", "Les dates du CSV ne recoupent pas la période choisie."
        cal = pd.date_range(start=eff_s, end=eff_e, freq="D")
        return s.reindex(cal.union(s.index)).sort_index().ffill().bfill().reindex(cal), "EUR", None
    except Exception as exc:
        return None, "EUR", f"Erreur lecture CSV : {str(exc)}"


def generate_dca_schedule(
    calendar_index: pd.DatetimeIndex, frequency: str, day_rule: str
) -> pd.DatetimeIndex:
    if calendar_index.empty:
        return pd.DatetimeIndex([])
    start_ts, end_ts = calendar_index.min(), calendar_index.max()
    if frequency == "Hebdomadaire":
        t = calendar_index[calendar_index.weekday == 0]
        return t[t > start_ts]

    mdf = pd.DataFrame(
        {
            "date": calendar_index,
            "y": calendar_index.year,
            "m": calendar_index.month,
            "b": calendar_index.weekday < 5,
        }
    )
    ym = mdf.groupby(["y", "m"], as_index=False)
    step = 1 if frequency == "Mensuelle" else 3
    selected: List[pd.Timestamp] = []
    for idx_g, k in enumerate(list(ym.groups.keys())):
        if idx_g % step != 0:
            continue
        sub = ym.get_group(k)
        bdays = sub[sub["b"]]["date"] if not sub[sub["b"]].empty else sub["date"]
        if day_rule == "1er du mois":
            ch = bdays.iloc[0]
        elif day_rule == "15 du mois":
            c15 = bdays[bdays.dt.day >= 15]
            ch = c15.iloc[0] if not c15.empty else bdays.iloc[-1]
        else:
            ch = bdays.iloc[-1]
        if start_ts < ch <= end_ts:
            selected.append(ch)
    return pd.DatetimeIndex(sorted(set(selected)))


def compute_xirr(cashflows: List[Tuple[pd.Timestamp, float]]) -> float:
    if not cashflows or len(cashflows) < 2:
        return 0.0
    dates = np.array([cf[0] for cf in cashflows])
    amounts = np.array([cf[1] for cf in cashflows], dtype=np.float64)
    if not (np.any(amounts < 0) and np.any(amounts > 0)):
        return 0.0
    t0 = dates[0]
    yf_arr = np.array([(d - t0).days / 365.25 for d in dates], dtype=np.float64)

    def npv(r: float) -> float:
        return 1e18 if r <= -0.9999 else float(np.sum(amounts / np.power(1.0 + r, yf_arr)))

    def npv_p(r: float) -> float:
        return -1e18 if r <= -0.9999 else float(np.sum(-yf_arr * amounts / np.power(1.0 + r, yf_arr + 1.0)))

    try:
        rt = optimize.newton(func=npv, x0=0.08, fprime=npv_p, tol=1e-7, maxiter=150)
        if np.isfinite(rt) and -0.999 < rt < 100.0 and abs(npv(rt)) < 1e-2:
            return float(rt)
    except Exception:
        pass
    try:
        rt_b = optimize.brentq(npv, -0.995, 50.0, xtol=1e-7, maxiter=300)
        if np.isfinite(rt_b):
            return float(rt_b)
    except Exception:
        pass
    return 0.0


def compute_order_fee(gross_amount: float, params: SimulationParams) -> float:
    if gross_amount <= 0.0:
        return 0.0
    if params.fee_mode == "Pourcentage (%)":
        fee = (
            max(gross_amount * params.fee_pct, params.fee_min_eur)
            if (params.fee_pct > 0 or params.fee_min_eur > 0)
            else 0.0
        )
    else:
        fee = params.fee_fixed_eur
    return min(fee, gross_amount * 0.99)


def compute_underwater_metrics(wealth_series: pd.Series) -> Tuple[pd.Series, float, int]:
    dd = (wealth_series / wealth_series.cummax()) - 1.0
    max_dd = float(dd.min()) if not dd.empty else 0.0
    is_uw = dd < -1e-9
    if not is_uw.any():
        return dd, 0.0, 0
    streaks = is_uw.groupby((~is_uw).cumsum()).sum()
    return dd, max_dd, (int(streaks.max()) if not streaks.empty else 0)


def run_asset_backtest(
    asset_label: str,
    ticker: str,
    raw_price_series: pd.Series,
    native_currency: str,
    is_bare_index: bool,
    params: SimulationParams,
    fx_eurusd_series: Optional[pd.Series] = None,
) -> AssetBacktestResult:
    prices = raw_price_series.copy().sort_index()
    eff_curr = native_currency
    if params.convert_usd_to_eur and native_currency == "USD" and fx_eurusd_series is not None:
        prices = prices / fx_eurusd_series.reindex(prices.index).ffill().bfill()
        eff_curr = "EUR"

    days_el = np.arange(len(prices), dtype=np.float64)
    adj_prices = (
        prices * np.power(1.0 - params.annual_ter, days_el / 365.25)
        if (params.annual_ter > 0 and (is_bare_index or params.apply_ter_to_etfs))
        else prices.copy()
    )
    dca_dates = generate_dca_schedule(adj_prices.index, params.dca_frequency, params.dca_day_rule)

    gross_dca = pd.Series(0.0, index=adj_prices.index, dtype=np.float64)
    fees_dca = pd.Series(0.0, index=adj_prices.index, dtype=np.float64)
    d0 = adj_prices.index[0]

    if params.initial_capital > 0:
        gross_dca.loc[d0] += params.initial_capital
        fees_dca.loc[d0] += compute_order_fee(params.initial_capital, params)
    if params.dca_amount > 0 and len(dca_dates) > 0:
        gross_dca.loc[dca_dates] += params.dca_amount
        fees_dca.loc[dca_dates] += compute_order_fee(params.dca_amount, params)
    if gross_dca.sum() == 0.0:
        gross_dca.loc[d0] = 1000.0
        fees_dca.loc[d0] = compute_order_fee(1000.0, params)

    net_dca = gross_dca - fees_dca
    shares_bought = net_dca / adj_prices
    cum_shares = shares_bought.cumsum()
    cum_inv = gross_dca.cumsum()
    cum_fees = fees_dca.cumsum()
    pru = pd.Series(np.where(cum_shares > 0, cum_inv / cum_shares, np.nan), index=adj_prices.index).ffill().bfill()
    val_dca = cum_shares * adj_prices
    infl_def = np.power(1.0 + params.inflation_rate, days_el / 365.25)
    real_val_dca = val_dca / infl_def

    tot_cap = float(cum_inv.iloc[-1])
    ls_fee = compute_order_fee(tot_cap, params)
    ls_shares = (tot_cap - ls_fee) / float(adj_prices.iloc[0])
    val_ls = adj_prices * ls_shares
    real_val_ls = val_ls / infl_def

    dd_dca, max_dd_dca, uw_dca = compute_underwater_metrics(val_dca)
    dd_ls, max_dd_ls, _ = compute_underwater_metrics(val_ls)
    dd_ass, _, uw_ass = compute_underwater_metrics(adj_prices)

    twr_dca = (((val_dca - net_dca) / val_dca.shift(1)) - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    b_rets_ass = adj_prices[adj_prices.index.weekday < 5].pct_change().dropna()
    b_rets_dca = twr_dca[twr_dca.index.weekday < 5]
    vol_dca = float(b_rets_dca.std() * np.sqrt(252.0)) if len(b_rets_dca) > 2 else 0.0
    vol_ls = float(b_rets_ass.std() * np.sqrt(252.0)) if len(b_rets_ass) > 2 else 0.0
    rf_d = (1.0 + params.risk_free_rate) ** (1.0 / 252.0) - 1.0
    dn_dca = b_rets_dca[b_rets_dca < rf_d] - rf_d
    dn_std_dca = float(np.sqrt(np.mean(np.square(dn_dca))) * np.sqrt(252.0)) if len(dn_dca) > 0 else 1e-9
    dn_ls = b_rets_ass[b_rets_ass < rf_d] - rf_d
    dn_std_ls = float(np.sqrt(np.mean(np.square(dn_ls))) * np.sqrt(252.0)) if len(dn_ls) > 0 else 1e-9

    tot_yrs = max((adj_prices.index[-1] - d0).days / 365.25, 1.0 / 365.25)
    u_cagr = float((adj_prices.iloc[-1] / adj_prices.iloc[0]) ** (1.0 / tot_yrs) - 1.0)

    fg_dca = float(val_dca.iloc[-1])
    gpnl_dca = fg_dca - tot_cap
    tax_dca = max(0.0, gpnl_dca * params.tax_rate)
    fn_dca = fg_dca - tax_dca
    npnl_dca = fn_dca - tot_cap
    fr_dca = fn_dca / float(infl_def[-1])

    fg_ls = float(val_ls.iloc[-1])
    gpnl_ls = fg_ls - tot_cap
    tax_ls = max(0.0, gpnl_ls * params.tax_rate)
    fn_ls = fg_ls - tax_ls
    npnl_ls = fn_ls - tot_cap
    fr_ls = fn_ls / float(infl_def[-1])

    cf_dca = [(ts, -float(amt)) for ts, amt in gross_dca[gross_dca > 0].items()] + [(adj_prices.index[-1], fn_dca)]
    xirr_dca = compute_xirr(cf_dca)
    cf_ls = [(d0, -tot_cap), (adj_prices.index[-1], fn_ls)]
    xirr_ls = compute_xirr(cf_ls)

    daily_df = pd.DataFrame(
        {
            "Market_Price": adj_prices,
            "Gross_Contribution": gross_dca,
            "Brokerage_Fee": fees_dca,
            "Net_Contribution": net_dca,
            "Shares_Bought": shares_bought,
            "Cum_Shares": cum_shares,
            "Cum_Invested": cum_inv,
            "Cum_Fees": cum_fees,
            "PRU": pru,
            "Portfolio_Gross_Value": val_dca,
            "Portfolio_Real_Gross_Value": real_val_dca,
            "Latent_PnL_Gross": val_dca - cum_inv,
            "Drawdown_DCA": dd_dca,
            "Drawdown_Underlying": dd_ass,
            "LumpSum_Gross_Value": val_ls,
            "LumpSum_Real_Gross_Value": real_val_ls,
            "Drawdown_LumpSum": dd_ls,
        },
        index=adj_prices.index,
    )
    daily_df.index.name = "Date"

    m_dca = BacktestMetrics(
        tot_cap,
        fg_dca,
        fn_dca,
        fr_dca,
        gpnl_dca,
        npnl_dca,
        npnl_dca / tot_cap if tot_cap > 0 else 0.0,
        tax_dca,
        float(cum_fees.iloc[-1]),
        xirr_dca,
        u_cagr,
        float((fn_dca / tot_cap) ** (1.0 / tot_yrs) - 1.0),
        vol_dca,
        (xirr_dca - params.risk_free_rate) / vol_dca if vol_dca > 1e-9 else 0.0,
        (xirr_dca - params.risk_free_rate) / dn_std_dca if dn_std_dca > 1e-9 else 0.0,
        max_dd_dca,
        uw_dca,
        float(cum_shares.iloc[-1]),
        float(pru.iloc[-1]),
        float(adj_prices.iloc[-1]),
        int((gross_dca > 0).sum()),
    )
    m_ls = BacktestMetrics(
        tot_cap,
        fg_ls,
        fn_ls,
        fr_ls,
        gpnl_ls,
        npnl_ls,
        npnl_ls / tot_cap if tot_cap > 0 else 0.0,
        tax_ls,
        ls_fee,
        xirr_ls,
        u_cagr,
        float((fn_ls / tot_cap) ** (1.0 / tot_yrs) - 1.0),
        vol_ls,
        (xirr_ls - params.risk_free_rate) / vol_ls if vol_ls > 1e-9 else 0.0,
        (xirr_ls - params.risk_free_rate) / dn_std_ls if dn_std_ls > 1e-9 else 0.0,
        max_dd_ls,
        uw_ass,
        ls_shares,
        float(adj_prices.iloc[0]) * (tot_cap / (tot_cap - ls_fee)),
        float(adj_prices.iloc[-1]),
        1,
    )
    return AssetBacktestResult(asset_label, ticker, eff_curr, daily_df, m_dca, m_ls, cf_dca, cf_ls)


def build_annual_summary_table(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.copy()
    df["Year"] = df.index.year
    records = []
    prev_val = 0.0
    for yr in sorted(df["Year"].unique()):
        sub = df[df["Year"] == yr]
        flows = float(sub["Gross_Contribution"].sum())
        cum_i = float(sub["Cum_Invested"].iloc[-1])
        end_v = float(sub["Portfolio_Gross_Value"].iloc[-1])
        lat = end_v - cum_i
        if prev_val == 0.0:
            denom = flows
        else:
            span = max((sub.index[-1] - sub.index[0]).days, 1)
            w_flows = float((sub["Gross_Contribution"] * ((sub.index[-1] - sub.index).days / span)).sum())
            denom = prev_val + max(w_flows, flows * 0.5)
        gain = end_v - prev_val - flows
        records.append(
            {
                "Année": int(yr),
                "Versé l'année (€)": round(flows, 0),
                "Cumul Versé (€)": round(cum_i, 0),
                "Valeur Fin d'Année (€)": round(end_v, 0),
                "Gain/Perte Annuel (€)": round(gain, 0),
                "Perf. Annuelle (%)": round((gain / denom * 100.0) if denom > 0 else 0.0, 1),
                "Plus-Value Latente (€)": round(lat, 0),
            }
        )
        prev_val = end_v
    return pd.DataFrame(records)


def run_monte_carlo_projection(
    res: AssetBacktestResult, params: SimulationParams, horizon_years: int, n_trajectories: int = 500
) -> pd.DataFrame:
    hist_df = res.daily_df
    b_prices = hist_df["Market_Price"][hist_df.index.weekday < 5]
    log_r = np.log(b_prices / b_prices.shift(1)).dropna()
    mu, sigma = float(log_r.mean() * 252.0), float(log_r.std() * np.sqrt(252.0))
    n_m, dt = horizon_years * 12, 1.0 / 12.0
    mult = (
        (52.0 / 12.0)
        if params.dca_frequency == "Hebdomadaire"
        else (1.0 if params.dca_frequency == "Mensuelle" else 1.0 / 3.0)
    )
    m_gross = params.dca_amount * mult
    m_net = max(0.0, m_gross - compute_order_fee(params.dca_amount, params) * mult)
    rng = np.random.default_rng(seed=42)
    growth = np.exp(
        (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_m, n_trajectories))
    )
    w_paths = np.zeros((n_m + 1, n_trajectories))
    w_paths[0, :] = float(hist_df["Portfolio_Gross_Value"].iloc[-1])
    inv_path = np.zeros(n_m + 1)
    inv_path[0] = float(hist_df["Cum_Invested"].iloc[-1])
    for m in range(1, n_m + 1):
        w_paths[m, :] = (w_paths[m - 1, :] + m_net) * growth[m - 1, :]
        inv_path[m] = inv_path[m - 1] + m_gross
    f_dates = pd.date_range(start=hist_df.index[-1], periods=n_m + 1, freq="ME")
    return pd.DataFrame(
        {
            "Date": f_dates,
            "Cum_Invested": inv_path,
            "P10": np.percentile(w_paths, 10, axis=1),
            "P50": np.percentile(w_paths, 50, axis=1),
            "P90": np.percentile(w_paths, 90, axis=1),
        }
    ).set_index("Date")


# ==============================================================================
# 4. GRAPHIQUES ÉPURÉS POUR MOBILE
# ==============================================================================


def style_mobile_chart(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.45)",
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 184, 0.12)", zeroline=False, tickformat=",.0f"),
    )
    return fig


# ==============================================================================
# 5. PANNEAU DE CONFIGURATION INTUITIF (SUR LA PAGE PRINCIPALE)
# ==============================================================================


def render_top_config() -> Tuple[List[Tuple[str, str, str, bool]], Optional[io.BytesIO], SimulationParams]:
    st.markdown("### 💎 Simulateur d'Investissement Bourse")

    with st.container(border=True):
        # 1. Choix rapide des actifs
        selected_labels = st.multiselect(
            "📌 Choisissez vos actions ou ETFs à simuler :",
            options=list(ASSET_CATALOG.keys()),
            default=[
                "🌍 MSCI World PEA (Amundi CW8)",
                "🇫🇷 Air Liquide (AI.PA - PEA)",
                "🇺🇸 S&P 500 ETF (SPY - Dividendes inclus)",
            ],
        )

        # 2. Montants d'investissement (Côte à côte même sur mobile)
        c1, c2 = st.columns(2)
        with c1:
            initial_capital = st.number_input(
                "💰 Dépôt initial (Jour 1) en €",
                min_value=0,
                max_value=10_000_000,
                value=2_000,
                step=500,
            )
        with c2:
            dca_amount = st.number_input(
                "🔄 Épargne régulière (DCA) en €",
                min_value=0,
                max_value=1_000_000,
                value=300,
                step=50,
            )

        # 3. Durée et enveloppe fiscale (Boutons rapides)
        c3, c4 = st.columns(2)
        with c3:
            horizon_choice = st.selectbox(
                "⏳ Durée du test historique",
                options=["5 dernières années", "10 dernières années", "15 dernières années", "20 dernières années", "Dates personnalisées"],
                index=2,
            )
        with c4:
            tax_choice = st.selectbox(
                "🏦 Enveloppe Fiscale",
                options=[
                    "PEA (> 5 ans : 17,2 %)",
                    "Compte-Titres (Flat Tax : 30 %)",
                    "Brut (0 % d'impôt)",
                ],
                index=0,
            )

        today = date.today()
        if horizon_choice == "Dates personnalisées":
            cd1, cd2 = st.columns(2)
            start_date = cd1.date_input("Date de début", value=today - timedelta(days=int(12 * 365.25)), min_value=date(1985, 1, 1))
            end_date = cd2.date_input("Date de fin", value=today)
        else:
            years_map = {
                "5 dernières années": 5,
                "10 dernières années": 10,
                "15 dernières années": 15,
                "20 dernières années": 20,
            }
            yrs = years_map[horizon_choice]
            start_date = today - timedelta(days=int(yrs * 365.25))
            end_date = today

        # 4. Options avancées repliées pour garder une interface ultra propre
        with st.expander("⚙️ Plus d'options ( Autres actions Yahoo, Frais, Fréquence, Inflation, CSV )"):
            custom_tickers_raw = st.text_input(
                "Ajouter d'autres codes Yahoo Finance (séparés par une virgule)",
                value="",
                placeholder="Ex: AAPL, GOOGL, TTE.PA, BNP.PA",
            )
            ca1, ca2 = st.columns(2)
            with ca1:
                dca_frequency = st.selectbox("Fréquence des versements", ["Mensuelle", "Hebdomadaire", "Trimestrielle"])
                fee_pct_input = st.number_input("Frais de courtage par achat (%)", 0.0, 5.0, 0.10, 0.05)
                inflation_input = st.number_input("Inflation annuelle moyenne (%)", 0.0, 15.0, 2.0, 0.2)
            with ca2:
                dca_day_rule = st.selectbox("Jour du versement", ["1er du mois", "15 du mois", "Dernier jour ouvré"])
                ter_input = st.number_input("Frais de gestion annuels indiciels (%)", 0.0, 3.0, 0.20, 0.05)
                convert_usd_to_eur = st.toggle("Convertir les actifs $ en € (EURUSD)", value=True)

            uploaded_csv = st.file_uploader("Importer un fichier CSV externe (Optionnel)", type=["csv"])

    assets_to_fetch: List[Tuple[str, str, str, bool]] = []
    for lbl in selected_labels:
        tck, curr, is_bare = ASSET_CATALOG[lbl]
        assets_to_fetch.append((lbl, tck, curr, is_bare))

    if custom_tickers_raw.strip():
        for raw_t in custom_tickers_raw.split(","):
            clean_t = raw_t.strip().upper()
            if clean_t and not any(clean_t == x[1] for x in assets_to_fetch):
                assets_to_fetch.append((f"Action ({clean_t})", clean_t, "USD", clean_t.startswith("^")))

    tax_map = {
        "PEA (> 5 ans : 17,2 %)": 0.172,
        "Compte-Titres (Flat Tax : 30 %)": 0.30,
        "Brut (0 % d'impôt)": 0.0,
    }

    params = SimulationParams(
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(initial_capital),
        dca_amount=float(dca_amount),
        dca_frequency=dca_frequency,
        dca_day_rule=dca_day_rule,
        fee_mode="Pourcentage (%)",
        fee_pct=float(fee_pct_input) / 100.0,
        fee_min_eur=0.0,
        fee_fixed_eur=0.0,
        annual_ter=float(ter_input) / 100.0,
        apply_ter_to_etfs=False,
        tax_regime_name=tax_choice,
        tax_rate=tax_map[tax_choice],
        inflation_rate=float(inflation_input) / 100.0,
        risk_free_rate=0.025,
        convert_usd_to_eur=convert_usd_to_eur,
    )

    return assets_to_fetch, uploaded_csv, params


# ==============================================================================
# 6. AFFICHAGE PRINCIPAL INTUITIF
# ==============================================================================


def main() -> None:
    assets_to_fetch, uploaded_csv, params = render_top_config()

    if not assets_to_fetch and uploaded_csv is None:
        st.info("👆 Sélectionnez au moins une action ou un ETF ci-dessus pour voir les résultats.")
        st.stop()

    fx_series: Optional[pd.Series] = None
    if params.convert_usd_to_eur:
        fx_series = fetch_eurusd_fx_series(params.start_date, params.end_date)

    results: List[AssetBacktestResult] = []
    with st.spinner("Chargement des cours de bourse en temps réel..."):
        if uploaded_csv is not None:
            cs, cc, ce = parse_custom_csv(uploaded_csv, params.start_date, params.end_date)
            if ce:
                st.error(ce)
            elif cs is not None:
                results.append(run_asset_backtest(f"CSV ({uploaded_csv.name})", "CSV", cs, cc, False, params, fx_series))

        for lbl, tck, d_curr, is_b in assets_to_fetch:
            sp, det_c, err = fetch_yahoo_series(tck, params.start_date, params.end_date)
            if err or sp is None:
                st.warning(err or f"Données indisponibles pour {tck}")
                continue
            results.append(
                run_asset_backtest(lbl, tck, sp, "EUR" if d_curr == "EUR" else det_c, is_b, params, fx_series)
            )

    if not results:
        st.error("Aucune donnée disponible sur cette période.")
        st.stop()

    # Sélecteur d'actif principal sous forme de pilules cliquables ou menu déroulant
    if len(results) > 1:
        selected_label = st.selectbox(
            "🔍 Afficher le zoom détaillé pour :",
            options=[r.asset_label for r in results],
            index=0,
        )
        primary_res = next(r for r in results if r.asset_label == selected_label)
    else:
        primary_res = results[0]

    m_dca = primary_res.dca_metrics
    m_ls = primary_res.lump_sum_metrics
    df = primary_res.daily_df

    # --- 1. HERO CARD (LE RÉSULTAT EN UN COUP D'ŒIL) ---
    badge_class = "hero-badge-pos" if m_dca.net_pnl >= 0 else "hero-badge-neg"
    multiplier = m_dca.final_net_value / m_dca.total_invested if m_dca.total_invested > 0 else 1.0

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-label">Capital Final Net en poche ({primary_res.asset_label})</div>
            <div class="hero-value">{m_dca.final_net_value:,.0f} €</div>
            <div class="{badge_class}">
                {m_dca.net_pnl:+,.0f} € de gain net ({m_dca.net_return_pct * 100:+.1f} % | x{multiplier:.2f})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- 2. GRILLE 2x2 ULTRA LISIBLE SUR MOBILE ---
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-box">
                <div class="kpi-title">💰 Total sorti de votre poche</div>
                <div class="kpi-num">{m_dca.total_invested:,.0f} €</div>
                <div class="kpi-sub">{m_dca.num_orders} versements effectués</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">🚀 Rendement Annuel Réel (TRI)</div>
                <div class="kpi-num">{m_dca.xirr * 100:.2f} % / an</div>
                <div class="kpi-sub">Net de frais et d'impôts</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">🛒 Pouvoir d'achat réel (Inflation déduite)</div>
                <div class="kpi-num">{m_dca.final_real_net_value:,.0f} €</div>
                <div class="kpi-sub">En euros constants d'aujourd'hui</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">📉 Pire baisse traversée (Krach)</div>
                <div class="kpi-num" style="color:#F87171;">{m_dca.max_drawdown * 100:.1f} %</div>
                <div class="kpi-sub">Depuis le plus haut atteint</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- 3. PHRASE D'EXPLICATION EN FRANÇAIS CLAIR ---
    start_str = df.index[0].strftime("%m/%Y")
    end_str = df.index[-1].strftime("%m/%Y")
    st.markdown(
        f"""
        <div class="insight-box">
            💡 <b>En clair :</b> En démarrant avec <b>{params.initial_capital:,.0f} €</b> en {start_str} puis en ajoutant
            <b>{params.dca_amount:,.0f} €</b> ({params.dca_frequency.lower()}), vous avez versé au total <b>{m_dca.total_invested:,.0f} €</b>.
            Les intérêts composés ont généré <b>{m_dca.gross_pnl:+,.0f} €</b> de gains bruts (soit <b>{m_dca.net_pnl:+,.0f} €</b> nets après fiscalité {params.tax_regime_name.split('(')[0]}).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- 4. ONGLETS SIMPLIFIÉS ---
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Évolution",
            "⚔️ Comparateur",
            "🔮 Futur (+10/20 ans)",
            "📅 Bilans & Détails",
        ]
    )

    # ONGLET 1 : COURBE PRINCIPALE
    with tab1:
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Cum_Invested"],
                name="Argent versé",
                line=dict(color="#94A3B8", width=2, shape="hv"),
                fill="tozeroy",
                fillcolor="rgba(148, 163, 184, 0.15)",
                hovertemplate="%{y:,.0f} €<extra>Argent versé</extra>",
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Portfolio_Gross_Value"],
                name="Valeur du portefeuille",
                line=dict(color="#38BDF8", width=3),
                fill="tonexty",
                fillcolor="rgba(56, 189, 248, 0.15)",
                hovertemplate="%{y:,.0f} €<extra>Portefeuille</extra>",
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Portfolio_Real_Gross_Value"],
                name="Valeur hors inflation",
                line=dict(color="#F59E0B", width=2, dash="dot"),
                hovertemplate="%{y:,.0f} €<extra>Hors inflation</extra>",
            )
        )
        st.plotly_chart(style_mobile_chart(fig1, height=370), use_container_width=True)

        with st.expander("🔍 Voir l'évolution de votre Prix d'Achat Moyen (PRU) et des Krachs"):
            st.caption("Lorsque la courbe verte (Prix du marché) passe au-dessus de la courbe orange (votre prix d'achat moyen), vous êtes en plus-value.")
            fig_pru = go.Figure()
            fig_pru.add_trace(go.Scatter(x=df.index, y=df["Market_Price"], name="Prix de l'action/ETF", line=dict(color="#10B981", width=2.2)))
            fig_pru.add_trace(go.Scatter(x=df.index, y=df["PRU"], name="Votre Prix Moyen (PRU)", line=dict(color="#F59E0B", width=2.5)))
            st.plotly_chart(style_mobile_chart(fig_pru, height=300), use_container_width=True)

            st.caption("Graphique des baisses (Drawdown) : montre de combien % votre portefeuille a reculé lors des crises avant de rebondir.")
            fig_dd = go.Figure()
            fig_dd.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Drawdown_DCA"] * 100,
                    name="Baisse depuis le record (%)",
                    line=dict(color="#F43F5E", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(244, 63, 94, 0.2)",
                )
            )
            st.plotly_chart(style_mobile_chart(fig_dd, height=260), use_container_width=True)

    # ONGLET 2 : COMPARATEUR MULTI-ACTIFS & DCA VS LUMP SUM
    with tab2:
        st.markdown("##### 🏆 Classement des actifs sélectionnés (pour le même effort d'épargne)")
        fig_comp = go.Figure()
        fig_comp.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Cum_Invested"],
                name="Total versé",
                line=dict(color="#64748B", width=2, dash="dash", shape="hv"),
            )
        )
        for idx, r in enumerate(results):
            fig_comp.add_trace(
                go.Scatter(
                    x=r.daily_df.index,
                    y=r.daily_df["Portfolio_Gross_Value"],
                    name=f"{r.asset_label.split('(')[0].strip()}",
                    line=dict(color=COLOR_PALETTE[idx % len(COLOR_PALETTE)], width=2.6),
                )
            )
        st.plotly_chart(style_mobile_chart(fig_comp, height=370), use_container_width=True)

        comp_table = [
            {
                "Actif": r.asset_label,
                "Argent Versé": f"{r.dca_metrics.total_invested:,.0f} €",
                "Capital Final Net": f"{r.dca_metrics.final_net_value:,.0f} €",
                "Gain Net": f"{r.dca_metrics.net_pnl:+,.0f} €",
                "Rendement / an": f"{r.dca_metrics.xirr * 100:.2f} %",
                "Pire Baisse": f"{r.dca_metrics.max_drawdown * 100:.1f} %",
            }
            for r in sorted(results, key=lambda x: x.dca_metrics.final_net_value, reverse=True)
        ]
        st.dataframe(pd.DataFrame(comp_table), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown(f"##### 🥊 Et si vous aviez tout investi le 1er jour ? (DCA vs Lump Sum sur {primary_res.asset_label})")
        st.caption(
            f"Compare votre épargne progressive (DCA) avec le fait d'investir la totalité des **{m_dca.total_invested:,.0f} €** dès le Jour 1."
        )
        col_ls1, col_ls2 = st.columns(2)
        col_ls1.metric("Avec Épargne Progressive (DCA)", f"{m_dca.final_net_value:,.0f} €", f"Pire baisse : {m_dca.max_drawdown * 100:.1f} %", delta_color="off")
        col_ls2.metric("En mettant tout au Jour 1 (Lump Sum)", f"{m_ls.final_net_value:,.0f} €", f"Pire baisse : {m_ls.max_drawdown * 100:.1f} %", delta_color="off")

        fig_ls = go.Figure()
        fig_ls.add_trace(go.Scatter(x=df.index, y=df["Portfolio_Gross_Value"], name="Épargne progressive (DCA)", line=dict(color="#38BDF8", width=2.8)))
        fig_ls.add_trace(go.Scatter(x=df.index, y=df["LumpSum_Gross_Value"], name="Tout au Jour 1 (Lump Sum)", line=dict(color="#A855F7", width=2.5)))
        st.plotly_chart(style_mobile_chart(fig_ls, height=320), use_container_width=True)

    # ONGLET 3 : PROJECTION DANS LE FUTUR (MONTE CARLO)
    with tab3:
        st.markdown(f"##### 🔮 Combien aurez-vous demain en continuant sur {primary_res.asset_label} ?")
        st.caption("Simulation de 500 scénarios futurs basés sur le rythme historique de cet actif en poursuivant votre épargne.")
        hz = st.slider("Choisissez le nombre d'années supplémentaires :", min_value=3, max_value=30, value=10)
        mc_df = run_monte_carlo_projection(primary_res, params, hz)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Scénario Prudent", f"{mc_df['P10'].iloc[-1]:,.0f} €", "Marché difficile", delta_color="off")
        mc2.metric("Scénario Médian", f"{mc_df['P50'].iloc[-1]:,.0f} €", f"Dans +{hz} ans", delta_color="normal")
        mc3.metric("Scénario Optimiste", f"{mc_df['P90'].iloc[-1]:,.0f} €", "Marché porteur", delta_color="normal")

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(x=mc_df.index, y=mc_df["P90"], name="Scénario Optimiste", line=dict(color="#10B981", width=1.5)))
        fig_mc.add_trace(
            go.Scatter(
                x=mc_df.index,
                y=mc_df["P10"],
                name="Scénario Prudent",
                line=dict(color="#F43F5E", width=1.5),
                fill="tonexty",
                fillcolor="rgba(56, 189, 248, 0.12)",
            )
        )
        fig_mc.add_trace(go.Scatter(x=mc_df.index, y=mc_df["P50"], name="Trajectoire Centrale", line=dict(color="#38BDF8", width=3)))
        fig_mc.add_trace(go.Scatter(x=mc_df.index, y=mc_df["Cum_Invested"], name="Total versé", line=dict(color="#94A3B8", width=2, dash="dash")))
        st.plotly_chart(style_mobile_chart(fig_mc, height=350), use_container_width=True)

    # ONGLET 4 : BILAN ANNÉE PAR ANNÉE & STATISTIQUES EXPERTES
    with tab4:
        st.markdown("##### 📅 Récapitulatif Année par Année")
        ann_df = build_annual_summary_table(df)
        st.dataframe(ann_df, use_container_width=True, hide_index=True)

        with st.expander("🤓 Statistiques Quantitatives Expertes (Sharpe, Sortino, Volatilité)"):
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Volatilité Annuelle", f"{m_dca.annualized_volatility * 100:.2f} %")
            s2.metric("Ratio de Sharpe", f"{m_dca.sharpe_ratio:.2f}")
            s3.metric("Ratio de Sortino", f"{m_dca.sortino_ratio:.2f}")
            s4.metric("Frais de courtage cumulés", f"{m_dca.total_brokerage_fees:,.0f} €")

        ce1, ce2 = st.columns(2)
        ce1.download_button(
            "📥 Télécharger l'historique complet (CSV)",
            df.to_csv(index=True).encode("utf-8"),
            f"portefeuille_{primary_res.ticker}.csv",
            "text/csv",
            use_container_width=True,
        )
        ce2.download_button(
            "📥 Télécharger le bilan annuel (CSV)",
            ann_df.to_csv(index=False).encode("utf-8"),
            f"bilan_annuel_{primary_res.ticker}.csv",
            "text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
