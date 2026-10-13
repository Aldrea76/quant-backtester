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
            tickers=ticker,
            start=dl_start,
            end=dl_end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if raw is None or raw.empty:
            return None, "USD", f"Aucune donnée renvoyée par Yahoo Finance pour '{ticker}'."

        series = _extract_price_series(raw, ticker).dropna()
        if series.empty:
            return None, "USD", f"Série de prix vide après nettoyage pour '{ticker}'."

        series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
        series = series[~series.index.duplicated(keep="last")].sort_index()

        currency = "USD"
        upper_t = ticker.upper()
        if upper_t.endswith((".PA", ".AS", ".DE", ".MI", ".MC", ".BR")) or upper_t in [
            "^STOXX50E",
            "^FCHI",
            "^GDAXI",
        ]:
            currency = "EUR"
        else:
            try:
                t_obj = yf.Ticker(ticker)
                fast_info = getattr(t_obj, "fast_info", None)
                if fast_info and "currency" in fast_info and fast_info["currency"]:
                    currency = str(fast_info["currency"]).upper()
            except Exception:
                pass

        effective_start = max(pd.Timestamp(start_date), series.index.min())
        effective_end = min(pd.Timestamp(end_date), series.index.max())

        if effective_start >= effective_end:
            return (
                None,
                currency,
                f"La plage de dates demandée ne contient pas de données cotées pour '{ticker}'.",
            )

        full_calendar = pd.date_range(start=effective_start, end=effective_end, freq="D")
        series = series.reindex(full_calendar.union(series.index)).sort_index()
        series = series.ffill().bfill().reindex(full_calendar)

        return series, currency, None

    except Exception as exc:
        return None, "USD", f"Erreur réseau/API sur '{ticker}' : {str(exc)}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_eurusd_fx_series(start_date: date, end_date: date) -> pd.Series:
    fx_series, _, err = fetch_yahoo_series("EURUSD=X", start_date, end_date)
    if fx_series is None or fx_series.empty or err is not None:
        idx = pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq="D")
        return pd.Series(1.10, index=idx, name="EURUSD")
    return fx_series


def parse_custom_csv(
    uploaded_file: io.BytesIO, start_date: date, end_date: date
) -> Tuple[Optional[pd.Series], str, Optional[str]]:
    try:
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        first_line = content.splitlines()[0] if content else ""
        sep = ";" if ";" in first_line else ("\t" if "\t" in first_line else ",")

        df = pd.read_csv(io.StringIO(content), sep=sep)
        if df.empty or len(df.columns) < 2:
            return None, "EUR", "Le fichier CSV doit contenir au moins une colonne Date et une colonne Prix."

        date_col: Optional[str] = None
        for col in df.columns:
            if any(k in str(col).lower() for k in ["date", "time", "jour", "timestamp", "day"]):
                date_col = col
                break
        if date_col is None:
            date_col = df.columns[0]

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        df = df.dropna(subset=[date_col]).sort_values(by=date_col)

        price_col: Optional[str] = None
        candidate_keywords = ["adj close", "close", "clôture", "cloture", "prix", "price", "valeur", "nav", "vl"]
        for kw in candidate_keywords:
            for col in df.columns:
                if col != date_col and kw in str(col).lower():
                    price_col = col
                    break
            if price_col is not None:
                break

        if price_col is None:
            remaining_cols = [c for c in df.columns if c != date_col]
            price_col = remaining_cols[0]

        if df[price_col].dtype == object:
            cleaned_vals = (
                df[price_col]
                .astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace("\u00a0", "", regex=False)
                .str.replace("€", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[price_col] = pd.to_numeric(cleaned_vals, errors="coerce")
        else:
            df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

        df = df.dropna(subset=[price_col])
        df = df[df[price_col] > 0]

        if df.empty:
            return None, "EUR", "Aucune valeur numérique positive valide trouvée dans la colonne de prix."

        series = pd.Series(df[price_col].values, index=df[date_col])
        series = series[~series.index.duplicated(keep="last")].sort_index()

        eff_start = max(pd.Timestamp(start_date), series.index.min())
        eff_end = min(pd.Timestamp(end_date), series.index.max())
        if eff_start >= eff_end:
            return (
                None,
                "EUR",
                f"Les dates du CSV ({series.index.min().date()} au {series.index.max().date()}) "
                f"ne recoupent pas la période sélectionnée.",
            )

        cal = pd.date_range(start=eff_start, end=eff_end, freq="D")
        series = series.reindex(cal.union(series.index)).sort_index().ffill().bfill().reindex(cal)
        return series, "EUR", None

    except Exception as exc:
        return None, "EUR", f"Erreur lors de la lecture du CSV : {str(exc)}"


# ==============================================================================
# 4. MOTEUR QUANTITATIF (CALENDRIER DCA, XIRR, BACKTEST VECTORISÉ)
# ==============================================================================


def generate_dca_schedule(
    calendar_index: pd.DatetimeIndex, frequency: str, day_rule: str
) -> pd.DatetimeIndex:
    if calendar_index.empty:
        return pd.DatetimeIndex([])

    start_ts = calendar_index.min()
    end_ts = calendar_index.max()

    if frequency == "Hebdomadaire":
        target_dates = calendar_index[calendar_index.weekday == 0]
        return target_dates[target_dates > start_ts]

    months_df = pd.DataFrame({"date": calendar_index})
    months_df["year"] = months_df["date"].dt.year
    months_df["month"] = months_df["date"].dt.month
    months_df["is_bday"] = months_df["date"].dt.weekday < 5

    ym_groups = months_df.groupby(["year", "month"], as_index=False)
    selected_dates: List[pd.Timestamp] = []

    step = 1 if frequency == "Mensuelle" else 3
    group_keys = list(ym_groups.groups.keys())

    for idx_g, (yr, mo) in enumerate(group_keys):
        if idx_g % step != 0:
            continue

        sub = ym_groups.get_group((yr, mo))
        bdays = sub[sub["is_bday"]]["date"]
        if bdays.empty:
            bdays = sub["date"]

        if day_rule == "1er du mois":
            chosen = bdays.iloc[0]
        elif day_rule == "15 du mois":
            on_or_after_15 = bdays[bdays.dt.day >= 15]
            chosen = on_or_after_15.iloc[0] if not on_or_after_15.empty else bdays.iloc[-1]
        else:
            chosen = bdays.iloc[-1]

        if start_ts < chosen <= end_ts:
            selected_dates.append(chosen)

    return pd.DatetimeIndex(sorted(set(selected_dates)))


def compute_xirr(cashflows: List[Tuple[pd.Timestamp, float]]) -> float:
    if not cashflows or len(cashflows) < 2:
        return 0.0

    dates = np.array([cf[0] for cf in cashflows])
    amounts = np.array([cf[1] for cf in cashflows], dtype=np.float64)

    if not (np.any(amounts < 0) and np.any(amounts > 0)):
        return 0.0

    t0 = dates[0]
    years_frac = np.array([(d - t0).days / 365.25 for d in dates], dtype=np.float64)

    def npv(rate: float) -> float:
        if rate <= -0.9999:
            return 1e18
        return float(np.sum(amounts / np.power(1.0 + rate, years_frac)))

    def npv_prime(rate: float) -> float:
        if rate <= -0.9999:
            return -1e18
        return float(np.sum(-years_frac * amounts / np.power(1.0 + rate, years_frac + 1.0)))

    try:
        root = optimize.newton(func=npv, x0=0.08, fprime=npv_prime, tol=1e-7, maxiter=150)
        if np.isfinite(root) and -0.999 < root < 100.0 and abs(npv(root)) < 1e-2:
            return float(root)
    except Exception:
        pass

    try:
        root_brent = optimize.brentq(npv, -0.995, 50.0, xtol=1e-7, maxiter=300)
        if np.isfinite(root_brent):
            return float(root_brent)
    except Exception:
        pass

    return 0.0


def compute_order_fee(gross_amount: float, params: SimulationParams) -> float:
    if gross_amount <= 0.0:
        return 0.0
    if params.fee_mode == "Pourcentage (%)":
        computed = gross_amount * params.fee_pct
        fee = max(computed, params.fee_min_eur) if params.fee_pct > 0 or params.fee_min_eur > 0 else 0.0
    else:
        fee = params.fee_fixed_eur
    return min(fee, gross_amount * 0.99)


def compute_underwater_metrics(wealth_series: pd.Series) -> Tuple[pd.Series, float, int]:
    running_max = wealth_series.cummax()
    drawdown_series = (wealth_series / running_max) - 1.0
    max_dd = float(drawdown_series.min()) if not drawdown_series.empty else 0.0

    is_underwater = drawdown_series < -1e-9
    if not is_underwater.any():
        return drawdown_series, 0.0, 0

    ath_reached_groups = (~is_underwater).cumsum()
    underwater_streaks = is_underwater.groupby(ath_reached_groups).sum()
    max_underwater_days = int(underwater_streaks.max()) if not underwater_streaks.empty else 0

    return drawdown_series, max_dd, max_underwater_days


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
    effective_currency = native_currency

    if params.convert_usd_to_eur and native_currency == "USD" and fx_eurusd_series is not None:
        fx_aligned = fx_eurusd_series.reindex(prices.index).ffill().bfill()
        prices = prices / fx_aligned
        effective_currency = "EUR"

    should_deduct_ter = params.annual_ter > 0 and (is_bare_index or params.apply_ter_to_etfs)
    n_days = len(prices)
    days_elapsed = np.arange(n_days, dtype=np.float64)

    if should_deduct_ter:
        daily_ter_drag = np.power(1.0 - params.annual_ter, days_elapsed / 365.25)
        adj_prices = prices * daily_ter_drag
    else:
        adj_prices = prices.copy()

    dca_dates = generate_dca_schedule(adj_prices.index, params.dca_frequency, params.dca_day_rule)

    gross_contrib_dca = pd.Series(0.0, index=adj_prices.index, dtype=np.float64)
    fees_dca = pd.Series(0.0, index=adj_prices.index, dtype=np.float64)

    first_day = adj_prices.index[0]
    if params.initial_capital > 0:
        gross_contrib_dca.loc[first_day] += params.initial_capital
        fees_dca.loc[first_day] += compute_order_fee(params.initial_capital, params)

    if params.dca_amount > 0 and len(dca_dates) > 0:
        single_dca_fee = compute_order_fee(params.dca_amount, params)
        gross_contrib_dca.loc[dca_dates] += params.dca_amount
        fees_dca.loc[dca_dates] += single_dca_fee

    if gross_contrib_dca.sum() == 0.0:
        gross_contrib_dca.loc[first_day] = 1000.0
        fees_dca.loc[first_day] = compute_order_fee(1000.0, params)

    net_contrib_dca = gross_contrib_dca - fees_dca

    shares_bought_dca = net_contrib_dca / adj_prices
    cum_shares_dca = shares_bought_dca.cumsum()
    cum_gross_invested_dca = gross_contrib_dca.cumsum()
    cum_fees_dca = fees_dca.cumsum()

    pru_series = np.where(cum_shares_dca > 0, cum_gross_invested_dca / cum_shares_dca, np.nan)
    pru_series = pd.Series(pru_series, index=adj_prices.index).ffill().bfill()

    portfolio_val_dca = cum_shares_dca * adj_prices

    inflation_deflator = np.power(1.0 + params.inflation_rate, days_elapsed / 365.25)
    portfolio_real_val_dca = portfolio_val_dca / inflation_deflator

    total_capital_mobilized = float(cum_gross_invested_dca.iloc[-1])
    ls_fee_day1 = compute_order_fee(total_capital_mobilized, params)
    ls_net_invested_day1 = total_capital_mobilized - ls_fee_day1
    ls_shares = ls_net_invested_day1 / float(adj_prices.iloc[0])
    portfolio_val_ls = adj_prices * ls_shares
    portfolio_real_val_ls = portfolio_val_ls / inflation_deflator

    dd_dca, max_dd_dca, max_uw_days_dca = compute_underwater_metrics(portfolio_val_dca)
    dd_ls, max_dd_ls, max_uw_days_ls = compute_underwater_metrics(portfolio_val_ls)
    dd_asset, max_dd_asset, max_uw_days_asset = compute_underwater_metrics(adj_prices)

    prev_val_dca = portfolio_val_dca.shift(1)
    twr_daily_dca = ((portfolio_val_dca - net_contrib_dca) / prev_val_dca) - 1.0
    twr_daily_dca = twr_daily_dca.replace([np.inf, -np.inf], np.nan).dropna()

    bday_mask = adj_prices.index.weekday < 5
    asset_bday_returns = adj_prices[bday_mask].pct_change().dropna()
    dca_bday_returns = twr_daily_dca[twr_daily_dca.index.weekday < 5]

    ann_factor = 252.0
    vol_dca = float(dca_bday_returns.std() * np.sqrt(ann_factor)) if len(dca_bday_returns) > 2 else 0.0
    vol_ls = float(asset_bday_returns.std() * np.sqrt(ann_factor)) if len(asset_bday_returns) > 2 else 0.0

    rf_daily = (1.0 + params.risk_free_rate) ** (1.0 / ann_factor) - 1.0

    downside_dca = dca_bday_returns[dca_bday_returns < rf_daily] - rf_daily
    downside_std_dca = (
        float(np.sqrt(np.mean(np.square(downside_dca))) * np.sqrt(ann_factor))
        if len(downside_dca) > 0
        else 1e-9
    )

    downside_ls = asset_bday_returns[asset_bday_returns < rf_daily] - rf_daily
    downside_std_ls = (
        float(np.sqrt(np.mean(np.square(downside_ls))) * np.sqrt(ann_factor))
        if len(downside_ls) > 0
        else 1e-9
    )

    total_years = max((adj_prices.index[-1] - adj_prices.index[0]).days / 365.25, 1.0 / 365.25)
    underlying_cagr = float((adj_prices.iloc[-1] / adj_prices.iloc[0]) ** (1.0 / total_years) - 1.0)

    final_gross_dca = float(portfolio_val_dca.iloc[-1])
    gross_pnl_dca = final_gross_dca - total_capital_mobilized
    tax_dca = max(0.0, gross_pnl_dca * params.tax_rate)
    final_net_dca = final_gross_dca - tax_dca
    net_pnl_dca = final_net_dca - total_capital_mobilized
    final_real_net_dca = final_net_dca / float(inflation_deflator[-1])

    final_gross_ls = float(portfolio_val_ls.iloc[-1])
    gross_pnl_ls = final_gross_ls - total_capital_mobilized
    tax_ls = max(0.0, gross_pnl_ls * params.tax_rate)
    final_net_ls = final_gross_ls - tax_ls
    net_pnl_ls = final_net_ls - total_capital_mobilized
    final_real_net_ls = final_net_ls / float(inflation_deflator[-1])

    contrib_events = gross_contrib_dca[gross_contrib_dca > 0]
    cashflows_dca: List[Tuple[pd.Timestamp, float]] = [
        (ts, -float(amt)) for ts, amt in contrib_events.items()
    ]
    cashflows_dca.append((adj_prices.index[-1], final_net_dca))
    xirr_dca = compute_xirr(cashflows_dca)

    cashflows_ls: List[Tuple[pd.Timestamp, float]] = [
        (adj_prices.index[0], -total_capital_mobilized),
        (adj_prices.index[-1], final_net_ls),
    ]
    xirr_ls = compute_xirr(cashflows_ls)

    sharpe_dca = (xirr_dca - params.risk_free_rate) / vol_dca if vol_dca > 1e-9 else 0.0
    sortino_dca = (xirr_dca - params.risk_free_rate) / downside_std_dca if downside_std_dca > 1e-9 else 0.0

    sharpe_ls = (xirr_ls - params.risk_free_rate) / vol_ls if vol_ls > 1e-9 else 0.0
    sortino_ls = (xirr_ls - params.risk_free_rate) / downside_std_ls if downside_std_ls > 1e-9 else 0.0

    portfolio_cagr_dca = float((final_net_dca / total_capital_mobilized) ** (1.0 / total_years) - 1.0)
    portfolio_cagr_ls = float((final_net_ls / total_capital_mobilized) ** (1.0 / total_years) - 1.0)

    daily_df = pd.DataFrame(
        {
            "Market_Price": adj_prices,
            "Gross_Contribution": gross_contrib_dca,
            "Brokerage_Fee": fees_dca,
            "Net_Contribution": net_contrib_dca,
            "Shares_Bought": shares_bought_dca,
            "Cum_Shares": cum_shares_dca,
            "Cum_Invested": cum_gross_invested_dca,
            "Cum_Fees": cum_fees_dca,
            "PRU": pru_series,
            "Portfolio_Gross_Value": portfolio_val_dca,
            "Portfolio_Real_Gross_Value": portfolio_real_val_dca,
            "Latent_PnL_Gross": portfolio_val_dca - cum_gross_invested_dca,
            "Drawdown_DCA": dd_dca,
            "Drawdown_Underlying": dd_asset,
            "LumpSum_Gross_Value": portfolio_val_ls,
            "LumpSum_Real_Gross_Value": portfolio_real_val_ls,
            "Drawdown_LumpSum": dd_ls,
        },
        index=adj_prices.index,
    )
    daily_df.index.name = "Date"

    dca_metrics = BacktestMetrics(
        total_invested=total_capital_mobilized,
        final_gross_value=final_gross_dca,
        final_net_value=final_net_dca,
        final_real_net_value=final_real_net_dca,
        gross_pnl=gross_pnl_dca,
        net_pnl=net_pnl_dca,
        net_return_pct=(net_pnl_dca / total_capital_mobilized) if total_capital_mobilized > 0 else 0.0,
        tax_paid=tax_dca,
        total_brokerage_fees=float(cum_fees_dca.iloc[-1]),
        xirr=xirr_dca,
        underlying_cagr=underlying_cagr,
        portfolio_cagr=portfolio_cagr_dca,
        annualized_volatility=vol_dca,
        sharpe_ratio=sharpe_dca,
        sortino_ratio=sortino_dca,
        max_drawdown=max_dd_dca,
        max_underwater_days=max_uw_days_dca,
        total_shares=float(cum_shares_dca.iloc[-1]),
        final_pru=float(pru_series.iloc[-1]),
        final_market_price=float(adj_prices.iloc[-1]),
        num_orders=int((gross_contrib_dca > 0).sum()),
    )

    ls_metrics = BacktestMetrics(
        total_invested=total_capital_mobilized,
        final_gross_value=final_gross_ls,
        final_net_value=final_net_ls,
        final_real_net_value=final_real_net_ls,
        gross_pnl=gross_pnl_ls,
        net_pnl=net_pnl_ls,
        net_return_pct=(net_pnl_ls / total_capital_mobilized) if total_capital_mobilized > 0 else 0.0,
        tax_paid=tax_ls,
        total_brokerage_fees=ls_fee_day1,
        xirr=xirr_ls,
        underlying_cagr=underlying_cagr,
        portfolio_cagr=portfolio_cagr_ls,
        annualized_volatility=vol_ls,
        sharpe_ratio=sharpe_ls,
        sortino_ratio=sortino_ls,
        max_drawdown=max_dd_ls,
        max_underwater_days=max_uw_days_asset,
        total_shares=ls_shares,
        final_pru=float(adj_prices.iloc[0]) * (total_capital_mobilized / ls_net_invested_day1),
        final_market_price=float(adj_prices.iloc[-1]),
        num_orders=1,
    )

    return AssetBacktestResult(
        asset_label=asset_label,
        ticker=ticker,
        currency=effective_currency,
        daily_df=daily_df,
        dca_metrics=dca_metrics,
        lump_sum_metrics=ls_metrics,
        cashflows_dca=cashflows_dca,
        cashflows_ls=cashflows_ls,
    )


def build_annual_summary_table(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.copy()
    df["Year"] = df.index.year
    years = sorted(df["Year"].unique())

    records: List[Dict[str, float]] = []
    prev_end_val = 0.0

    for yr in years:
        sub = df[df["Year"] == yr]
        flows_yr = float(sub["Gross_Contribution"].sum())
        cum_invested_end = float(sub["Cum_Invested"].iloc[-1])
        end_val = float(sub["Portfolio_Gross_Value"].iloc[-1])
        latent_pnl = end_val - cum_invested_end
        latent_pnl_pct = (latent_pnl / cum_invested_end * 100.0) if cum_invested_end > 0 else 0.0

        if prev_end_val == 0.0:
            denom = flows_yr
        else:
            t_start = sub.index[0]
            t_end = sub.index[-1]
            span_days = max((t_end - t_start).days, 1)
            weights = (t_end - sub.index).days / span_days
            weighted_flows = float((sub["Gross_Contribution"] * weights).sum())
            denom = prev_end_val + max(weighted_flows, flows_yr * 0.5)

        gain_yr = end_val - prev_end_val - flows_yr
        perf_yr_pct = (gain_yr / denom * 100.0) if denom > 0 else 0.0

        records.append(
            {
                "Année": int(yr),
                "Versements de l'année (€)": round(flows_yr, 2),
                "Cumul Versé (€)": round(cum_invested_end, 2),
                "Valeur Fin d'Année (€)": round(end_val, 2),
                "Gain / Perte de l'année (€)": round(gain_yr, 2),
                "Performance Annuelle (%)": round(perf_yr_pct, 2),
                "Plus/Moins-Value Latente (€)": round(latent_pnl, 2),
                "Plus/Moins-Value Latente (%)": round(latent_pnl_pct, 2),
                "PRU Fin d'Année (€)": round(float(sub["PRU"].iloc[-1]), 2),
                "Prix Marché Fin d'Année (€)": round(float(sub["Market_Price"].iloc[-1]), 2),
            }
        )
        prev_end_val = end_val

    return pd.DataFrame(records)


def run_monte_carlo_projection(
    res: AssetBacktestResult,
    params: SimulationParams,
    horizon_years: int,
    n_trajectories: int = 500,
) -> pd.DataFrame:
    hist_df = res.daily_df
    bday_prices = hist_df["Market_Price"][hist_df.index.weekday < 5]
    log_rets = np.log(bday_prices / bday_prices.shift(1)).dropna()

    mu_annual = float(log_rets.mean() * 252.0)
    sigma_annual = float(log_rets.std() * np.sqrt(252.0))

    n_months = horizon_years * 12
    dt = 1.0 / 12.0

    if params.dca_frequency == "Hebdomadaire":
        monthly_gross_dca = params.dca_amount * (52.0 / 12.0)
        monthly_fee = compute_order_fee(params.dca_amount, params) * (52.0 / 12.0)
    elif params.dca_frequency == "Mensuelle":
        monthly_gross_dca = params.dca_amount
        monthly_fee = compute_order_fee(params.dca_amount, params)
    else:
        monthly_gross_dca = params.dca_amount / 3.0
        monthly_fee = compute_order_fee(params.dca_amount, params) / 3.0

    monthly_net_dca = max(0.0, monthly_gross_dca - monthly_fee)

    rng = np.random.default_rng(seed=42)
    z = rng.standard_normal((n_months, n_trajectories))
    monthly_growth = np.exp((mu_annual - 0.5 * sigma_annual**2) * dt + sigma_annual * np.sqrt(dt) * z)

    start_wealth = float(hist_df["Portfolio_Gross_Value"].iloc[-1])
    start_invested = float(hist_df["Cum_Invested"].iloc[-1])

    wealth_paths = np.zeros((n_months + 1, n_trajectories), dtype=np.float64)
    wealth_paths[0, :] = start_wealth
    invested_path = np.zeros(n_months + 1, dtype=np.float64)
    invested_path[0] = start_invested

    for m in range(1, n_months + 1):
        wealth_paths[m, :] = (wealth_paths[m - 1, :] + monthly_net_dca) * monthly_growth[m - 1, :]
        invested_path[m] = invested_path[m - 1] + monthly_gross_dca

    future_dates = pd.date_range(
        start=hist_df.index[-1], periods=n_months + 1, freq="ME"
    )

    return pd.DataFrame(
        {
            "Date": future_dates,
            "Cum_Invested": invested_path,
            "P10": np.percentile(wealth_paths, 10, axis=1),
            "P25": np.percentile(wealth_paths, 25, axis=1),
            "P50": np.percentile(wealth_paths, 50, axis=1),
            "P75": np.percentile(wealth_paths, 75, axis=1),
            "P90": np.percentile(wealth_paths, 90, axis=1),
        }
    ).set_index("Date")


# ==============================================================================
# 5. COMPOSANTS GRAPHIQUES (PLOTLY DARK / MODERN THEME)
# ==============================================================================


def apply_modern_dark_layout(fig: go.Figure, title: str, height: int = 480) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#F8FAFC"), x=0.01),
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.0)",
        plot_bgcolor="rgba(15, 23, 42, 0.45)",
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.12)",
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.12)",
            zeroline=False,
        ),
    )
    return fig


def plot_portfolio_overview(res: AssetBacktestResult, params: SimulationParams) -> go.Figure:
    df = res.daily_df
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Cum_Invested"],
            name="Cumul des Versements (Capital Investi)",
            mode="lines",
            line=dict(color="#94A3B8", width=2, shape="hv"),
            fill="tozeroy",
            fillcolor="rgba(148, 163, 184, 0.12)",
            hovertemplate="%{y:,.0f} €<extra>Capital Investi</extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Portfolio_Gross_Value"],
            name=f"Valorisation Portefeuille ({res.asset_label})",
            mode="lines",
            line=dict(color="#00D4FF", width=2.8),
            fill="tonexty",
            fillcolor="rgba(0, 212, 255, 0.12)",
            hovertemplate="%{y:,.0f} €<extra>Valeur Portefeuille</extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Portfolio_Real_Gross_Value"],
            name=f"Pouvoir d'Achat Réel (Inflation {params.inflation_rate * 100:.1f}%/an)",
            mode="lines",
            line=dict(color="#F59E0B", width=2, dash="dot"),
            hovertemplate="%{y:,.0f} €<extra>Valeur Réelle (Euros Constants)</extra>",
        )
    )

    fig.update_yaxes(title_text="Valorisation (€)", tickformat=",.0f")
    return apply_modern_dark_layout(
        fig,
        title=f"Évolution de la Valorisation, du Capital Versé et du Pouvoir d'Achat — {res.asset_label}",
        height=500,
    )


def plot_underwater_drawdown(res: AssetBacktestResult) -> go.Figure:
    df = res.daily_df
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Drawdown_Underlying"] * 100.0,
            name="Drawdown Sous-Jacent Pur (Marché)",
            mode="lines",
            line=dict(color="rgba(248, 113, 113, 0.55)", width=1.4, dash="dot"),
            hovertemplate="%{y:.2f} %<extra>DD Sous-Jacent</extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Drawdown_DCA"] * 100.0,
            name="Drawdown Portefeuille DCA",
            mode="lines",
            line=dict(color="#EF4444", width=2.2),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.25)",
            hovertemplate="%{y:.2f} %<extra>DD Portefeuille DCA</extra>",
        )
    )

    fig.update_yaxes(title_text="Baisse depuis le plus haut (%)", ticksuffix=" %")
    return apply_modern_dark_layout(
        fig,
        title=f"Graphique Underwater (Drawdown depuis le dernier ATH) — Max DD : {res.dca_metrics.max_drawdown * 100:.2f}%",
        height=420,
    )


def plot_market_price_vs_pru(res: AssetBacktestResult) -> go.Figure:
    df = res.daily_df
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Market_Price"],
            name=f"Prix du Marché ({res.ticker})",
            mode="lines",
            line=dict(color="#10B981", width=2.2),
            hovertemplate="%{y:,.2f} €<extra>Prix du Marché</extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["PRU"],
            name="PRU Moyen Pondéré (DCA)",
            mode="lines",
            line=dict(color="#F59E0B", width=2.6),
            hovertemplate="%{y:,.2f} €<extra>PRU (Coût Moyen)</extra>",
        )
    )

    bargain_buys = df[(df["Gross_Contribution"] > 0) & (df["Market_Price"] < df["PRU"])]
    if not bargain_buys.empty:
        fig.add_trace(
            go.Scatter(
                x=bargain_buys.index,
                y=bargain_buys["Market_Price"],
                name="Achats DCA sous le PRU (Relutifs)",
                mode="markers",
                marker=dict(color="#38BDF8", size=6, symbol="circle"),
                hovertemplate="Achat à %{y:,.2f} € (sous le PRU)<extra></extra>",
            )
        )

    fig.update_yaxes(title_text="Prix unitaire (€)", tickformat=",.2f")
    return apply_modern_dark_layout(
        fig,
        title=f"Dynamique du Prix de Marché vs Prix de Revient Unitaire (PRU) — {res.asset_label}",
        height=440,
    )


def plot_multi_asset_comparison(results: List[AssetBacktestResult]) -> go.Figure:
    fig = go.Figure()

    if not results:
        return fig

    ref_df = results[0].daily_df
    fig.add_trace(
        go.Scatter(
            x=ref_df.index,
            y=ref_df["Cum_Invested"],
            name="Capital Total Versé",
            mode="lines",
            line=dict(color="#94A3B8", width=2, dash="dash", shape="hv"),
            hovertemplate="%{y:,.0f} €<extra>Capital Versé</extra>",
        )
    )

    for idx, res in enumerate(results):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=res.daily_df.index,
                y=res.daily_df["Portfolio_Gross_Value"],
                name=f"{res.asset_label} (TRI: {res.dca_metrics.xirr * 100:.2f}%)",
                mode="lines",
                line=dict(color=color, width=2.5),
                hovertemplate=f"%{{y:,.0f}} €<extra>{res.asset_label}</extra>",
            )
        )

    fig.update_yaxes(title_text="Valorisation Brute (€)", tickformat=",.0f")
    return apply_modern_dark_layout(
        fig,
        title="Comparaison Multi-Actifs à Stratégie DCA Identique",
        height=480,
    )


def plot_dca_vs_lumpsum(res: AssetBacktestResult) -> go.Figure:
    df = res.daily_df
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=(
            f"Valorisation Comparée : DCA vs Lump Sum ({res.asset_label})",
            "Drawdown Comparé (% depuis le plus haut)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Cum_Invested"],
            name="Capital Cumulé DCA",
            mode="lines",
            line=dict(color="#64748B", width=1.8, dash="dot", shape="hv"),
            hovertemplate="%{y:,.0f} €<extra>Cumul Versé DCA</extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Portfolio_Gross_Value"],
            name="Stratégie DCA (Progressif)",
            mode="lines",
            line=dict(color="#00D4FF", width=2.8),
            hovertemplate="%{y:,.0f} €<extra>Portefeuille DCA</extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["LumpSum_Gross_Value"],
            name="Stratégie 100% Lump Sum (Jour 1)",
            mode="lines",
            line=dict(color="#A855F7", width=2.5),
            hovertemplate="%{y:,.0f} €<extra>Portefeuille Lump Sum</extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Drawdown_DCA"] * 100.0,
            name="Drawdown DCA (%)",
            mode="lines",
            line=dict(color="#00D4FF", width=1.8),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 255, 0.15)",
            hovertemplate="%{y:.2f} %<extra>DD DCA</extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Drawdown_LumpSum"] * 100.0,
            name="Drawdown Lump Sum (%)",
            mode="lines",
            line=dict(color="#A855F7", width=1.8),
            hovertemplate="%{y:.2f} %<extra>DD Lump Sum</extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Valorisation (€)", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix=" %", row=2, col=1)
    return apply_modern_dark_layout(fig, title="", height=600)


def plot_monte_carlo_cone(mc_df: pd.DataFrame, asset_label: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=mc_df.index,
            y=mc_df["P90"],
            name="Scénario Optimiste (90e percentile)",
            mode="lines",
            line=dict(color="rgba(16, 185, 129, 0.5)", width=1.5),
            hovertemplate="%{y:,.0f} €<extra>P90 (Optimiste)</extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=mc_df.index,
            y=mc_df["P10"],
            name="Scénario Prudent (10e percentile)",
            mode="lines",
            line=dict(color="rgba(239, 68, 68, 0.5)", width=1.5),
            fill="tonexty",
            fillcolor="rgba(0, 212, 255, 0.12)",
            hovertemplate="%{y:,.0f} €<extra>P10 (Prudent)</extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=mc_df.index,
            y=mc_df["P50"],
            name="Trajectoire Médiane (P50)",
            mode="lines",
            line=dict(color="#00D4FF", width=3.0),
            hovertemplate="%{y:,.0f} €<extra>Médiane (P50)</extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=mc_df.index,
            y=mc_df["Cum_Invested"],
            name="Capital Cumulé Projeté",
            mode="lines",
            line=dict(color="#94A3B8", width=2.0, dash="dash"),
            hovertemplate="%{y:,.0f} €<extra>Capital Projeté</extra>",
        )
    )

    fig.update_yaxes(title_text="Valorisation Projetée (€)", tickformat=",.0f")
    return apply_modern_dark_layout(
        fig,
        title=f"Projection Probabiliste Monte Carlo (500 trajectoires GBM) — {asset_label}",
        height=440,
    )


# ==============================================================================
# 6. INTERFACE UTILISATEUR : SIDEBAR DE PARAMÉTRAGE
# ==============================================================================


def render_sidebar() -> Tuple[List[Tuple[str, str, str, bool]], Optional[io.BytesIO], SimulationParams]:
    st.sidebar.title("⚙️ Paramètres Quant")
    st.sidebar.caption("Configuration du moteur de backtesting vectorisé")

    st.sidebar.subheader("1. Univers d'Investissement")
    default_choices = [
        "S&P 500 - ETF SPY (Total Return / Div. Réinvestis)",
        "Amundi MSCI World PEA (CW8.PA - EUR)",
        "CAC 40 - Indice (^FCHI)",
    ]
    selected_labels = st.sidebar.multiselect(
        "Indices & ETFs préconfigurés",
        options=list(PRECONFIGURED_ASSETS.keys()),
        default=default_choices,
    )

    custom_tickers_raw = st.sidebar.text_input(
        "Tickers Yahoo personnalisés (séparés par des virgules)",
        value="",
        placeholder="Ex: ASML.AS, AI.PA, SU.PA, NVDA",
    )

    uploaded_csv = st.sidebar.file_uploader(
        "Import CSV personnalisé (Optionnel)",
        type=["csv"],
    )

    convert_usd_to_eur = st.sidebar.toggle(
        "Convertir les actifs USD en EUR (Taux réel EURUSD=X)",
        value=True,
    )

    assets_to_fetch: List[Tuple[str, str, str, bool]] = []
    for lbl in selected_labels:
        tck, curr, is_bare = PRECONFIGURED_ASSETS[lbl]
        assets_to_fetch.append((lbl, tck, curr, is_bare))

    if custom_tickers_raw.strip():
        for raw_t in custom_tickers_raw.split(","):
            clean_t = raw_t.strip().upper()
            if clean_t and not any(clean_t == x[1] for x in assets_to_fetch):
                is_bare_custom = clean_t.startswith("^")
                assets_to_fetch.append((f"Custom ({clean_t})", clean_t, "USD", is_bare_custom))

    st.sidebar.divider()

    st.sidebar.subheader("2. Période Historique")
    today = date.today()
    default_start = today - timedelta(days=int(15 * 365.25))

    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1:
        start_date = st.date_input(
            "Date de début",
            value=default_start,
            min_value=date(1980, 1, 1),
            max_value=today - timedelta(days=30),
        )
    with col_d2:
        end_date = st.date_input(
            "Date de fin",
            value=today,
            min_value=start_date + timedelta(days=30),
            max_value=today,
        )

    st.sidebar.divider()

    st.sidebar.subheader("3. Flux d'Investissement")
    initial_capital = st.sidebar.number_input(
        "Capital initial (Lump Sum au Jour 1) en €",
        min_value=0.0,
        max_value=10_000_000.0,
        value=5_000.0,
        step=500.0,
    )
    dca_amount = st.sidebar.number_input(
        "Versement programmé (DCA) en €",
        min_value=0.0,
        max_value=1_000_000.0,
        value=400.0,
        step=50.0,
    )
    dca_frequency = st.sidebar.selectbox(
        "Fréquence du DCA",
        options=["Mensuelle", "Hebdomadaire", "Trimestrielle"],
        index=0,
    )

    if dca_frequency in ["Mensuelle", "Trimestrielle"]:
        dca_day_rule = st.sidebar.selectbox(
            "Jour d'exécution dans le mois",
            options=["1er du mois", "15 du mois", "Dernier jour ouvré"],
            index=0,
        )
    else:
        dca_day_rule = "1er du mois"

    st.sidebar.divider()

    st.sidebar.subheader("4. Frais & Fiscalité")
    fee_mode = st.sidebar.radio(
        "Type de frais de courtage par ordre",
        options=["Pourcentage (%)", "Fixe (€)"],
        horizontal=True,
    )

    if fee_mode == "Pourcentage (%)":
        col_f1, col_f2 = st.sidebar.columns(2)
        with col_f1:
            fee_pct_input = st.number_input(
                "Courtage (%)", min_value=0.0, max_value=5.0, value=0.10, step=0.05, format="%.2f"
            )
        with col_f2:
            fee_min_eur = st.number_input(
                "Minimum / ordre (€)", min_value=0.0, max_value=50.0, value=0.0, step=0.5, format="%.2f"
            )
        fee_fixed_eur = 0.0
    else:
        fee_fixed_eur = st.sidebar.number_input(
            "Frais fixe par ordre (€)", min_value=0.0, max_value=100.0, value=1.0, step=0.5, format="%.2f"
        )
        fee_pct_input = 0.0
        fee_min_eur = 0.0

    ter_input = st.sidebar.number_input(
        "Frais de gestion annuels ETF (TER en %)",
        min_value=0.0,
        max_value=3.0,
        value=0.20,
        step=0.05,
        format="%.2f",
    )
    apply_ter_to_etfs = st.sidebar.checkbox(
        "Déduire aussi ce TER sur les ETFs cotés",
        value=False,
    )

    tax_options = {
        "PEA (> 5 ans) — 17,2 % (Prélèvements sociaux)": 0.172,
        "Compte-Titres Ordinaire (CTO - Flat Tax) — 30,0 %": 0.30,
        "Brut d'impôt — 0,0 %": 0.0,
    }
    tax_regime_name = st.sidebar.selectbox(
        "Fiscalité à la sortie (sur la plus-value nette)",
        options=list(tax_options.keys()),
        index=0,
    )

    col_m1, col_m2 = st.sidebar.columns(2)
    with col_m1:
        inflation_input = st.number_input(
            "Inflation moy. (%/an)", min_value=0.0, max_value=15.0, value=2.0, step=0.2, format="%.1f"
        )
    with col_m2:
        rf_input = st.number_input(
            "Taux sans risque (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.25, format="%.2f"
        )

    params = SimulationParams(
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(initial_capital),
        dca_amount=float(dca_amount),
        dca_frequency=dca_frequency,
        dca_day_rule=dca_day_rule,
        fee_mode=fee_mode,
        fee_pct=float(fee_pct_input) / 100.0,
        fee_min_eur=float(fee_min_eur),
        fee_fixed_eur=float(fee_fixed_eur),
        annual_ter=float(ter_input) / 100.0,
        apply_ter_to_etfs=apply_ter_to_etfs,
        tax_regime_name=tax_regime_name,
        tax_rate=tax_options[tax_regime_name],
        inflation_rate=float(inflation_input) / 100.0,
        risk_free_rate=float(rf_input) / 100.0,
        convert_usd_to_eur=convert_usd_to_eur,
    )

    return assets_to_fetch, uploaded_csv, params


# ==============================================================================
# 7. ORCHESTRATION PRINCIPALE & AFFICHAGE DES 4 ONGLETS
# ==============================================================================


def main() -> None:
    st.title("📊 QuantDCA & Lump Sum Backtester Pro")
    st.markdown(
        "**Simulateur quantitatif d'investissement programmé (DCA) et Lump Sum** — "
        "Données historiques réelles Yahoo Finance & CSV, calcul exact du TRI (XIRR par Newton-Raphson), "
        "analyse de Drawdown (Underwater), fiscalité PEA/CTO et projections Monte Carlo."
    )

    assets_to_fetch, uploaded_csv, params = render_sidebar()

    if not assets_to_fetch and uploaded_csv is None:
        st.warning(
            "👈 Veuillez sélectionner au moins un actif dans la barre latérale ou importer un fichier CSV pour lancer la simulation."
        )
        st.stop()

    fx_series: Optional[pd.Series] = None
    if params.convert_usd_to_eur:
        with st.spinner("Synchronisation de la parité historique EUR/USD (EURUSD=X)..."):
            fx_series = fetch_eurusd_fx_series(params.start_date, params.end_date)

    results: List[AssetBacktestResult] = []
    errors: List[str] = []

    with st.spinner("Téléchargement des séries temporelles et exécution des calculs vectorisés..."):
        if uploaded_csv is not None:
            csv_series, csv_curr, csv_err = parse_custom_csv(
                uploaded_csv, params.start_date, params.end_date
            )
            if csv_err:
                errors.append(csv_err)
            elif csv_series is not None:
                csv_label = f"CSV Personnalisé ({uploaded_csv.name})"
                res_csv = run_asset_backtest(
                    asset_label=csv_label,
                    ticker="CSV_CUSTOM",
                    raw_price_series=csv_series,
                    native_currency=csv_curr,
                    is_bare_index=False,
                    params=params,
                    fx_eurusd_series=fx_series,
                )
                results.append(res_csv)

        for label, ticker, default_curr, is_bare in assets_to_fetch:
            s_price, detected_curr, err = fetch_yahoo_series(
                ticker, params.start_date, params.end_date
            )
            if err or s_price is None:
                errors.append(err or f"Erreur inconnue sur {ticker}")
                continue

            eff_curr = default_curr if default_curr == "EUR" else detected_curr
            res = run_asset_backtest(
                asset_label=label,
                ticker=ticker,
                raw_price_series=s_price,
                native_currency=eff_curr,
                is_bare_index=is_bare,
                params=params,
                fx_eurusd_series=fx_series,
            )
            results.append(res)

    for e in errors:
        st.error(e)

    if not results:
        st.error("Aucun actif n'a pu être chargé sur la période demandée. Vérifiez vos tickers ou votre connexion.")
        st.stop()

    if len(results) > 1:
        selected_primary_label = st.selectbox(
            "🎯 Actif principal analysé en détail (Onglets 1, 2 et 4) :",
            options=[r.asset_label for r in results],
            index=0,
        )
        primary_res = next(r for r in results if r.asset_label == selected_primary_label)
    else:
        primary_res = results[0]

    m_dca = primary_res.dca_metrics
    m_ls = primary_res.lump_sum_metrics

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 1. Vue d'ensemble & Performance",
            "🛡️ 2. Analyse du Risque & PRU",
            "⚔️ 3. Comparateur Multi-Indices & DCA vs Lump Sum",
            "🗂️ 4. Données Annuelles & Export",
        ]
    )

    with tab1:
        st.subheader(f"Synthèse de Performance — {primary_res.asset_label}")
        eff_start_str = primary_res.daily_df.index[0].strftime("%d/%m/%Y")
        eff_end_str = primary_res.daily_df.index[-1].strftime("%d/%m/%Y")
        st.caption(
            f"Période effective cotée : du **{eff_start_str}** au **{eff_end_str}** "
            f"({m_dca.num_orders} ordres exécutés | Fiscalité : {params.tax_regime_name})"
        )

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric(
            label="Capital Total Investi",
            value=f"{m_dca.total_invested:,.0f} €",
            delta=f"Frais courtage : {m_dca.total_brokerage_fees:,.0f} €",
            delta_color="off",
        )
        k2.metric(
            label="Valeur Finale Brute",
            value=f"{m_dca.final_gross_value:,.0f} €",
            delta=f"{(m_dca.gross_pnl / m_dca.total_invested) * 100:+.1f} % brut",
        )
        k3.metric(
            label="Valeur Finale Nette",
            value=f"{m_dca.final_net_value:,.0f} €",
            delta=f"Impôt : -{m_dca.tax_paid:,.0f} €",
            delta_color="normal",
        )
        k4.metric(
            label="Plus-Value Nette",
            value=f"{m_dca.net_pnl:+,.0f} €",
            delta=f"{m_dca.net_return_pct * 100:+.2f} % net",
        )
        k5.metric(
            label="TRI Annualisé (XIRR)",
            value=f"{m_dca.xirr * 100:.2f} % / an",
            delta=f"CAGR Actif : {m_dca.underlying_cagr * 100:.2f} %",
        )
        k6.metric(
            label="Max Drawdown (DCA)",
            value=f"{m_dca.max_drawdown * 100:.2f} %",
            delta=f"{m_dca.max_underwater_days} j sous l'eau max",
            delta_color="inverse",
        )

        st.plotly_chart(
            plot_portfolio_overview(primary_res, params),
            use_container_width=True,
        )

        c_inf1, c_inf2, c_inf3, c_inf4 = st.columns(4)
        c_inf1.metric(
            "Pouvoir d'Achat Net Final (Euros Constants)",
            f"{m_dca.final_real_net_value:,.0f} €",
        )
        c_inf2.metric(
            "Volatilité Annualisée",
            f"{m_dca.annualized_volatility * 100:.2f} %",
        )
        c_inf3.metric(
            f"Ratio de Sharpe (Rf = {params.risk_free_rate * 100:.1f}%)",
            f"{m_dca.sharpe_ratio:.2f}",
        )
        c_inf4.metric(
            f"Ratio de Sortino (Rf = {params.risk_free_rate * 100:.1f}%)",
            f"{m_dca.sortino_ratio:.2f}",
        )

        with st.expander("🔮 Projection Probabiliste Future (Simulation Monte Carlo continuant le DCA)", expanded=False):
            col_mc1, col_mc2 = st.columns([1, 3])
            with col_mc1:
                mc_horizon = st.slider(
                    "Horizon de projection future (Années)",
                    min_value=3,
                    max_value=30,
                    value=10,
                    step=1,
                )
                mc_df = run_monte_carlo_projection(primary_res, params, horizon_years=mc_horizon)
                st.markdown(f"**Résultats projetés à +{mc_horizon} ans :**")
                st.metric("Capital Cumulé Projeté", f"{mc_df['Cum_Invested'].iloc[-1]:,.0f} €")
                st.metric("Scénario Médian (P50)", f"{mc_df['P50'].iloc[-1]:,.0f} €")
                st.metric("Scénario Optimiste (P90)", f"{mc_df['P90'].iloc[-1]:,.0f} €")
                st.metric("Scénario Prudent (P10)", f"{mc_df['P10'].iloc[-1]:,.0f} €")
            with col_mc2:
                st.plotly_chart(
                    plot_monte_carlo_cone(mc_df, primary_res.asset_label),
                    use_container_width=True,
                )

    with tab2:
        st.subheader(f"🛡️ Analyse Psychologique du Risque (Drawdown) & Lissage du PRU — {primary_res.asset_label}")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("PRU Final Moyen Pondéré", f"{m_dca.final_pru:,.2f} €")
        r2.metric(
            "Prix de Marché Final",
            f"{m_dca.final_market_price:,.2f} €",
            delta=f"{((m_dca.final_market_price / m_dca.final_pru) - 1.0) * 100:+.2f} % vs PRU",
        )
        r3.metric("Nombre Total de Parts", f"{m_dca.total_shares:,.4f}")
        r4.metric("Plus longue période Underwater", f"{m_dca.max_underwater_days} jours")

        st.plotly_chart(
            plot_underwater_drawdown(primary_res),
            use_container_width=True,
        )

        st.plotly_chart(
            plot_market_price_vs_pru(primary_res),
            use_container_width=True,
        )

    with tab3:
        st.subheader("⚔️ Comparaison Multi-Actifs & Duel DCA vs Lump Sum")

        st.plotly_chart(
            plot_multi_asset_comparison(results),
            use_container_width=True,
        )

        comp_rows = []
        for r in results:
            dm = r.dca_metrics
            lm = r.lump_sum_metrics
            comp_rows.append(
                {
                    "Actif": r.asset_label,
                    "Ticker": r.ticker,
                    "Capital Versé (€)": round(dm.total_invested, 0),
                    "Valeur Finale Nette DCA (€)": round(dm.final_net_value, 0),
                    "Plus-Value Nette DCA (€)": round(dm.net_pnl, 0),
                    "TRI / XIRR DCA (%)": round(dm.xirr * 100, 2),
                    "Valeur Finale Nette Lump Sum (€)": round(lm.final_net_value, 0),
                    "TRI Lump Sum (%)": round(lm.xirr * 100, 2),
                    "CAGR Sous-Jacent (%)": round(dm.underlying_cagr * 100, 2),
                    "Volatilité (%)": round(dm.annualized_volatility * 100, 2),
                    "Max Drawdown DCA (%)": round(dm.max_drawdown * 100, 2),
                    "Sharpe (DCA)": round(dm.sharpe_ratio, 2),
                    "Sortino (DCA)": round(dm.sortino_ratio, 2),
                }
            )
        st.markdown("#### 📋 Tableau Comparatif Multi-Actifs (Net de fiscalité et de frais)")
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        st.divider()

        st.markdown(f"#### 🥊 Duel Chiffré & Visuel : DCA vs 100% Lump Sum — **{primary_res.asset_label}**")
        st.caption(
            f"Le benchmark **Lump Sum** simule l'investissement immédiat de la totalité des **{m_dca.total_invested:,.0f} €** "
            f"(capital initial + somme de tous les versements DCA futurs) dès le premier jour du backtest ({eff_start_str})."
        )

        diff_net_val = m_dca.final_net_value - m_ls.final_net_value
        winner_str = "DCA Gagnant" if diff_net_val >= 0 else "Lump Sum Gagnant"

        d1, d2, d3, d4 = st.columns(4)
        d1.metric(
            "Valeur Nette Finale — DCA",
            f"{m_dca.final_net_value:,.0f} €",
            delta=f"TRI (XIRR) : {m_dca.xirr * 100:.2f} %/an",
        )
        d2.metric(
            "Valeur Nette Finale — Lump Sum",
            f"{m_ls.final_net_value:,.0f} €",
            delta=f"TRI (CAGR Net) : {m_ls.xirr * 100:.2f} %/an",
        )
        d3.metric(
            "Écart Net (DCA - Lump Sum)",
            f"{diff_net_val:+,.0f} €",
            delta=winner_str,
            delta_color="normal" if diff_net_val >= 0 else "inverse",
        )
        d4.metric(
            "Réduction du Max Drawdown via DCA",
            f"{(m_dca.max_drawdown - m_ls.max_drawdown) * 100:+.2f} pts",
            delta=f"DCA: {m_dca.max_drawdown * 100:.1f}% vs LS: {m_ls.max_drawdown * 100:.1f}%",
            delta_color="normal",
        )

        st.plotly_chart(
            plot_dca_vs_lumpsum(primary_res),
            use_container_width=True,
        )

    with tab4:
        st.subheader(f"📅 Bilan Annuel Détaillé & Export CSV — {primary_res.asset_label}")

        annual_df = build_annual_summary_table(primary_res.daily_df)
        st.dataframe(
            annual_df.style.format(
                {
                    "Versements de l'année (€)": "{:,.2f} €",
                    "Cumul Versé (€)": "{:,.2f} €",
                    "Valeur Fin d'Année (€)": "{:,.2f} €",
                    "Gain / Perte de l'année (€)": "{:+,.2f} €",
                    "Performance Annuelle (%)": "{:+.2f} %",
                    "Plus/Moins-Value Latente (€)": "{:+,.2f} €",
                    "Plus/Moins-Value Latente (%)": "{:+.2f} %",
                    "PRU Fin d'Année (€)": "{:,.2f} €",
                    "Prix Marché Fin d'Année (€)": "{:,.2f} €",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.markdown("#### 💾 Export des Données Quotidiennes et de l'Historique des Transactions")

        col_exp1, col_exp2, col_exp3 = st.columns(3)

        daily_csv_bytes = primary_res.daily_df.to_csv(index=True).encode("utf-8")
        with col_exp1:
            st.download_button(
                label="📥 Télécharger la Valorisation Quotidienne (CSV)",
                data=daily_csv_bytes,
                file_name=f"valorisation_quotidienne_{primary_res.ticker}_{params.start_date}_{params.end_date}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        tx_df = primary_res.daily_df[primary_res.daily_df["Gross_Contribution"] > 0][
            [
                "Market_Price",
                "Gross_Contribution",
                "Brokerage_Fee",
                "Net_Contribution",
                "Shares_Bought",
                "Cum_Shares",
                "PRU",
                "Cum_Invested",
                "Portfolio_Gross_Value",
            ]
        ].copy()
        tx_csv_bytes = tx_df.to_csv(index=True).encode("utf-8")
        with col_exp2:
            st.download_button(
                label="📥 Télécharger le Journal des Ordres DCA (CSV)",
                data=tx_csv_bytes,
                file_name=f"transactions_dca_{primary_res.ticker}_{params.start_date}_{params.end_date}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        annual_csv_bytes = annual_df.to_csv(index=False).encode("utf-8")
        with col_exp3:
            st.download_button(
                label="📥 Télécharger le Bilan Annuel (CSV)",
                data=annual_csv_bytes,
                file_name=f"bilan_annuel_{primary_res.ticker}_{params.start_date}_{params.end_date}.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
