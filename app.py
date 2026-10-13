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

st.set_page_config(page_title="QuantBacktester Pro | DCA vs Lump Sum", page_icon="📈", layout="wide")

st.markdown("""<style>
div[data-testid="metric-container"]{background-color:rgba(28,31,48,0.65);border:1px solid rgba(128,128,128,0.25);padding:14px 18px;border-radius:10px;}
</style>""", unsafe_allow_html=True)

PRECONFIGURED_ASSETS: Dict[str, Tuple[str, str, bool]] = {
    "S&P 500 - Indice Nu (^GSPC)": ("^GSPC", "USD", True),
    "S&P 500 - ETF SPY (Div. Réinvestis)": ("SPY", "USD", False),
    "Nasdaq 100 - Indice Nu (^NDX)": ("^NDX", "USD", True),
    "Nasdaq 100 - ETF Invesco QQQ": ("QQQ", "USD", False),
    "MSCI World - iShares Core (IWDA.AS - EUR)": ("IWDA.AS", "EUR", False),
    "MSCI World - iShares ETF (URTH - USD)": ("URTH", "USD", False),
    "Amundi MSCI World PEA (CW8.PA - EUR)": ("CW8.PA", "EUR", False),
    "Euro Stoxx 50 - Indice (^STOXX50E)": ("^STOXX50E", "EUR", True),
    "CAC 40 - Indice (^FCHI)": ("^FCHI", "EUR", True),
    "Or Physique - ETF GLD": ("GLD", "USD", False),
    "Bitcoin (BTC-USD)": ("BTC-USD", "USD", False),
}

COLOR_PALETTE = ["#00D4FF", "#FF6B6B", "#10B981", "#F59E0B", "#A855F7", "#EC4899", "#3B82F6", "#84CC16"]

@dataclass
class SimulationParams:
    start_date: date; end_date: date; initial_capital: float; dca_amount: float
    dca_frequency: str; dca_day_rule: str; fee_mode: str; fee_pct: float
    fee_min_eur: float; fee_fixed_eur: float; annual_ter: float; apply_ter_to_etfs: bool
    tax_regime_name: str; tax_rate: float; inflation_rate: float; risk_free_rate: float
    convert_usd_to_eur: bool

@dataclass
class BacktestMetrics:
    total_invested: float; final_gross_value: float; final_net_value: float; final_real_net_value: float
    gross_pnl: float; net_pnl: float; net_return_pct: float; tax_paid: float
    total_brokerage_fees: float; xirr: float; underlying_cagr: float; portfolio_cagr: float
    annualized_volatility: float; sharpe_ratio: float; sortino_ratio: float
    max_drawdown: float; max_underwater_days: int; total_shares: float
    final_pru: float; final_market_price: float; num_orders: int

@dataclass
class AssetBacktestResult:
    asset_label: str; ticker: str; currency: str; daily_df: pd.DataFrame
    dca_metrics: BacktestMetrics; lump_sum_metrics: BacktestMetrics
    cashflows_dca: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)
    cashflows_ls: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)

def _extract_price_series(raw_df: pd.DataFrame, ticker: str) -> pd.Series:
    if raw_df is None or raw_df.empty:
        return pd.Series(dtype=float)
    df = raw_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        l0 = df.columns.get_level_values(0).unique().tolist()
        l1 = df.columns.get_level_values(1).unique().tolist()
        for c in ["Adj Close", "Close"]:
            if c in l0:
                sub = df[c]
                return (sub[ticker] if isinstance(sub, pd.DataFrame) and ticker in sub.columns else (sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub)).astype(float)
            if c in l1:
                sub = df.xs(c, axis=1, level=1)
                return (sub[ticker] if isinstance(sub, pd.DataFrame) and ticker in sub.columns else (sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub)).astype(float)
        return df.iloc[:, 0].astype(float)
    for col in ["Adj Close", "Close", "adj close", "close"]:
        if col in df.columns:
            s = df[col]
            return (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s).astype(float)
    num_cols = df.select_dtypes(include=[np.number]).columns
    return df[num_cols[0]].astype(float) if len(num_cols) > 0 else pd.Series(dtype=float)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_series(ticker: str, start_date: date, end_date: date) -> Tuple[Optional[pd.Series], str, Optional[str]]:
    try:
        dl_start = (pd.Timestamp(start_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        dl_end = (pd.Timestamp(end_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        raw = yf.download(tickers=ticker, start=dl_start, end=dl_end, auto_adjust=True, progress=False, threads=False)
        if raw is None or raw.empty:
            return None, "USD", f"Aucune donnée renvoyée par Yahoo Finance pour '{ticker}'."
        series = _extract_price_series(raw, ticker).dropna()
        if series.empty:
            return None, "USD", f"Série de prix vide pour '{ticker}'."
        series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
        series = series[~series.index.duplicated(keep="last")].sort_index()
        currency = "EUR" if ticker.upper().endswith((".PA", ".AS", ".DE", ".MI", ".MC", ".BR")) or ticker.upper() in ["^STOXX50E", "^FCHI", "^GDAXI"] else "USD"
        eff_start, eff_end = max(pd.Timestamp(start_date), series.index.min()), min(pd.Timestamp(end_date), series.index.max())
        if eff_start >= eff_end:
            return None, currency, f"Plage de dates sans cotation pour '{ticker}'."
        cal = pd.date_range(start=eff_start, end=eff_end, freq="D")
        series = series.reindex(cal.union(series.index)).sort_index().ffill().bfill().reindex(cal)
        return series, currency, None
    except Exception as exc:
        return None, "USD", f"Erreur réseau sur '{ticker}' : {str(exc)}"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_eurusd_fx_series(start_date: date, end_date: date) -> pd.Series:
    fx, _, err = fetch_yahoo_series("EURUSD=X", start_date, end_date)
    if fx is None or fx.empty or err is not None:
        return pd.Series(1.10, index=pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq="D"))
    return fx
    def parse_custom_csv(uploaded_file: io.BytesIO, start_date: date, end_date: date) -> Tuple[Optional[pd.Series], str, Optional[str]]:
    try:
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        first_line = content.splitlines()[0] if content else ""
        sep = ";" if ";" in first_line else ("\t" if "\t" in first_line else ",")
        df = pd.read_csv(io.StringIO(content), sep=sep)
        if df.empty or len(df.columns) < 2:
            return None, "EUR", "Le CSV doit contenir une colonne Date et une colonne Prix."
        date_col = next((c for c in df.columns if any(k in str(c).lower() for k in ["date", "time", "jour", "day"])), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        df = df.dropna(subset=[date_col]).sort_values(by=date_col)
        kws = ["adj close", "close", "clôture", "cloture", "prix", "price", "valeur", "nav", "vl"]
        price_col = next((c for k in kws for c in df.columns if c != date_col and k in str(c).lower()), [c for c in df.columns if c != date_col][0])
        if df[price_col].dtype == object:
            df[price_col] = pd.to_numeric(df[price_col].astype(str).str.replace(" ", "", regex=False).str.replace("€", "", regex=False).str.replace("$", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
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

def generate_dca_schedule(calendar_index: pd.DatetimeIndex, frequency: str, day_rule: str) -> pd.DatetimeIndex:
    if calendar_index.empty:
        return pd.DatetimeIndex([])
    start_ts, end_ts = calendar_index.min(), calendar_index.max()
    if frequency == "Hebdomadaire":
        t = calendar_index[calendar_index.weekday == 0]
        return t[t > start_ts]
    mdf = pd.DataFrame({"date": calendar_index, "y": calendar_index.year, "m": calendar_index.month, "b": calendar_index.weekday < 5})
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
        fee = max(gross_amount * params.fee_pct, params.fee_min_eur) if (params.fee_pct > 0 or params.fee_min_eur > 0) else 0.0
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
    def run_asset_backtest(asset_label: str, ticker: str, raw_price_series: pd.Series, native_currency: str, is_bare_index: bool, params: SimulationParams, fx_eurusd_series: Optional[pd.Series] = None) -> AssetBacktestResult:
    prices = raw_price_series.copy().sort_index()
    eff_curr = native_currency
    if params.convert_usd_to_eur and native_currency == "USD" and fx_eurusd_series is not None:
        prices = prices / fx_eurusd_series.reindex(prices.index).ffill().bfill()
        eff_curr = "EUR"
    days_el = np.arange(len(prices), dtype=np.float64)
    adj_prices = prices * np.power(1.0 - params.annual_ter, days_el / 365.25) if (params.annual_ter > 0 and (is_bare_index or params.apply_ter_to_etfs)) else prices.copy()
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

    fg_dca = float(val_dca.iloc[-1]); gpnl_dca = fg_dca - tot_cap; tax_dca = max(0.0, gpnl_dca * params.tax_rate)
    fn_dca = fg_dca - tax_dca; npnl_dca = fn_dca - tot_cap; fr_dca = fn_dca / float(infl_def[-1])

    fg_ls = float(val_ls.iloc[-1]); gpnl_ls = fg_ls - tot_cap; tax_ls = max(0.0, gpnl_ls * params.tax_rate)
    fn_ls = fg_ls - tax_ls; npnl_ls = fn_ls - tot_cap; fr_ls = fn_ls / float(infl_def[-1])

    cf_dca = [(ts, -float(amt)) for ts, amt in gross_dca[gross_dca > 0].items()] + [(adj_prices.index[-1], fn_dca)]
    xirr_dca = compute_xirr(cf_dca)
    cf_ls = [(d0, -tot_cap), (adj_prices.index[-1], fn_ls)]
    xirr_ls = compute_xirr(cf_ls)

    daily_df = pd.DataFrame({
        "Market_Price": adj_prices, "Gross_Contribution": gross_dca, "Brokerage_Fee": fees_dca,
        "Net_Contribution": net_dca, "Shares_Bought": shares_bought, "Cum_Shares": cum_shares,
        "Cum_Invested": cum_inv, "Cum_Fees": cum_fees, "PRU": pru,
        "Portfolio_Gross_Value": val_dca, "Portfolio_Real_Gross_Value": real_val_dca,
        "Latent_PnL_Gross": val_dca - cum_inv, "Drawdown_DCA": dd_dca,
        "Drawdown_Underlying": dd_ass, "LumpSum_Gross_Value": val_ls,
        "LumpSum_Real_Gross_Value": real_val_ls, "Drawdown_LumpSum": dd_ls,
    }, index=adj_prices.index)
    daily_df.index.name = "Date"

    m_dca = BacktestMetrics(tot_cap, fg_dca, fn_dca, fr_dca, gpnl_dca, npnl_dca, npnl_dca / tot_cap if tot_cap > 0 else 0.0, tax_dca, float(cum_fees.iloc[-1]), xirr_dca, u_cagr, float((fn_dca / tot_cap) ** (1.0 / tot_yrs) - 1.0), vol_dca, (xirr_dca - params.risk_free_rate) / vol_dca if vol_dca > 1e-9 else 0.0, (xirr_dca - params.risk_free_rate) / dn_std_dca if dn_std_dca > 1e-9 else 0.0, max_dd_dca, uw_dca, float(cum_shares.iloc[-1]), float(pru.iloc[-1]), float(adj_prices.iloc[-1]), int((gross_dca > 0).sum()))
    m_ls = BacktestMetrics(tot_cap, fg_ls, fn_ls, fr_ls, gpnl_ls, npnl_ls, npnl_ls / tot_cap if tot_cap > 0 else 0.0, tax_ls, ls_fee, xirr_ls, u_cagr, float((fn_ls / tot_cap) ** (1.0 / tot_yrs) - 1.0), vol_ls, (xirr_ls - params.risk_free_rate) / vol_ls if vol_ls > 1e-9 else 0.0, (xirr_ls - params.risk_free_rate) / dn_std_ls if dn_std_ls > 1e-9 else 0.0, max_dd_ls, uw_ass, ls_shares, float(adj_prices.iloc[0]) * (tot_cap / (tot_cap - ls_fee)), float(adj_prices.iloc[-1]), 1)
    return AssetBacktestResult(asset_label, ticker, eff_curr, daily_df, m_dca, m_ls, cf_dca, cf_ls)

def build_annual_summary_table(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.copy(); df["Year"] = df.index.year
    records = []; prev_val = 0.0
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
        records.append({
            "Année": int(yr), "Versements de l'année (€)": round(flows, 2), "Cumul Versé (€)": round(cum_i, 2),
            "Valeur Fin d'Année (€)": round(end_v, 2), "Gain / Perte de l'année (€)": round(gain, 2),
            "Performance Annuelle (%)": round((gain / denom * 100.0) if denom > 0 else 0.0, 2),
            "Plus/Moins-Value Latente (€)": round(lat, 2), "Plus/Moins-Value Latente (%)": round((lat / cum_i * 100.0) if cum_i > 0 else 0.0, 2),
            "PRU Fin d'Année (€)": round(float(sub["PRU"].iloc[-1]), 2), "Prix Marché Fin d'Année (€)": round(float(sub["Market_Price"].iloc[-1]), 2),
        })
        prev_val = end_v
    return pd.DataFrame(records)

def run_monte_carlo_projection(res: AssetBacktestResult, params: SimulationParams, horizon_years: int, n_trajectories: int = 500) -> pd.DataFrame:
    hist_df = res.daily_df
    b_prices = hist_df["Market_Price"][hist_df.index.weekday < 5]
    log_r = np.log(b_prices / b_prices.shift(1)).dropna()
    mu, sigma = float(log_r.mean() * 252.0), float(log_r.std() * np.sqrt(252.0))
    n_m, dt = horizon_years * 12, 1.0 / 12.0
    mult = (52.0 / 12.0) if params.dca_frequency == "Hebdomadaire" else (1.0 if params.dca_frequency == "Mensuelle" else 1.0 / 3.0)
    m_gross = params.dca_amount * mult
    m_net = max(0.0, m_gross - compute_order_fee(params.dca_amount, params) * mult)
    rng = np.random.default_rng(seed=42)
    growth = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_m, n_trajectories)))
    w_paths = np.zeros((n_m + 1, n_trajectories)); w_paths[0, :] = float(hist_df["Portfolio_Gross_Value"].iloc[-1])
    inv_path = np.zeros(n_m + 1); inv_path[0] = float(hist_df["Cum_Invested"].iloc[-1])
    for m in range(1, n_m + 1):
        w_paths[m, :] = (w_paths[m - 1, :] + m_net) * growth[m - 1, :]
        inv_path[m] = inv_path[m - 1] + m_gross
    f_dates = pd.date_range(start=hist_df.index[-1], periods=n_m + 1, freq="ME")
    return pd.DataFrame({"Date": f_dates, "Cum_Invested": inv_path, "P10": np.percentile(w_paths, 10, axis=1), "P50": np.percentile(w_paths, 50, axis=1), "P90": np.percentile(w_paths, 90, axis=1)}).set_index("Date")

def apply_dark_layout(fig: go.Figure, title: str, height: int = 460) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=17, color="#F8FAFC"), x=0.01), template="plotly_dark", paper_bgcolor="rgba(15,23,42,0.0)", plot_bgcolor="rgba(15,23,42,0.45)", height=height, margin=dict(l=20, r=20, t=55, b=20), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0))
    return fig

def render_sidebar() -> Tuple[List[Tuple[str, str, str, bool]], Optional[io.BytesIO], SimulationParams]:
    st.sidebar.title("⚙️ Paramètres Quant")
    default_ch = ["S&P 500 - ETF SPY (Div. Réinvestis)", "Amundi MSCI World PEA (CW8.PA - EUR)", "CAC 40 - Indice (^FCHI)"]
    sel_lbls = st.sidebar.multiselect("Indices & ETFs préconfigurés", options=list(PRECONFIGURED_ASSETS.keys()), default=default_ch)
    custom_raw = st.sidebar.text_input("Tickers Yahoo personnalisés", value="", placeholder="Ex: ASML.AS, AI.PA, SU.PA, NVDA")
    up_csv = st.sidebar.file_uploader("Import CSV personnalisé", type=["csv"])
    conv_fx = st.sidebar.toggle("Convertir USD en EUR (EURUSD=X)", value=True)
    assets = [(l, *PRECONFIGURED_ASSETS[l]) for l in sel_lbls]
    if custom_raw.strip():
        for rt in custom_raw.split(","):
            ct = rt.strip().upper()
            if ct and not any(ct == x[1] for x in assets):
                assets.append((f"Custom ({ct})", ct, "USD", ct.startswith("^")))
    st.sidebar.divider()
    today = date.today()
    c1, c2 = st.sidebar.columns(2)
    start_d = c1.date_input("Début", value=today - timedelta(days=int(15 * 365.25)), min_value=date(1980, 1, 1), max_value=today - timedelta(days=30))
    end_d = c2.date_input("Fin", value=today, min_value=start_d + timedelta(days=30), max_value=today)
    st.sidebar.divider()
    init_cap = st.sidebar.number_input("Capital initial (Jour 1) en €", min_value=0.0, max_value=10000000.0, value=5000.0, step=500.0)
    dca_amt = st.sidebar.number_input("Versement programmé (DCA) en €", min_value=0.0, max_value=1000000.0, value=400.0, step=50.0)
    dca_freq = st.sidebar.selectbox("Fréquence du DCA", options=["Mensuelle", "Hebdomadaire", "Trimestrielle"])
    dca_rule = st.sidebar.selectbox("Jour du mois", options=["1er du mois", "15 du mois", "Dernier jour ouvré"]) if dca_freq != "Hebdomadaire" else "1er du mois"
    st.sidebar.divider()
    fee_mode = st.sidebar.radio("Frais de courtage", options=["Pourcentage (%)", "Fixe (€)"], horizontal=True)
    if fee_mode == "Pourcentage (%)":
        cf1, cf2 = st.sidebar.columns(2)
        f_pct = cf1.number_input("Courtage (%)", 0.0, 5.0, 0.10, 0.05)
        f_min = cf2.number_input("Min / ordre (€)", 0.0, 50.0, 0.0, 0.5)
        f_fix = 0.0
    else:
        f_fix = st.sidebar.number_input("Frais fixe (€)", 0.0, 100.0, 1.0, 0.5)
        f_pct, f_min = 0.0, 0.0
    ter_in = st.sidebar.number_input("Frais de gestion annuels ETF (TER %)", 0.0, 3.0, 0.20, 0.05)
    ter_etf = st.sidebar.checkbox("Déduire aussi ce TER sur les ETFs cotés", value=False)
    tax_opts = {"PEA (> 5 ans) — 17,2 %": 0.172, "CTO (Flat Tax) — 30,0 %": 0.30, "Brut d'impôt — 0,0 %": 0.0}
    tax_name = st.sidebar.selectbox("Fiscalité à la sortie", options=list(tax_opts.keys()))
    cm1, cm2 = st.sidebar.columns(2)
    infl_in = cm1.number_input("Inflation (%/an)", 0.0, 15.0, 2.0, 0.2)
    rf_in = cm2.number_input("Taux sans risque (%)", 0.0, 10.0, 2.5, 0.25)
    return assets, up_csv, SimulationParams(start_d, end_d, float(init_cap), float(dca_amt), dca_freq, dca_rule, fee_mode, float(f_pct) / 100.0, float(f_min), float(f_fix), float(ter_in) / 100.0, ter_etf, tax_name, tax_opts[tax_name], float(infl_in) / 100.0, float(rf_in) / 100.0, conv_fx)

def main() -> None:
    st.title("📊 QuantDCA & Lump Sum Backtester Pro")
    assets, up_csv, params = render_sidebar()
    if not assets and up_csv is None:
        st.warning("👈 Sélectionnez au moins un actif dans la barre latérale.")
        st.stop()
    fx_s = fetch_eurusd_fx_series(params.start_date, params.end_date) if params.convert_usd_to_eur else None
    results: List[AssetBacktestResult] = []
    with st.spinner("Calculs quantitatifs en cours..."):
        if up_csv is not None:
            cs, cc, ce = parse_custom_csv(up_csv, params.start_date, params.end_date)
            if ce: st.error(ce)
            elif cs is not None: results.append(run_asset_backtest(f"CSV ({up_csv.name})", "CSV", cs, cc, False, params, fx_s))
        for lbl, tck, d_curr, is_b in assets:
            sp, det_c, err = fetch_yahoo_series(tck, params.start_date, params.end_date)
            if err or sp is None: st.error(err or f"Erreur sur {tck}"); continue
            results.append(run_asset_backtest(lbl, tck, sp, "EUR" if d_curr == "EUR" else det_c, is_b, params, fx_s))
    if not results:
        st.error("Aucun actif disponible sur cette période."); st.stop()

    primary_res = next(r for r in results if r.asset_label == st.selectbox("🎯 Actif principal analysé :", [r.asset_label for r in results])) if len(results) > 1 else results[0]
    m_dca, m_ls, df = primary_res.dca_metrics, primary_res.lump_sum_metrics, primary_res.daily_df
    tab1, tab2, tab3, tab4 = st.tabs(["📈 1. Vue d'ensemble", "🛡️ 2. Risque & PRU", "⚔️ 3. Multi-Indices & Lump Sum", "🗂️ 4. Données Annuelles & Export"])

    with tab1:
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Capital Investi", f"{m_dca.total_invested:,.0f} €", f"Frais: {m_dca.total_brokerage_fees:,.0f} €", delta_color="off")
        k2.metric("Valeur Finale Brute", f"{m_dca.final_gross_value:,.0f} €", f"{(m_dca.gross_pnl / m_dca.total_invested) * 100:+.1f} %")
        k3.metric("Valeur Finale Nette", f"{m_dca.final_net_value:,.0f} €", f"Impôt: -{m_dca.tax_paid:,.0f} €")
        k4.metric("Plus-Value Nette", f"{m_dca.net_pnl:+,.0f} €", f"{m_dca.net_return_pct * 100:+.2f} %")
        k5.metric("TRI Annualisé (XIRR)", f"{m_dca.xirr * 100:.2f} %/an", f"CAGR Actif: {m_dca.underlying_cagr * 100:.2f} %")
        k6.metric("Max Drawdown", f"{m_dca.max_drawdown * 100:.2f} %", f"{m_dca.max_underwater_days} j max", delta_color="inverse")

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df.index, y=df["Cum_Invested"], name="Capital Investi", line=dict(color="#94A3B8", width=2, shape="hv"), fill="tozeroy", fillcolor="rgba(148,163,184,0.12)"))
        fig1.add_trace(go.Scatter(x=df.index, y=df["Portfolio_Gross_Value"], name="Valorisation Portefeuille", line=dict(color="#00D4FF", width=2.8), fill="tonexty", fillcolor="rgba(0,212,255,0.12)"))
        fig1.add_trace(go.Scatter(x=df.index, y=df["Portfolio_Real_Gross_Value"], name="Pouvoir d'Achat Réel (Euros constants)", line=dict(color="#F59E0B", width=2, dash="dot")))
        st.plotly_chart(apply_dark_layout(fig1, f"Valorisation, Versements et Pouvoir d'Achat — {primary_res.asset_label}"), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pouvoir d'Achat Net Final", f"{m_dca.final_real_net_value:,.0f} €")
        c2.metric("Volatilité Annualisée", f"{m_dca.annualized_volatility * 100:.2f} %")
        c3.metric("Ratio de Sharpe", f"{m_dca.sharpe_ratio:.2f}")
        c4.metric("Ratio de Sortino", f"{m_dca.sortino_ratio:.2f}")

        with st.expander("🔮 Projection Future Monte Carlo (500 trajectoires)"):
            hz = st.slider("Horizon (Années)", 3, 30, 10)
            mc_df = run_monte_carlo_projection(primary_res, params, hz)
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=mc_df.index, y=mc_df["P90"], name="Optimiste (P90)", line=dict(color="#10B981", width=1.5)))
            fig_mc.add_trace(go.Scatter(x=mc_df.index, y=mc_df["P10"], name="Prudent (P10)", line=dict(color="#EF4444", width=1.5), fill="tonexty", fillcolor="rgba(0,212,255,0.12)"))
            fig_mc.add_trace(go.Scatter(x=mc_df.index, y=mc_df["P50"], name="Médian (P50)", line=dict(color="#00D4FF", width=3)))
            fig_mc.add_trace(go.Scatter(x=mc_df.index, y=mc_df["Cum_Invested"], name="Capital Projeté", line=dict(color="#94A3B8", width=2, dash="dash")))
            st.plotly_chart(apply_dark_layout(fig_mc, f"Projection Monte Carlo à +{hz} ans"), use_container_width=True)

    with tab2:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("PRU Final Moyen", f"{m_dca.final_pru:,.2f} €")
        r2.metric("Prix de Marché Final", f"{m_dca.final_market_price:,.2f} €", f"{((m_dca.final_market_price / m_dca.final_pru) - 1.0) * 100:+.2f} % vs PRU")
        r3.metric("Total Parts Acquises", f"{m_dca.total_shares:,.4f}")
        r4.metric("Durée Max Underwater", f"{m_dca.max_underwater_days} jours")

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=df.index, y=df["Drawdown_Underlying"] * 100, name="DD Sous-Jacent (%)", line=dict(color="rgba(248,113,113,0.5)", width=1.4, dash="dot")))
        fig_dd.add_trace(go.Scatter(x=df.index, y=df["Drawdown_DCA"] * 100, name="DD Portefeuille DCA (%)", line=dict(color="#EF4444", width=2.2), fill="tozeroy", fillcolor="rgba(239,68,68,0.25)"))
        st.plotly_chart(apply_dark_layout(fig_dd, "Graphique Underwater (Drawdown en % depuis le dernier plus haut)"), use_container_width=True)

        fig_pru = go.Figure()
        fig_pru.add_trace(go.Scatter(x=df.index, y=df["Market_Price"], name="Prix du Marché", line=dict(color="#10B981", width=2.2)))
        fig_pru.add_trace(go.Scatter(x=df.index, y=df["PRU"], name="PRU Moyen Pondéré", line=dict(color="#F59E0B", width=2.6)))
        st.plotly_chart(apply_dark_layout(fig_pru, "Prix du Marché vs Évolution du PRU (Prix de Revient Unitaire)"), use_container_width=True)

    with tab3:
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=df.index, y=df["Cum_Invested"], name="Capital Versé", line=dict(color="#94A3B8", width=2, dash="dash", shape="hv")))
        for idx, r in enumerate(results):
            fig_m.add_trace(go.Scatter(x=r.daily_df.index, y=r.daily_df["Portfolio_Gross_Value"], name=f"{r.asset_label} (TRI: {r.dca_metrics.xirr * 100:.2f}%)", line=dict(color=COLOR_PALETTE[idx % len(COLOR_PALETTE)], width=2.5)))
        st.plotly_chart(apply_dark_layout(fig_m, "Comparaison Multi-Actifs à Stratégie DCA Identique"), use_container_width=True)

        comp = [{"Actif": r.asset_label, "Capital Versé (€)": round(r.dca_metrics.total_invested, 0), "Valeur Nette DCA (€)": round(r.dca_metrics.final_net_value, 0), "TRI DCA (%)": round(r.dca_metrics.xirr * 100, 2), "Valeur Nette Lump Sum (€)": round(r.lump_sum_metrics.final_net_value, 0), "TRI Lump Sum (%)": round(r.lump_sum_metrics.xirr * 100, 2), "Max DD DCA (%)": round(r.dca_metrics.max_drawdown * 100, 2), "Sharpe": round(r.dca_metrics.sharpe_ratio, 2)} for r in results]
        st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

        fig_ls = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.68, 0.32])
        fig_ls.add_trace(go.Scatter(x=df.index, y=df["Cum_Invested"], name="Cumul DCA", line=dict(color="#64748B", width=1.8, dash="dot", shape="hv")), row=1, col=1)
        fig_ls.add_trace(go.Scatter(x=df.index, y=df["Portfolio_Gross_Value"], name="Stratégie DCA", line=dict(color="#00D4FF", width=2.8)), row=1, col=1)
        fig_ls.add_trace(go.Scatter(x=df.index, y=df["LumpSum_Gross_Value"], name="Stratégie 100% Lump Sum (J1)", line=dict(color="#A855F7", width=2.5)), row=1, col=1)
        fig_ls.add_trace(go.Scatter(x=df.index, y=df["Drawdown_DCA"] * 100, name="DD DCA (%)", line=dict(color="#00D4FF", width=1.8)), row=2, col=1)
        fig_ls.add_trace(go.Scatter(x=df.index, y=df["Drawdown_LumpSum"] * 100, name="DD Lump Sum (%)", line=dict(color="#A855F7", width=1.8)), row=2, col=1)
        st.plotly_chart(apply_dark_layout(fig_ls, f"Duel DCA vs Lump Sum — {primary_res.asset_label}", height=560), use_container_width=True)

    with tab4:
        ann_df = build_annual_summary_table(df)
        st.dataframe(ann_df, use_container_width=True, hide_index=True)
        ce1, ce2 = st.columns(2)
        ce1.download_button("📥 Télécharger Valorisation Quotidienne (CSV)", df.to_csv(index=True).encode("utf-8"), f"daily_{primary_res.ticker}.csv", "text/csv", use_container_width=True)
        ce2.download_button("📥 Télécharger Bilan Annuel (CSV)", ann_df.to_csv(index=False).encode("utf-8"), f"annual_{primary_res.ticker}.csv", "text/csv", use_container_width=True)

if __name__ == "__main__":
    main()
