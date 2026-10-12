"""20-year production forecast (2008-04-01 .. 2028-01-01) and its backtest.

Why a dedicated forecast model?
  The simulated cases only cover 1998-2008, so the proxy cannot be queried beyond
  2008. Production after 2008 is forecast with a physics-based decline model whose
  form was chosen by backtesting on the 100 simulated cases (truncate history,
  forecast, compare with the simulator).

Chosen model ("oil-cut power law"):
  * Wells run on liquid-rate control, so field liquid rate q_L is extrapolated from
    its recent trend (exponential fit of the last 9 quarters).
  * Oil cut f_o = q_o / q_L follows a power law of cumulative liquid x:
        f_o(x) = a * x^(-beta)
    which is the late-life Buckley-Leverett behaviour of a waterflood (oil cut decays
    as a power of pore volumes produced). It is fitted on the last 9 quarters.
  * water = q_L - oil, gas = GOR * oil (GOR is constant: undersaturated oil).

History-match coupling:
  The observed late-time exponent (beta ~ 1.3) is steeper than in any history-matched
  realisation (beta ~ 0.75-0.8): the 4-parameter model has structural error. The
  forecast is therefore anchored on the observed trend, and each of the 10 matched
  realisations k perturbs it by its own deviation from the ensemble mean:
        beta_k = beta_obs + (beta_k_proxy - mean_j beta_j_proxy)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

FORECAST_DATES = pd.date_range(C.FORECAST_START, C.FORECAST_END, freq="QS-JAN")   # 80 template rows


def _mid_cum(rate, days, start=0.0):
    """Cumulative at mid-quarter for a series of quarterly average rates."""
    cum_end = start + np.cumsum(rate * days)
    return cum_end - 0.5 * rate * days


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def fit_powerlaw_oilcut(oil, water, days, n_fit, cumliq0=0.0):
    """Fit ln(f_o) = ln(a) - beta ln(x) on the last n_fit quarters. Returns (ln_a, beta)."""
    liq = oil + water
    x = _mid_cum(liq, days, cumliq0)
    sl = slice(len(oil) - n_fit, len(oil))
    slope, ln_a = np.polyfit(np.log(x[sl]), np.log(oil[sl] / liq[sl]), 1)
    return ln_a, -slope


def fit_liquid_trend(liq, days, n_fit):
    """Exponential trend of liquid rate over the last n_fit quarters -> (q_last_fit, decline per day)."""
    t = np.cumsum(days) - 0.5 * days
    sl = slice(len(liq) - n_fit, len(liq))
    slope, icpt = np.polyfit(t[sl], np.log(liq[sl]), 1)
    return icpt, slope, t[-1] + 0.5 * days[-1]          # also return time at end of history


def liquid_projection(liq_hist, days_hist, days_fc, mode=C.FC_LIQUID_MODE, n_fit=C.FC_FIT_QUARTERS):
    if mode == "constant":
        return np.full(len(days_fc), liq_hist[-1])
    icpt, slope, t_end = fit_liquid_trend(liq_hist, days_hist, n_fit)
    t_mid = t_end + np.cumsum(days_fc) - 0.5 * days_fc
    return np.exp(icpt + slope * t_mid)


def powerlaw_forecast(ln_a, beta, liq_fc, days_fc, cumliq_start):
    x = _mid_cum(liq_fc, days_fc, cumliq_start)
    fo = np.clip(np.exp(ln_a - beta * np.log(x)), 0.0, 1.0)
    oil = liq_fc * fo
    return oil, liq_fc - oil


# --------------------------------------------------------------------------
# Observed history incl. Q1-2008 and the anchored field forecast
# --------------------------------------------------------------------------
def observed_series(q, obs_table):
    """Quarterly observed rates 1998-Q1 .. 2008-Q1 (41 quarters) + cumulative at 2008-04-01."""
    o = q.rate("Oil").loc["observed"].values.astype(float)
    w = q.rate("Water").loc["observed"].values.astype(float)
    g = q.rate("Gas").loc["observed"].values.astype(float)
    last = obs_table.iloc[-1]                                   # Q1-2008 (forward labelled)
    days_q1 = float((pd.Timestamp(C.FORECAST_START) - last["Date"]).days)
    o = np.append(o, last["oil_rate"])
    w = np.append(w, last["water_rate"])
    g = np.append(g, last["gas_rate"])
    days = np.append(q.days, days_q1)
    cum_anchor = {ph: float(q.cum[ph].loc["observed"].iloc[-1]) + r[-1] * days_q1
                  for ph, r in (("Oil", o), ("Water", w), ("Gas", g))}
    return {"oil": o, "water": w, "gas": g, "days": days, "cum_anchor": cum_anchor,
            "quarter_start": list(q.dates[:-1]) + [last["Date"]]}


def forecast_days():
    d = FORECAST_DATES
    return np.diff(d.values).astype("timedelta64[D]").astype(float)     # 79 quarters


def member_beta_shift(proxy_oil, proxy_water, hist_days, n_fit=C.FC_FIT_QUARTERS - 1):
    """beta of each matched realisation (from its proxy curves) minus the ensemble mean."""
    betas = np.array([fit_powerlaw_oilcut(o, w, hist_days, n_fit)[1] for o, w in zip(proxy_oil, proxy_water)])
    return betas - betas.mean(), betas


def field_forecast(obs, beta_shift=(0.0,), liquid_mode=C.FC_LIQUID_MODE, n_fit=C.FC_FIT_QUARTERS):
    """Forecast for each beta shift. Returns list of DataFrames indexed by FORECAST_DATES."""
    o, w, days = obs["oil"], obs["water"], obs["days"]
    _, beta_obs = fit_powerlaw_oilcut(o, w, days, n_fit)
    liq_hist = o + w
    dfc = forecast_days()
    liq_fc = liquid_projection(liq_hist, days, dfc, liquid_mode, n_fit)
    cumliq_anchor = float(np.sum(liq_hist * days))
    sl = slice(len(o) - n_fit, len(o))
    gor = float(np.sum(obs["gas"][sl] * days[sl]) / np.sum(o[sl] * days[sl]))
    # keep the fitted curve continuous with the last observed oil cut
    x_last = _mid_cum(liq_hist, days)[-1]
    out = []
    for dbeta in beta_shift:
        beta = beta_obs + dbeta
        ln_a_k = np.log(o[-1] / liq_hist[-1]) + beta * np.log(x_last)
        oil, wat = powerlaw_forecast(ln_a_k, beta, liq_fc, dfc, cumliq_anchor)
        gas = gor * oil
        base = obs["cum_anchor"] if C.CUMULATIVE_FROM_START else {"Oil": o[-1] * days[-1],
                                                                  "Water": w[-1] * days[-1],
                                                                  "Gas": obs["gas"][-1] * days[-1]}
        cum = {
            "Oil": base["Oil"] + np.concatenate([[0], np.cumsum(oil * dfc)]),
            "Water": base["Water"] + np.concatenate([[0], np.cumsum(wat * dfc)]),
            "Gas": base["Gas"] + np.concatenate([[0], np.cumsum(gas * dfc)]),
        }
        rates = lambda r, first: np.concatenate([[first], r])
        df = pd.DataFrame({
            "date": FORECAST_DATES,
            "gas_cum_mscf": cum["Gas"], "oil_cum_stb": cum["Oil"], "water_cum_stb": cum["Water"],
            "oil_rate_stbd": rates(oil, np.nan), "water_rate_stbd": rates(wat, np.nan),
            "gas_rate_mscfd": rates(gas, np.nan), "liquid_rate_stbd": rates(liq_fc, np.nan),
        })
        # rate columns describe the quarter *ending* at each date (first row = Q1-2008 observed)
        df.loc[0, ["oil_rate_stbd", "water_rate_stbd", "gas_rate_mscfd", "liquid_rate_stbd"]] = \
            [o[-1], w[-1], obs["gas"][-1], liq_hist[-1]]
        df["water_cut"] = df["water_rate_stbd"] / df["liquid_rate_stbd"]
        df.attrs.update(beta=beta, beta_obs=beta_obs, gor=gor)
        out.append(df)
    return out


def sensitivity(obs):
    """Scenario spread for the dashboard: fit window x liquid assumption."""
    rows = []
    for n_fit in (6, 9, 12):
        for mode in ("trend", "constant"):
            df = field_forecast(obs, (0.0,), mode, n_fit)[0]
            rows.append({"fit_quarters": n_fit, "liquid_mode": mode, "beta": df.attrs["beta"],
                         "oil_cum_2028_mmstb": df["oil_cum_stb"].iloc[-1] / 1e6,
                         "water_cum_2028_mmstb": df["water_cum_stb"].iloc[-1] / 1e6,
                         "oil_rate_2028_stbd": df["oil_rate_stbd"].iloc[-1]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Backtest on the simulated cases
# --------------------------------------------------------------------------
def _m_powerlaw(O, W, d, Lf, df, n):
    """Power law anchored on the last observed oil cut (the method used for the field)."""
    ln_a, beta = fit_powerlaw_oilcut(O, W, d, n)
    L = O + W
    ln_a = np.log(O[-1] / L[-1]) + beta * np.log(_mid_cum(L, d)[-1])
    return powerlaw_forecast(ln_a, beta, Lf, df, np.sum(L * d))[0]


def _m_powerlaw_regression(O, W, d, Lf, df, n):
    ln_a, beta = fit_powerlaw_oilcut(O, W, d, n)
    return powerlaw_forecast(ln_a, beta, Lf, df, np.sum((O + W) * d))[0]


def _m_wor_np(O, W, d, Lf, df, n):
    Np_mid = _mid_cum(O, d)
    sl = slice(len(O) - n, len(O))
    b = np.polyfit(Np_mid[sl], np.log(W[sl] / O[sl]), 1)
    co, oil = np.sum(O * d), np.zeros(len(Lf))
    for i in range(len(Lf)):
        q = 0.5 * Lf[i]
        for _ in range(6):
            q = Lf[i] / (1 + np.exp(np.polyval(b, co + 0.5 * q * df[i])))
        oil[i], co = q, co + q * df[i]
    return oil


def _m_arps_exp(O, W, d, Lf, df, n):
    t = np.cumsum(d) - 0.5 * d
    sl = slice(len(O) - n, len(O))
    b = np.polyfit(t[sl], np.log(O[sl]), 1)
    tf = t[-1] + 0.5 * d[-1] + np.cumsum(df) - 0.5 * df
    return np.minimum(np.exp(np.polyval(b, tf)), Lf)


def _m_logit_logx(O, W, d, Lf, df, n):
    L = O + W
    x = _mid_cum(L, d)
    sl = slice(len(O) - n, len(O))
    fw = W[sl] / L[sl]
    b = np.polyfit(np.log(x[sl]), np.log(fw / (1 - fw)), 1)
    xf = _mid_cum(Lf, df, np.sum(L * d))
    return Lf * (1 - 1 / (1 + np.exp(-np.polyval(b, np.log(xf)))))


BACKTEST_METHODS = {
    "Oil-cut power law, anchored (chosen)": _m_powerlaw,
    "Oil-cut power law, regression intercept": _m_powerlaw_regression,
    "WOR vs cum. oil (semilog)": _m_wor_np,
    "Arps exponential decline": _m_arps_exp,
    "Logit water cut vs log cum. liquid": _m_logit_logx,
}


def backtest(q, cut_quarters=(24, 28, 32), n_fit=8):
    """Truncate each simulated case at quarter T, forecast to 2008, compare to simulator.

    Future liquid rate is taken from the simulator so that the test isolates the
    oil-cut model (liquid follows the well controls, which are known in the history).
    """
    O = q.rate("Oil").loc[q.cases].values.astype(float)
    W = q.rate("Water").loc[q.cases].values.astype(float)
    d = q.days
    rows, curves = [], []
    for T in cut_quarters:
        # regime indicator: the power law assumes stable well operation over the horizon
        liq_change = 100 * np.mean((O[:, -1] + W[:, -1]) / (O[:, T] + W[:, T]) - 1)
        for name, fn in BACKTEST_METHODS.items():
            e = []
            for i in range(len(O)):
                Lf = O[i, T:] + W[i, T:]
                oil = fn(O[i, :T], W[i, :T], d[:T], Lf, d[T:], n_fit)
                wat = Lf - oil
                to, tw = O[i, T:], W[i, T:]
                e.append([(oil @ d[T:] - to @ d[T:]) / (to @ d[T:]),
                          (wat @ d[T:] - tw @ d[T:]) / (tw @ d[T:]),
                          (oil[-1] - to[-1]) / to[-1]])
                if T == cut_quarters[-1] and i in (0, 49, 99):
                    curves.append({"case": q.cases[i], "method": name, "T": T, "pred": oil, "true": to})
            e = np.array(e)
            rows.append({"cutoff": str(q.dates[T].date()), "horizon_quarters": 40 - T,
                         "liquid_change_over_horizon_%": liq_change, "method": name,
                         "MAPE_cum_oil_%": 100 * np.abs(e[:, 0]).mean(),
                         "MAPE_cum_water_%": 100 * np.abs(e[:, 1]).mean(),
                         "MAPE_final_oil_rate_%": 100 * np.abs(e[:, 2]).mean(),
                         "bias_cum_oil_%": 100 * e[:, 0].mean(),
                         "bias_final_oil_rate_%": 100 * e[:, 2].mean()})
    return pd.DataFrame(rows), curves


# --------------------------------------------------------------------------
# Simple economics (illustrative assumptions, see config)
# --------------------------------------------------------------------------
def economics(df, oil_price=C.ECON["oil_price_usd_bbl"], gas_price=C.ECON["gas_price_usd_mscf"],
              opex_fixed_per_year=C.ECON["opex_fixed_usd_year"], opex_per_bbl_liquid=C.ECON["opex_usd_bbl_liquid"],
              discount=C.ECON["discount_rate"]):
    d = df.iloc[1:].copy()
    days = forecast_days()
    rev = (d["oil_rate_stbd"] * oil_price + d["gas_rate_mscfd"] * gas_price) * days
    opex = opex_fixed_per_year * days / 365.25 + d["liquid_rate_stbd"] * opex_per_bbl_liquid * days
    ncf = (rev - opex).values
    t_years = (np.cumsum(days) - 0.5 * days) / 365.25
    disc = (1 + discount) ** -t_years
    neg = np.where(ncf < 0)[0]
    limit_date = d["date"].iloc[neg[0]] if len(neg) else pd.NaT
    keep = np.arange(len(ncf)) < (neg[0] if len(neg) else len(ncf))
    return {"npv10_musd": float(np.sum(ncf[keep] * disc[keep]) / 1e6),
            "undiscounted_cf_musd": float(np.sum(ncf[keep]) / 1e6),
            "economic_limit": limit_date,
            "net_cash_flow_musd": ncf / 1e6}
