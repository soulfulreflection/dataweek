"""STEP 4 - 20-year forecast (2008-04-01 .. 2028-01-01) for the 10 history-matched cases.

1. Backtest candidate forecasting methods on the 100 simulated cases.
2. Fit the chosen oil-cut power law on the observed field (last 9 quarters).
3. Give each history-matched case its own decline exponent:
       beta_k = beta_obs + (beta_k_proxy - mean beta_proxy)
4. Compare with the "uncorrected" forecast that would trust the proxy exponents.
5. Sensitivity (fit window x liquid-rate assumption) and simple economics.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dwc import config as C, data, forecast as F, plots as P  # noqa: E402
from dwc.proxy import PCAGPProxy  # noqa: E402


def main():
    P.style()
    q = data.build_quarterly()
    obs_tab = data.load_observed_table()

    # ---- 1. backtest
    bt, curves = F.backtest(q)
    bt.to_csv(C.METRICS / "forecast_backtest.csv", index=False)
    print(bt.round(2).to_string(index=False))

    # ---- 2./3. forecasts
    obs = F.observed_series(q, obs_tab)
    theta = np.load(C.MODELS / "hm_selected_theta.npy")
    proxy = PCAGPProxy.load(C.MODELS / "proxy_final.joblib")
    po, pw = proxy.predict(theta)
    shift, beta_proxy = F.member_beta_shift(po, pw, q.days)
    fcs = F.field_forecast(obs, shift)
    beta_obs = fcs[0].attrs["beta_obs"]
    print(f"  observed beta {beta_obs:.3f}; proxy betas {beta_proxy.round(3)}; GOR {fcs[0].attrs['gor']:.4f}")
    uncorrected = F.field_forecast(obs, beta_proxy - beta_obs)

    rows, long = [], []
    for k, (df, dfu) in enumerate(zip(fcs, uncorrected)):
        eco = F.economics(df)
        rows.append({"deliverable_case": f"Case {k + 1}", **dict(zip(C.PARAM_SHORT, theta[k])),
                     "beta_proxy": beta_proxy[k], "beta_forecast": df.attrs["beta"],
                     "oil_cum_2028_mmstb": df["oil_cum_stb"].iloc[-1] / 1e6,
                     "water_cum_2028_mmstb": df["water_cum_stb"].iloc[-1] / 1e6,
                     "gas_cum_2028_bscf": df["gas_cum_mscf"].iloc[-1] / 1e6,
                     "oil_rate_2028_stbd": df["oil_rate_stbd"].iloc[-1],
                     "water_cut_2028": df["water_cut"].iloc[-1],
                     "oil_cum_2028_uncorrected_mmstb": dfu["oil_cum_stb"].iloc[-1] / 1e6,
                     "npv10_musd": eco["npv10_musd"], "economic_limit": eco["economic_limit"]})
        long.append(df.assign(deliverable_case=f"Case {k + 1}"))
    summary = pd.DataFrame(rows)
    summary.to_csv(C.METRICS / "forecast_case_summary.csv", index=False)
    pd.concat(long).to_csv(C.METRICS / "forecast_cases_long.csv", index=False)
    print(summary.drop(columns="economic_limit").round(3).to_string(index=False))
    sens = F.sensitivity(obs)
    sens.to_csv(C.METRICS / "forecast_sensitivity.csv", index=False)
    print(sens.round(3).to_string(index=False))

    # ---- Figures
    t_hist = obs["quarter_start"]
    hist_oil, hist_wat = obs["oil"], obs["water"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for k, (df, dfu) in enumerate(zip(fcs, uncorrected)):
        lab = "10 forecast cases" if k == 0 else None
        ax[0].plot(df["date"], df["oil_rate_stbd"], color=P.BLUE, lw=1.1, label=lab)
        ax[1].plot(df["date"], df["water_rate_stbd"], color=P.BLUE, lw=1.1, label=lab)
        ax[2].plot(df["date"], df["oil_cum_stb"] / 1e6, color=P.BLUE, lw=1.1, label=lab)
        lab_u = "Uncorrected (proxy exponents)" if k == 0 else None
        ax[0].plot(dfu["date"], dfu["oil_rate_stbd"], color=P.ORANGE, lw=0.9, ls="--", label=lab_u)
        ax[2].plot(dfu["date"], dfu["oil_cum_stb"] / 1e6, color=P.ORANGE, lw=0.9, ls="--", label=lab_u)
    # history is plotted at quarter end so it lines up with the forecast's rate convention
    t_end = list(t_hist[1:]) + [pd.Timestamp(C.FORECAST_START)]
    ax[0].plot(t_end, hist_oil, "o", color=P.INK, ms=3, label="Observed")
    ax[1].plot(t_end, hist_wat, "o", color=P.INK, ms=3, label="Observed")
    cum_o = np.cumsum(hist_oil * obs["days"]) / 1e6
    ax[2].plot(t_end, cum_o, "o", color=P.INK, ms=3, label="Observed")
    for a, ttl in zip(ax, ("Oil rate (STB/d)", "Water rate (STB/d)", "Cumulative oil (MMSTB)")):
        a.axvline(pd.Timestamp(C.FORECAST_START), color=P.INK_2, lw=0.8, ls=":")
        a.set_title(ttl); P.year_axis(a, 4)
    ax[0].legend(loc="upper right")
    P.save(fig, C.FIGURES / "04_forecast.png")

    # oil cut power law diagnostic (log-log)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    L = hist_oil + hist_wat
    x = np.cumsum(L * obs["days"]) - 0.5 * L * obs["days"]
    ok = (hist_oil > 0) & (L > 0)
    ax.loglog(x[ok][8:] / 1e6, hist_oil[ok][8:] / L[ok][8:], "o", color=P.INK, ms=4, label="Observed")
    for k, (o_, w_) in enumerate(zip(po, pw)):
        Lk = o_ + w_
        xk = np.cumsum(Lk * q.days) - 0.5 * Lk * q.days
        ax.loglog(xk[12:] / 1e6, o_[12:] / Lk[12:], color=P.PRIOR, lw=1, label="History-matched cases (proxy)" if k == 0 else None)
    df = fcs[0]
    xf = F._mid_cum(df["liquid_rate_stbd"].values[1:], F.forecast_days(), float(np.sum(L * obs["days"])))
    ax.loglog(xf / 1e6, df["oil_rate_stbd"].values[1:] / df["liquid_rate_stbd"].values[1:], color=P.BLUE,
              label=f"Forecast, β = {df.attrs['beta']:.2f}")
    ax.set_xlabel("Cumulative liquid (MMSTB)"); ax.set_ylabel("Oil cut")
    ax.set_title("Oil cut follows a power law of cumulative liquid"); ax.legend(loc="lower left")
    P.save(fig, C.FIGURES / "04_oilcut_powerlaw.png")

    b = bt[bt.cutoff == bt.cutoff.max()].sort_values("MAPE_cum_oil_%")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(b))
    ax.barh(y - 0.18, b["MAPE_cum_oil_%"], height=0.34, color=P.BLUE, label="Cumulative oil")
    ax.barh(y + 0.18, b["MAPE_final_oil_rate_%"], height=0.34, color=P.ORANGE, label="Final oil rate")
    ax.set_yticks(y, b["method"]); ax.invert_yaxis()
    ax.set_xlabel("Mean absolute error over 100 cases (%)")
    ax.set_title(f"Backtest: forecast {b.horizon_quarters.iloc[0]} quarters from {b.cutoff.iloc[0]}")
    ax.legend(loc="upper right")
    P.save(fig, C.FIGURES / "04_forecast_backtest.png")


if __name__ == "__main__":
    main()
