"""STEP 1 - Data preparation & exploratory analysis.

* Puts the 100 simulated cases and the observed history on one quarterly grid.
* Ranks the provided cases by how well they match the observed history.
* Measures which uncertainty parameter drives the production response.
* Shows the key physical insight used later for forecasting: water cut collapses onto
  one curve when cumulative liquid is scaled by the porosity multiplier.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

from dwc import config as C, data, plots as P  # noqa: E402
from dwc.history_match import Misfit  # noqa: E402


def main():
    P.style()
    q = data.build_quarterly(cache=False)
    obs = data.load_observed_table()
    par = data.load_parameters()
    par.to_csv(C.PROCESSED / "parameters.csv")
    gap = data.check_observed_consistency(q, obs)
    print(f"  {len(q.cases)} cases, {len(q.dates)} quarterly dates; observed table vs RAMP_History gap {gap:.1e}")

    O, W, G = (q.rate(p) for p in ("Oil", "Water", "Gas"))
    cases = q.cases
    gor = (q.cum["Gas"].iloc[:, -1] / q.cum["Oil"].iloc[:, -1])
    print(f"  produced GOR: {gor.min():.4f} - {gor.max():.4f} MSCF/STB (constant -> gas = GOR x oil)")

    # --- rank the provided cases against the observed history (real simulator output)
    m = Misfit(O.loc["observed"].values, W.loc["observed"].values)
    rows = []
    for c in cases:
        no, nw = m.nrmse(O.loc[c].values, W.loc[c].values)
        rows.append({"case": c, "nrmse_oil": no, "nrmse_water": nw, "nrmse_total": no + nw,
                     "chi2": m.chi2(O.loc[c].values, W.loc[c].values)})
    rank = pd.DataFrame(rows).set_index("case").join(par).sort_values("nrmse_total")
    rank.to_csv(C.METRICS / "provided_cases_misfit.csv")
    print("  best provided cases:\n" + rank.head(5).round(4).to_string())

    # --- parameter sensitivity (Spearman rank correlation)
    outcomes = pd.DataFrame({
        "Cum. oil 2008": q.cum["Oil"].loc[cases].iloc[:, -1],
        "Cum. water 2008": q.cum["Water"].loc[cases].iloc[:, -1],
        "Water breakthrough quarter": (W.loc[cases] > 50).values.argmax(1),
        "Oil rate 2007-Q4": O.loc[cases].iloc[:, -1],
    }, index=cases)
    sens = par.join(outcomes).corr("spearman").loc[C.PARAM_SHORT, outcomes.columns]
    sens.to_csv(C.METRICS / "parameter_sensitivity_spearman.csv")
    print("  Spearman sensitivity:\n" + sens.round(2).to_string())

    # --- Figure: all cases vs observed
    t = q.dates[:-1]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for c in cases:
        ax[0].plot(t, O.loc[c], color=P.PRIOR, lw=0.7)
        ax[1].plot(t, W.loc[c], color=P.PRIOR, lw=0.7)
        ax[2].plot(t, O.loc[c] + W.loc[c], color=P.PRIOR, lw=0.7)
    for a, s in zip(ax, (O, W, O + W)):
        a.plot(t, s.loc["observed"], "o", color=P.INK, ms=3.5, label="Observed field")
    ax[0].plot([], [], color=P.PRIOR, label="100 simulated cases")
    ax[0].set_title("Oil rate (STB/d)"); ax[1].set_title("Water rate (STB/d)")
    ax[2].set_title("Liquid rate (STB/d): same in all cases")
    ax[0].legend(loc="lower left")
    for a in ax:
        P.year_axis(a)
    P.save(fig, C.FIGURES / "01_history_all_cases.png")

    # --- Figure: water-cut collapse
    pm = par["poro_mult"]
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#2a78d6", "#0d366b"])
    norm = Normalize(pm.min(), pm.max())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    days = q.days
    for c in cases:
        L = (O.loc[c] + W.loc[c]).values
        x = np.cumsum(L * days) - 0.5 * L * days
        fw = W.loc[c].values / np.maximum(L, 1e-9)
        ax[0].plot(t, fw, color=cmap(norm(pm[c])), lw=0.8)
        ax[1].plot(x / pm[c] / 1e6, fw, color=cmap(norm(pm[c])), lw=0.8)
    Lo = (O.loc["observed"] + W.loc["observed"]).values
    xo = np.cumsum(Lo * days) - 0.5 * Lo * days
    fwo = W.loc["observed"].values / np.maximum(Lo, 1e-9)
    pm_hm = float(rank.head(10)["poro_mult"].mean())
    ax[0].plot(t, fwo, "o", color=P.INK, ms=3.5, label="Observed")
    ax[1].plot(xo / pm_hm / 1e6, fwo, "o", color=P.INK, ms=3.5, label=f"Observed (poro mult {pm_hm:.2f})")
    ax[0].set_title("Water cut vs time"); ax[0].legend(loc="upper left"); P.year_axis(ax[0])
    ax[1].set_title("Water cut vs cumulative liquid / porosity multiplier")
    ax[1].set_xlabel("Cumulative liquid / poro mult (MMSTB)"); ax[1].legend(loc="upper left")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="Porosity multiplier", fraction=0.025)
    P.save(fig, C.FIGURES / "01_watercut_collapse.png")

    # --- Figure: sensitivity bars
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = ["Fault transmissibility", "Porosity multiplier", "Permeability multiplier", "Aquifer pore volume"]
    y = np.arange(4)
    ax.barh(y + 0.18, sens["Cum. oil 2008"].values, height=0.34, color=P.BLUE, label="Cum. oil 2008")
    ax.barh(y - 0.18, sens["Cum. water 2008"].values, height=0.34, color=P.ORANGE, label="Cum. water 2008")
    ax.set_yticks(y, labels); ax.axvline(0, color=P.INK_2, lw=0.8)
    ax.set_xlim(-1.05, 1.05); ax.set_xlabel("Spearman rank correlation")
    ax.set_title("What drives the 10-year history?"); ax.legend(loc="lower right")
    P.save(fig, C.FIGURES / "01_parameter_sensitivity.png")


if __name__ == "__main__":
    main()
