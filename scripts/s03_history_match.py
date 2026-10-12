"""STEP 3 - History matching with the proxy.

A) Blind test of the workflow: each test case (86-100) plays the role of "observed
   field"; the proxy trained on cases 1-85 must recover its known parameters.
B) The real history match against the observed field (proxy trained on all 100).
   Output: posterior samples and the 10 selected, diverse, best-matching cases.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dwc import config as C, data, plots as P  # noqa: E402
from dwc.history_match import run_history_match, summarise_posterior  # noqa: E402
from dwc.proxy import PCAGPProxy  # noqa: E402

LABELS = {"fault_trans": "Fault transmissibility", "poro_mult": "Porosity multiplier",
          "perm_mult": "Permeability multiplier", "aquifer_pv": "Aquifer pore volume"}


def blind_test(q, par):
    proxy = PCAGPProxy.load(C.MODELS / "proxy_eval.joblib")
    rows = []
    for cid in C.TEST_CASES:
        o, w = q.rate("Oil").loc[cid].values.astype(float), q.rate("Water").loc[cid].values.astype(float)
        res = run_history_match(proxy, o, w, n_samples=2 ** 14, mcmc_steps=500, verbose=False)
        post = pd.DataFrame(res["posterior"], columns=C.PARAM_SHORT)
        true = par.loc[cid]
        row = {"case": cid}
        for p in C.PARAM_SHORT:
            lo, hi = C.PARAM_BOUNDS[p]
            row[f"{p}_true"] = true[p]
            row[f"{p}_map"] = res["map"][C.PARAM_SHORT.index(p)]
            row[f"{p}_abs_err_%range"] = 100 * abs(row[f"{p}_map"] - true[p]) / (hi - lo)
            q10, q90 = post[p].quantile([0.05, 0.95])
            row[f"{p}_in_90%_CI"] = bool(q10 <= true[p] <= q90)
            row[f"{p}_CI_width_%range"] = 100 * (q90 - q10) / (hi - lo)
        rows.append(row)
        print(f"  case {cid}: poro true {true['poro_mult']:.3f} map {row['poro_mult_map']:.3f} | "
              f"perm true {true['perm_mult']:.2f} map {row['perm_mult_map']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(C.METRICS / "hm_blind_test.csv", index=False)
    summ = pd.DataFrame({p: {"MAP error (% of prior range)": df[f"{p}_abs_err_%range"].mean(),
                             "90% interval width (% of prior range)": df[f"{p}_CI_width_%range"].mean(),
                             "truth inside 90% interval (%)": 100 * df[f"{p}_in_90%_CI"].mean()}
                         for p in C.PARAM_SHORT}).T
    summ.to_csv(C.METRICS / "hm_blind_test_summary.csv")
    print(summ.round(1).to_string())
    return df, summ


def main():
    P.style()
    q = data.build_quarterly()
    par = data.load_parameters()

    print("A) blind test on cases 86-100")
    bt, summ = blind_test(q, par)

    print("B) history match of the observed field")
    proxy = PCAGPProxy.load(C.MODELS / "proxy_final.joblib")
    o_obs = q.rate("Oil").loc["observed"].values.astype(float)
    w_obs = q.rate("Water").loc["observed"].values.astype(float)
    res = run_history_match(proxy, o_obs, w_obs)
    misfit = res["misfit"]

    post = pd.DataFrame(res["posterior"], columns=C.PARAM_SHORT)
    post["chi2"] = res["posterior_chi2"]
    post.sample(min(len(post), 20000), random_state=C.SEED).to_csv(C.METRICS / "hm_posterior_samples.csv", index=False)
    summarise_posterior(res["posterior"]).to_csv(C.METRICS / "hm_posterior_summary.csv")
    print(summarise_posterior(res["posterior"]).round(3).to_string())

    sel = pd.DataFrame(res["selected"], columns=C.PARAM_SHORT)
    so, sw = proxy.predict(res["selected"])
    nro, nrw = misfit.nrmse(so, sw)
    sel.insert(0, "deliverable_case", [f"Case {i + 1}" for i in range(len(sel))])
    sel["chi2_scaled"] = res["selected_chi2"]
    sel["nrmse_oil"], sel["nrmse_water"] = nro, nrw
    sel.to_csv(C.METRICS / "hm_selected_cases.csv", index=False)
    np.save(C.MODELS / "hm_selected_theta.npy", res["selected"])
    print(sel.round(4).to_string(index=False))

    prov = pd.read_csv(C.METRICS / "provided_cases_misfit.csv")
    print(f"  best provided case NRMSE oil+water {prov['nrmse_total'].min():.4f} | "
          f"best matched case {(nro + nrw).min():.4f}")

    # ---- Figures
    t = q.dates[:-1]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    for c in q.cases:
        ax[0].plot(t, q.rate("Oil").loc[c], color=P.PRIOR, lw=0.6)
        ax[1].plot(t, q.rate("Water").loc[c], color=P.PRIOR, lw=0.6)
    for k in range(len(so)):
        ax[0].plot(t, so[k], color=P.BLUE, lw=1.2, alpha=0.85)
        ax[1].plot(t, sw[k], color=P.BLUE, lw=1.2, alpha=0.85)
    for a, s in zip(ax, (o_obs, w_obs)):
        a.plot(t, s, "o", color=P.INK, ms=3.5)
        P.year_axis(a)
    ax[0].plot([], [], color=P.PRIOR, label="100 provided cases (prior)")
    ax[0].plot([], [], color=P.BLUE, label="10 history-matched cases")
    ax[0].plot([], [], "o", color=P.INK, ms=3.5, label="Observed")
    ax[0].legend(loc="lower left")
    ax[0].set_title("Oil rate (STB/d)"); ax[1].set_title("Water rate (STB/d)")
    P.save(fig, C.FIGURES / "03_history_match.png")

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    for a, p in zip(ax, C.PARAM_SHORT):
        lo, hi = C.PARAM_BOUNDS[p]
        bins = np.linspace(lo, hi, 36)
        a.hist(par[p], bins=bins, color=P.PRIOR, density=True, label="Prior (100 cases)")
        a.hist(post[p], bins=bins, color=P.BLUE, alpha=0.8, density=True, label="Posterior")
        for v in sel[p]:
            a.axvline(v, color=P.ORANGE, lw=1)
        a.set_title(LABELS[p]); a.set_yticks([])
    ax[0].plot([], [], color=P.ORANGE, label="10 selected cases"); ax[0].legend(loc="upper left", fontsize=8)
    P.save(fig, C.FIGURES / "03_posterior_parameters.png")

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    for a, p in zip(ax, C.PARAM_SHORT):
        lo, hi = C.PARAM_BOUNDS[p]
        a.plot([lo, hi], [lo, hi], color=P.INK_2, lw=1, ls="--")
        a.plot(bt[f"{p}_true"], bt[f"{p}_map"], "o", color=P.BLUE, ms=7, mec=P.SURFACE, mew=2)
        a.set_title(LABELS[p]); a.set_xlabel("True"); a.set_ylabel("Recovered (MAP)")
    fig.suptitle("Blind test: parameters recovered for unseen test cases 86-100", fontweight="bold", y=1.04)
    P.save(fig, C.FIGURES / "03_blind_test_recovery.png")


if __name__ == "__main__":
    main()
