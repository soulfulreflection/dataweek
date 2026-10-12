"""STEP 2 - Train the ML proxy (4 parameters -> 10-year oil & water rate curves).

* Benchmarks 6 model families on the official split
  (train = cases 1-70, validation = 71-85, test = 86-100).
* Selects the best on validation, reports test performance once.
* Saves two fitted proxies:
    proxy_eval.joblib  - trained on cases 1-85 (used for the blind history-match test on 86-100)
    proxy_final.joblib - trained on all 100 cases (used to history-match the real field)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dwc import config as C, data, plots as P  # noqa: E402
from dwc.proxy import PCAGPProxy, benchmark_models, score  # noqa: E402


def arrays(q, par, ids):
    return (par.loc[ids].values, q.rate("Oil").loc[ids].values.astype(float),
            q.rate("Water").loc[ids].values.astype(float))


def main():
    P.style()
    q = data.build_quarterly()
    par = data.load_parameters()
    tr, va, te = (arrays(q, par, ids) for ids in (C.TRAIN_CASES, C.VALID_CASES, C.TEST_CASES))

    rows = []
    for name, make in benchmark_models().items():
        t0 = time.time()
        m = make().fit(*tr)
        fit_s = time.time() - t0
        for split, (th, o, w) in (("validation", va), ("test", te)):
            po, pw = m.predict(th)
            rows.append({"model": name, "split": split, "fit_seconds": fit_s, **score(o, w, po, pw, q.days)})
        print(f"  {name:32s} val NRMSE oil {rows[-2]['NRMSE_oil_%']:.2f}% water {rows[-2]['NRMSE_water_%']:.2f}%"
              f" | test R2 {rows[-1]['R2']:.4f}")
    bench = pd.DataFrame(rows)
    bench.to_csv(C.METRICS / "proxy_benchmark.csv", index=False)
    val = bench[bench.split == "validation"].assign(err=lambda d: d["NRMSE_oil_%"] + d["NRMSE_water_%"])
    best = val.sort_values("err").iloc[0]["model"]
    print(f"  selected on validation: {best}")

    # ---- proxy trained on 1-85 (blind test of the HM workflow) and on all 100 (final)
    th_tv = np.vstack([tr[0], va[0]]); o_tv = np.vstack([tr[1], va[1]]); w_tv = np.vstack([tr[2], va[2]])
    proxy_eval = PCAGPProxy().fit(th_tv, o_tv, w_tv)
    proxy_eval.save(C.MODELS / "proxy_eval.joblib")
    th_all, o_all, w_all = arrays(q, par, q.cases)
    proxy = PCAGPProxy().fit(th_all, o_all, w_all)
    proxy.save(C.MODELS / "proxy_final.joblib")
    print(f"  PCA keeps {proxy.explained_variance() * 100:.3f}% of output variance")

    # ---- uncertainty calibration on the test cases (proxy trained on 1-85)
    po, pw, so, sw = proxy_eval.predict(te[0], return_std=True)
    z = np.abs(np.hstack([po - te[1], pw - te[2]])) / np.hstack([so, sw])
    cover = float(np.mean(z[:, np.r_[4:40, 44:80]] < 2))
    s_tv = score(te[1], te[2], po, pw, q.days)
    pd.DataFrame([{"proxy": "PCA-GP trained on 1-85", "split": "test 86-100", **s_tv,
                   "coverage_2sigma": cover}]).to_csv(C.METRICS / "proxy_final_test.csv", index=False)
    print(f"  proxy(1-85) on test: R2 {s_tv['R2']:.4f}, 2-sigma coverage {cover:.2f}")

    # ---- Figures
    t = q.dates[:-1]
    fig, ax = plt.subplots(2, 3, figsize=(15, 6.5), sharex=True)
    for j, k in enumerate([0, 7, 14]):
        cid = C.TEST_CASES[k]
        for r, (true, pred, sd, lab) in enumerate(((te[1], po, so, "Oil"), (te[2], pw, sw, "Water"))):
            a = ax[r, j]
            a.fill_between(t, pred[k] - 2 * sd[k], pred[k] + 2 * sd[k], color=P.BLUE_LIGHT, lw=0, label="Proxy ±2σ")
            a.plot(t, pred[k], color=P.BLUE, label="Proxy prediction")
            a.plot(t, true[k], "o", color=P.INK, ms=3, label="Simulator")
            a.set_title(f"Test case {cid} - {lab} rate (STB/d)")
            P.year_axis(a)
    ax[0, 0].legend(loc="lower left")
    fig.suptitle("Blind test: proxy trained on cases 1-85 vs simulator on unseen cases", fontweight="bold", y=1.02)
    P.save(fig, C.FIGURES / "02_proxy_test_cases.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    for a, (true, pred, lab) in zip(ax, ((te[1], po, "Cum. oil 2008"), (te[2], pw, "Cum. water 2008"))):
        x, y = true @ q.days / 1e6, pred @ q.days / 1e6
        lim = [min(x.min(), y.min()) * 0.97, max(x.max(), y.max()) * 1.03]
        a.plot(lim, lim, color=P.INK_2, lw=1, ls="--")
        a.plot(x, y, "o", color=P.BLUE, ms=7, mec=P.SURFACE, mew=2)
        a.set_xlabel("Simulator (MMSTB)"); a.set_ylabel("Proxy (MMSTB)"); a.set_title(f"{lab} - test cases 86-100")
    P.save(fig, C.FIGURES / "02_proxy_parity.png")

    b = bench[bench.split == "test"].sort_values("NRMSE_water_%")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(b))
    ax.barh(y - 0.18, b["NRMSE_oil_%"], height=0.34, color=P.BLUE, label="Oil")
    ax.barh(y + 0.18, b["NRMSE_water_%"], height=0.34, color=P.ORANGE, label="Water")
    ax.set_yticks(y, b["model"]); ax.invert_yaxis()
    ax.set_xlabel("NRMSE on test cases 86-100 (%)"); ax.set_title("Proxy model benchmark"); ax.legend()
    P.save(fig, C.FIGURES / "02_proxy_benchmark.png")


if __name__ == "__main__":
    main()
