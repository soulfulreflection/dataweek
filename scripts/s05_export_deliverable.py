"""STEP 5 - Fill '09 Template Deliverable' and export tidy tables for Power BI.

* outputs/deliverable/09_Template_Deliverable.xlsx : the template, untouched except for
  columns B-D of sheets 'Case 1' .. 'Case 10' (cumulative gas / oil / water).
* outputs/powerbi/dashboard_data.xlsx + CSVs : long-format tables for the dashboard.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import openpyxl  # noqa: E402
import pandas as pd  # noqa: E402

from dwc import config as C, data, forecast as F  # noqa: E402
from dwc.proxy import PCAGPProxy  # noqa: E402


def fill_template(fc_long: pd.DataFrame) -> Path:
    src = data.find_raw(C.RAW_FILES["template"])
    wb = openpyxl.load_workbook(src)
    cases = sorted(fc_long["deliverable_case"].unique(), key=lambda s: int(s.split()[1]))
    for case in cases:
        ws = wb[case]
        df = fc_long[fc_long.deliverable_case == case].reset_index(drop=True)
        header = [ws.cell(1, c).value for c in range(1, 5)]
        assert header[1].startswith("Gas") and header[2].startswith("Oil") and header[3].startswith("Water"), header
        for i, row in df.iterrows():
            r = i + 2
            cell_date = ws.cell(r, 1).value
            want = row["date"].strftime("%m/%d/%Y")
            got = cell_date.strftime("%m/%d/%Y") if hasattr(cell_date, "strftime") else str(cell_date).strip()
            if got != want:
                raise ValueError(f"{case} row {r}: template date {got} != forecast date {want}")
            ws.cell(r, 2).value = round(float(row["gas_cum_mscf"]), 2)
            ws.cell(r, 3).value = round(float(row["oil_cum_stb"]), 2)
            ws.cell(r, 4).value = round(float(row["water_cum_stb"]), 2)
        if ws.max_row != len(df) + 1:
            raise ValueError(f"{case}: template has {ws.max_row - 1} rows, forecast has {len(df)}")
    out = C.DELIVERABLE / "09_Template_Deliverable.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out


def verify(path: Path, fc_long: pd.DataFrame):
    wb = openpyxl.load_workbook(path, data_only=True)
    worst = 0.0
    for case, df in fc_long.groupby("deliverable_case"):
        ws = wb[case]
        vals = np.array([[ws.cell(r, c).value for c in (2, 3, 4)] for r in range(2, ws.max_row + 1)], float)
        ref = df[["gas_cum_mscf", "oil_cum_stb", "water_cum_stb"]].values
        worst = max(worst, float(np.max(np.abs(vals - ref) / ref)))
        assert np.all(np.diff(vals, axis=0) >= 0), f"{case}: cumulative not monotonic"
    print(f"  verified {len(wb.sheetnames)} sheets, max relative difference {worst:.1e}, cumulatives monotonic")


def powerbi_tables(q, fc_long, summary):
    obs_tab = data.load_observed_table()
    obs = F.observed_series(q, obs_tab)
    theta = np.load(C.MODELS / "hm_selected_theta.npy")
    proxy = PCAGPProxy.load(C.MODELS / "proxy_final.joblib")
    po, pw = proxy.predict(theta)
    gor = summary  # noqa: F841 (kept for readability of the table list below)
    quarter_end = list(q.dates[1:])

    hist = []
    for k in range(len(theta)):
        case = f"Case {k + 1}"
        g = F.field_forecast(obs, (0.0,))[0].attrs["gor"]
        hist.append(pd.DataFrame({"deliverable_case": case, "date": quarter_end, "period": "History (proxy)",
                                  "oil_rate_stbd": po[k], "water_rate_stbd": pw[k], "gas_rate_mscfd": g * po[k],
                                  "oil_cum_stb": np.cumsum(po[k] * q.days), "water_cum_stb": np.cumsum(pw[k] * q.days),
                                  "gas_cum_mscf": np.cumsum(g * po[k] * q.days)}))
    fc = fc_long.assign(period="Forecast")[["deliverable_case", "date", "period", "oil_rate_stbd", "water_rate_stbd",
                                            "gas_rate_mscfd", "oil_cum_stb", "water_cum_stb", "gas_cum_mscf"]]
    cases_ts = pd.concat(hist + [fc], ignore_index=True)
    cases_ts["water_cut"] = cases_ts["water_rate_stbd"] / (cases_ts["oil_rate_stbd"] + cases_ts["water_rate_stbd"])

    observed = pd.DataFrame({
        "date": list(q.dates[1:]) + [pd.Timestamp(C.FORECAST_START)],
        "oil_rate_stbd": obs["oil"], "water_rate_stbd": obs["water"], "gas_rate_mscfd": obs["gas"],
        "oil_cum_stb": np.cumsum(obs["oil"] * obs["days"]), "water_cum_stb": np.cumsum(obs["water"] * obs["days"]),
        "gas_cum_mscf": np.cumsum(obs["gas"] * obs["days"]),
        "water_injection_stbd": list(obs_tab["water_inj_rate"].values),
    })
    observed["water_cut"] = observed["water_rate_stbd"] / (observed["oil_rate_stbd"] + observed["water_rate_stbd"])

    prior = data.load_parameters().reset_index()
    prov = pd.read_csv(C.METRICS / "provided_cases_misfit.csv")[["case", "nrmse_oil", "nrmse_water", "nrmse_total"]]
    prior = prior.merge(prov, on="case").assign(split=lambda d: np.select(
        [d.case <= 70, d.case <= 85], ["Train", "Validation"], "Test"))

    tables = {
        "cases_timeseries": cases_ts,
        "observed_history": observed,
        "case_summary": summary.merge(pd.read_csv(C.METRICS / "hm_selected_cases.csv")[
            ["deliverable_case", "nrmse_oil", "nrmse_water"]], on="deliverable_case"),
        "provided_cases": prior,
        "proxy_benchmark": pd.read_csv(C.METRICS / "proxy_benchmark.csv"),
        "forecast_backtest": pd.read_csv(C.METRICS / "forecast_backtest.csv"),
        "forecast_sensitivity": pd.read_csv(C.METRICS / "forecast_sensitivity.csv"),
        "hm_blind_test": pd.read_csv(C.METRICS / "hm_blind_test.csv"),
    }
    C.POWERBI.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(C.POWERBI / "dashboard_data.xlsx") as xw:
        for name, df in tables.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
            df.to_csv(C.POWERBI / f"{name}.csv", index=False)
    print(f"  Power BI tables -> outputs/powerbi ({', '.join(tables)})")


def main():
    q = data.build_quarterly()
    fc_long = pd.read_csv(C.METRICS / "forecast_cases_long.csv", parse_dates=["date"])
    summary = pd.read_csv(C.METRICS / "forecast_case_summary.csv")
    out = fill_template(fc_long)
    print(f"  deliverable -> {out.relative_to(C.ROOT)}")
    verify(out, fc_long)
    powerbi_tables(q, fc_long, summary)


if __name__ == "__main__":
    main()
