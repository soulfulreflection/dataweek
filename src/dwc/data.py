"""Load the challenge workbooks and put every case on the same quarterly time grid.

Each simulated case in ``02 Field Production`` is reported on its own irregular
time steps, but all of them contain the 41 quarterly dates 1998-01-01 .. 2008-01-01.
We therefore keep the *cumulative* volumes at those dates (exact, no interpolation)
and derive quarterly average rates as  d(cumulative) / d(days).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

SHEETS = {
    "Oil": "Oil production cumulative",
    "Water": "Water production cumulative",
    "Gas": "Gas production cumulative",
}


def find_raw(keyword: str):
    hits = sorted(p for p in C.RAW.glob("*.xlsx") if keyword.lower() in p.name.lower().replace(" ", "_")
                  and not p.name.startswith("~$"))
    if not hits:
        raise FileNotFoundError(f"No workbook containing '{keyword}' in {C.RAW}. See data/raw/README.md.")
    return hits[0]


def load_parameters() -> pd.DataFrame:
    """Uncertainty parameters of cases 1-100 (index = case number)."""
    df = pd.read_excel(find_raw(C.RAW_FILES["intro"]), sheet_name="Uncertainty Values")
    df["case"] = df["Item"].str.extract(r"(\d+)").astype(int)
    df = df.set_index("case")[C.PARAMS]
    df.columns = C.PARAM_SHORT
    return df.astype(float)


def load_observed_table() -> pd.DataFrame:
    """The observed 'Field Production' sheet.

    Its rates are *forward labelled*: the value dated d is the average rate over the
    quarter that starts at d, so the last row (2008-01-01) already gives Q1-2008.
    """
    df = pd.read_excel(find_raw(C.RAW_FILES["production"]), sheet_name="Field Production")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.rename(columns={
        "Oil production rate (STB/d)": "oil_rate",
        "Gas production rate (MSCF/d)": "gas_rate",
        "Water production rate (STB/d)": "water_rate",
        "Water injection rate (STB/d)": "water_inj_rate",
    })[["Date", "oil_rate", "gas_rate", "water_rate", "water_inj_rate"]]


def _read_cumulative_sheet(path, sheet, dates) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    names = list(raw.iloc[1, 1:])                       # Case_1 .. Case_100, RAMP_History
    body = raw.iloc[3:].copy()
    body.columns = ["Date"] + names
    body = body.dropna(subset=["Date"])
    body["Date"] = pd.to_datetime(body["Date"])
    body = body.set_index("Date").apply(pd.to_numeric, errors="coerce")
    body = body[~body.index.duplicated()]
    out = body.reindex(dates)
    if out.isna().any().any():
        missing = out.columns[out.isna().any()].tolist()
        raise ValueError(f"{sheet}: quarterly dates missing for {missing[:5]} ...")
    return out.T                                          # rows = cases, cols = dates


@dataclass
class Quarterly:
    """Quarterly history of every simulated case and of the observed field."""
    dates: pd.DatetimeIndex          # 41 report dates (1998-01-01 .. 2008-01-01)
    days: np.ndarray                 # 40 quarter lengths
    cum: dict                        # phase -> DataFrame (index: case 1..100 + 'observed', cols: dates)

    @property
    def cases(self):
        return [c for c in self.cum["Oil"].index if c != "observed"]

    def rate(self, phase: str) -> pd.DataFrame:
        """Quarterly average rates (40 columns, labelled by the quarter's start date)."""
        c = self.cum[phase]
        r = c.diff(axis=1).iloc[:, 1:].div(self.days, axis=1)
        r.columns = self.dates[:-1]
        return r

    def to_long(self) -> pd.DataFrame:
        frames = []
        for ph in C.PHASES:
            s = self.cum[ph].stack()
            s.name = f"{ph.lower()}_cum"
            frames.append(s)
        df = pd.concat(frames, axis=1).reset_index()
        df.columns = ["case", "date"] + [f"{p.lower()}_cum" for p in C.PHASES]
        return df


def build_quarterly(cache: bool = True) -> Quarterly:
    cache_file = C.PROCESSED / "quarterly_cumulative.csv"
    if cache and cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["date"])
        df["case"] = df["case"].map(lambda v: v if v == "observed" else int(v))
        dates = pd.DatetimeIndex(sorted(df["date"].unique()))
        cum = {ph: df.pivot(index="case", columns="date", values=f"{ph.lower()}_cum")
               .reindex(columns=dates) for ph in C.PHASES}
        order = sorted(c for c in cum["Oil"].index if c != "observed") + ["observed"]
        cum = {ph: v.reindex(order) for ph, v in cum.items()}
    else:
        obs = load_observed_table()
        dates = pd.DatetimeIndex(obs["Date"])
        path = find_raw(C.RAW_FILES["production"])
        cum = {}
        for ph, sheet in SHEETS.items():
            t = _read_cumulative_sheet(path, sheet, dates)
            t.index = [("observed" if n == C.OBSERVED_COLUMN else int(str(n).split("_")[1])) for n in t.index]
            cum[ph] = t.astype(float)
    days = np.diff(dates.values).astype("timedelta64[D]").astype(float)
    q = Quarterly(dates=dates, days=days, cum=cum)
    if cache and not cache_file.exists():
        C.PROCESSED.mkdir(parents=True, exist_ok=True)
        q.to_long().to_csv(cache_file, index=False)
    return q


def check_observed_consistency(q: Quarterly, obs: pd.DataFrame) -> float:
    """Max relative gap between the observed table rates and RAMP_History cumulative."""
    r = q.rate("Oil").loc["observed"].values
    t = obs["oil_rate"].values[:-1]          # forward labelled -> same quarters
    return float(np.max(np.abs(r - t)) / np.max(t))
