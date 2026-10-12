"""Summarise the large grid workbooks (03-08) into one small table, one row per case.

The grid files are too big to share easily, and the production model does not need
cell-level data (each case is fully defined by its 4 uncertainty parameters). This
tool streams the workbooks row by row (low memory) and keeps per-case aggregates
that are useful for EDA and slides, e.g. pore volume, hydrocarbon pore volume,
average permeability, aquifer cells.

Usage (from the repository root, with the files in data/raw):
    python tools/extract_grid_features.py                         # all files 03-08
    python tools/extract_grid_features.py --files data/raw/03*.xlsx
    python tools/extract_grid_features.py --map-case 1            # also draw maps of case 1

Output: data/processed/grid_features.csv   (+ outputs/figures/00_grid_maps_case<N>.png)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NULL = -999.0           # the files use -999.25 for inactive cells

CODE_RE = re.compile(r"\(([A-Z]+)\)")


def column_codes(header):
    """Map header text -> short code, e.g. 'Pore Volume (PORV)' -> 'PORV', 'I Index' -> 'I'."""
    codes = []
    for h in header:
        h = str(h or "").strip()
        m = CODE_RE.search(h)
        if m:
            codes.append(m.group(1))
        elif h.endswith("Index"):
            codes.append(h.split()[0].upper())
        elif h.endswith("Coordinate"):
            codes.append(h.split()[0].upper() + "COORD")
        else:
            codes.append(h)
    return codes


class CaseAccumulator:
    WEIGHTED = ["PORO", "PERMX", "PERMY", "PERMZ", "NTG", "PRESSURE", "SWAT", "SOIL", "SGAS", "RS", "SWL", "SOWCR"]

    def __init__(self):
        self.n_rows = 0
        self.n_active = 0
        self.porv = 0.0
        self.porv_max = 0.0
        self.wsum = defaultdict(float)
        self.tran = defaultdict(float)
        self.tran_n = defaultdict(int)
        self.water_zone_porv = 0.0
        self.n_water_zone = 0
        self.dims = [0, 0, 0]
        self.fipnum = set()

    def add(self, r: dict):
        self.n_rows += 1
        for k, key in enumerate(("I", "J", "K")):
            v = r.get(key)
            if v is not None:
                self.dims[k] = max(self.dims[k], int(v))
        pv = r.get("PORV")
        if pv is None or pv <= 0 or pv < NULL + 1:
            return
        self.n_active += 1
        self.porv += pv
        self.porv_max = max(self.porv_max, pv)
        for key in self.WEIGHTED:
            v = r.get(key)
            if v is not None and v > NULL + 1:
                self.wsum[key] += pv * v
        for key in ("TRANX", "TRANY", "TRANZ"):
            v = r.get(key)
            if v is not None and v > NULL + 1:
                self.tran[key] += v
                self.tran_n[key] += 1
        sw = r.get("SWAT")
        if sw is not None and sw >= 0.99:
            self.water_zone_porv += pv
            self.n_water_zone += 1
        if r.get("FIPNUM") is not None:
            self.fipnum.add(r["FIPNUM"])

    def result(self) -> dict:
        out = {"n_rows": self.n_rows, "n_active": self.n_active, "ni": self.dims[0], "nj": self.dims[1],
               "nk": self.dims[2], "porv_total": self.porv, "porv_max_cell": self.porv_max,
               "hcpv_oil": self.wsum["SOIL"], "hcpv_gas": self.wsum["SGAS"],
               "water_zone_porv": self.water_zone_porv, "n_water_zone_cells": self.n_water_zone,
               "n_fipnum": len(self.fipnum)}
        for key in self.WEIGHTED:
            out[f"pv_avg_{key.lower()}"] = self.wsum[key] / self.porv if self.porv else np.nan
        for key in ("TRANX", "TRANY", "TRANZ"):
            out[f"mean_{key.lower()}"] = self.tran[key] / self.tran_n[key] if self.tran_n[key] else np.nan
        return out


def process_workbook(path: Path, map_case: int | None, maps: dict):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = []
    for ws in wb.worksheets:
        m = re.search(r"(\d+)", ws.title)
        if not m:
            continue
        case = int(m.group(1))
        t0 = time.time()
        it = ws.iter_rows(values_only=True)
        codes = column_codes(next(it))
        acc = CaseAccumulator()
        want_map = map_case == case
        cells = [] if want_map else None
        for vals in it:
            if vals is None or vals[0] is None:
                continue
            r = {c: (v if isinstance(v, (int, float)) else None) for c, v in zip(codes, vals)}
            acc.add(r)
            if want_map:
                cells.append(r)
        rows.append({"case": case, "source_file": path.name, **acc.result()})
        if want_map:
            maps[case] = pd.DataFrame(cells)
        print(f"  {path.name} / {ws.title}: {acc.n_rows} cells ({acc.n_active} active) in {time.time() - t0:.0f}s",
              flush=True)
    wb.close()
    return rows


def draw_maps(case: int, cells: pd.DataFrame):
    sys.path.insert(0, str(ROOT / "src"))
    import matplotlib.pyplot as plt
    from dwc import plots as P
    P.style()
    df = cells[(cells["PORV"] > 0)].copy()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, (col, how, ttl) in zip(ax, (("PERMX", "mean", "Mean permeability X (mD)"),
                                       ("SOIL", "mean", "Initial oil saturation (layer average)"),
                                       ("PORV", "sum", "Pore volume (column sum)"))):
        if col not in df:
            a.set_visible(False)
            continue
        g = df.groupby(["J", "I"])[col].agg(how).unstack()
        im = a.imshow(g.values, origin="lower", aspect="auto", cmap="Blues")
        fig.colorbar(im, ax=a, fraction=0.04)
        a.set_title(ttl); a.set_xlabel("I"); a.set_ylabel("J"); a.grid(False)
    fig.suptitle(f"Reservoir model - case {case} (top view)", fontweight="bold")
    out = ROOT / "outputs" / "figures" / f"00_grid_maps_case{case}.png"
    P.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", help="workbooks to process (default: data/raw/0[3-8]*.xlsx)")
    ap.add_argument("--map-case", type=int, default=None, help="draw top-view maps for this case")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "grid_features.csv"))
    a = ap.parse_args()
    files = [Path(f) for f in a.files] if a.files else sorted(
        p for p in (ROOT / "data" / "raw").glob("*.xlsx") if re.match(r"0[3-8]", p.name))
    if not files:
        sys.exit("No grid workbooks found (expected data/raw/03_*.xlsx ... 08_*.xlsx).")
    maps, rows = {}, []
    for f in files:
        rows += process_workbook(f, a.map_case, maps)
    out = pd.DataFrame(rows).sort_values("case")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"saved {len(out)} cases -> {a.out}")
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from dwc.data import load_parameters
        par = load_parameters()
        j = out.set_index("case").join(par, how="inner")
        cols = ["porv_total", "hcpv_oil", "water_zone_porv", "pv_avg_permx", "mean_tranx"]
        print("Spearman correlation with the uncertainty parameters:")
        print(j[cols + list(par.columns)].corr("spearman").loc[cols, par.columns].round(2).to_string())
    except Exception as e:  # parameters file not present
        print(f"(skipped correlation check: {e})")
    for case, cells in maps.items():
        draw_maps(case, cells)


if __name__ == "__main__":
    main()
