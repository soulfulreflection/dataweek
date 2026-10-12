# ExxonMobil DataWorks Challenge 2026 — ML-Assisted History Matching & Production Forecasting

A reproducible pipeline for the challenge workflow:

```
data prep & EDA → ML proxy → proxy-based history matching → 20-year forecast → template + Power BI tables
```

## Headline results

| Stage | Result |
|---|---|
| **Proxy** (4 parameters → 10-year oil & water curves) | PCA + Gaussian process. **R² = 0.997 on the held-out test cases 86–100**, 0.12% error on 2008 cumulative oil; 97% of test points fall inside its ±2σ band |
| **History match** | Porosity multiplier **1.349 ± 0.010**; permeability multiplier 0.54 (P10–P90 0.51–0.64); aquifer PV 162 (107–192); fault transmissibility not identifiable. Best match beats every provided case (NRMSE 0.049 vs 0.052) |
| **Workflow blind test** | On unseen test cases the workflow recovers the porosity multiplier to 1% of its range; the true value is inside the 90% interval for 100% of cases |
| **Forecast backtest** | The chosen oil-cut power law is **1.2% off on cumulative oil** over 2 years across 100 simulated cases (WOR trend 2.5%, Arps 11.2%) |
| **20-year forecast (2028)** | Cumulative oil **55.9 MMSTB** (+17.1 MMSTB after Q1-2008), water cut 90%, oil rate ≈1,220 STB/d, cumulative gas 20.3 BSCF |
| **Key insight** | The observed late-time water-cut rise is steeper than any realisation of the 4-parameter model can produce (decline exponent β = 1.32 vs 0.81). Trusting the model blindly would overstate the 2028 cumulative oil by 5 MMSTB (+29% of remaining oil). Our forecast corrects for this model error |

The filled deliverable is `outputs/deliverable/09_Template_Deliverable.xlsx`.

---

## How to run

```bash
pip install -r requirements.txt
# copy 01_Introduction.xlsx, 02_Field_Production.xlsx and 09_Template_Deliverable.xlsx into data/raw/
python run_all.py                 # all 5 steps, ~6 minutes
python run_all.py s04 s05         # re-run only some steps
python tests/test_sanity.py       # sanity checks
```

The grid workbooks (03–08) are large and are **not needed** by the model. Each case is fully defined by its four
uncertainty parameters in `01_Introduction.xlsx`. To summarise them for EDA or slides, run this on a machine that has them:

```bash
python tools/extract_grid_features.py --map-case 1     # -> data/processed/grid_features.csv (+ maps of case 1)
```

## Step by step

### Step 1 — Data preparation & EDA (`scripts/s01_prepare_data.py`)
* Every simulated case reports on its own irregular time steps, but all contain the 41 quarterly dates
  1998-01-01 … 2008-01-01. We keep cumulative volumes at those dates (exact values, no interpolation) and derive
  quarterly average rates.
* The observed table is **forward-labelled**: the rate dated *d* applies to the quarter that starts at *d*. So the
  Q1-2008 rate is already known, and the template's first row (2008-04-01) is anchored on observed data.
* Findings:
  * The produced GOR is constant at 0.3632 MSCF/STB in every case (undersaturated oil), so **gas = GOR × oil**.
  * Liquid rate is the same in all cases: wells run on historical liquid-rate control, and the cases differ in
    *how fast water replaces oil*.
  * The **porosity multiplier dominates** the history (Spearman ρ = 0.99 with cumulative oil). The other three
    parameters are weak, so the problem is non-unique.
  * **Water cut collapses onto one curve** when cumulative liquid is divided by the porosity multiplier
    (`01_watercut_collapse.png`). The observed field departs from that curve after ~30 MMSTB, which is the first
    sign of model error.

### Step 2 — ML proxy (`scripts/s02_train_proxy.py`, `src/dwc/proxy.py`)
* Inputs are the 4 parameters, scaled to [0, 1] (permeability on a log scale). The output is 80 values
  (40 quarterly oil rates + 40 water rates).
* PCA compresses the curves to 12 components (99.97% of variance). One GP per component (Matérn 5/2, ARD) maps
  the parameters to the component scores. The GP's predictive σ is propagated back to rates.
* Six model families are benchmarked on the official split (train 1–70 / validation 71–85 / test 86–100).
  The model is chosen on validation and reported once on test (`outputs/metrics/proxy_benchmark.csv`).

| Model (trained on 1–70) | Test R² | Oil NRMSE | Water NRMSE |
|---|---|---|---|
| **PCA-GP, Matérn, 12 PCs (selected)** | **0.9967** | **0.27%** | **0.47%** |
| PCA-GP, RBF, 8 PCs | 0.9962 | 0.33% | 0.51% |
| MLP ensemble (5 × 64-64) | 0.9958 | 0.37% | 0.58% |
| RBF interpolation | 0.9945 | 0.35% | 0.59% |
| Polynomial ridge, degree 2 / 3 | 0.993 / 0.991 | 0.43% / 0.47% | 0.83% / 0.83% |

### Step 3 — History matching (`scripts/s03_history_match.py`, `src/dwc/history_match.py`)
1. **Global screening:** 131,072 Sobol samples of the prior box, evaluated through the proxy in seconds.
2. **Local refinement:** L-BFGS-B from the best 20 samples gives the MAP estimate.
3. **Likelihood:** measurement error of 3% (oil) / 5% (water), with a 150 STB/d floor, plus the proxy's own σ.
4. **Posterior:** vectorised Metropolis MCMC (32 chains, 32k samples).
5. **Selection:** the 10 lowest-misfit posterior samples that are at least 0.06 apart in normalised parameter
   space. These become *Case 1 … Case 10* of the deliverable.
6. **Blind test of the workflow:** each test case 86–100 plays "observed field" for a proxy trained on 1–85
   (`03_blind_test_recovery.png`).

| Parameter | Blind test: MAP error (% of range) | Truth in 90% interval | Field posterior |
|---|---|---|---|
| Porosity multiplier | 1.0% | 100% | 1.349 ± 0.010 |
| Permeability multiplier | 9.9% | 93% | 0.54 (0.51–0.64), at the lower bound |
| Aquifer pore volume | 12.4% | 100% | 162 (107–192) |
| Fault transmissibility | 42.7% (not identifiable) | 80% | unconstrained |

### Step 4 — 20-year forecast (`scripts/s04_forecast.py`, `src/dwc/forecast.py`)
The simulated cases stop in 2008, so the proxy cannot see the future. Forecasting therefore uses a physics-based
decline model, **chosen by backtest**: each of the 100 cases is truncated, forecast and compared with the
simulator.

| Method (2006 cut-off, 8 quarters ahead, 100 cases) | Cum-oil error | Final-rate error | Bias |
|---|---|---|---|
| **Oil-cut power law vs cumulative liquid, anchored (chosen)** | **1.2%** | **2.2%** | −1.0% |
| Oil-cut power law, regression intercept | 1.5% | 2.3% | +0.2% |
| WOR vs cumulative oil (semilog) | 2.5% | 5.2% | −2.4% |
| Logit water cut vs log cumulative liquid | 3.1% | 6.6% | −3.1% |
| Arps exponential decline | 11.2% | 21.1% | −11.2% |

Earlier cut-offs (2004, 2005) are harder for every method because the liquid rate falls 15% inside their horizon.
The power law assumes stable operation, which is the field's state since 2006 (liquid within ±1%).

**Model:**
* Liquid rate continues its 2-year trend (−0.13% per quarter; constant liquid is a sensitivity case).
* Oil cut `f_o = a · x^(−β)`, where *x* is cumulative liquid. This is the late-life Buckley–Leverett behaviour
  of a waterflood. β is fitted on the last 9 observed quarters and anchored on the Q1-2008 oil cut.
* Water = liquid − oil; gas = 0.3633 × oil (observed GOR).

**Model-error correction:** the observed β is 1.32, but every history-matched realisation has β ≈ 0.81.
Each delivered case therefore uses `β_k = β_obs + (β_k,proxy − mean β_proxy)`: the observed trend plus that
realisation's own parameter-driven deviation. The uncorrected alternative is kept for comparison.

| 2028 result | 10 delivered cases | Sensitivity (fit window 6/9/12 q × liquid trend/constant) | Uncorrected |
|---|---|---|---|
| Cumulative oil | 55.89–55.99 MMSTB | 54.8–56.4 MMSTB | 60.9 MMSTB |
| Oil rate | 1,209–1,223 STB/d | 1,051–1,293 STB/d | ≈2,000 STB/d |

### Step 5 — Deliverable & dashboard data (`scripts/s05_export_deliverable.py`)
* Writes only columns B–D of sheets *Case 1 … Case 10* of the original template, after checking every row's
  date. The values are then re-read and verified, including that each cumulative is monotonic.
* Writes `outputs/powerbi/dashboard_data.xlsx` plus CSVs: long-format time series (history + forecast) for the
  10 cases, the observed history, case parameters and economics, and the benchmark, backtest, sensitivity and
  blind-test tables. See `docs/powerbi_guide.md`.

## Assumptions to confirm with the organisers
1. **Cumulative volumes continue from first production (1998)**, like simulator field totals. If the organisers want
   volumes counted from 2008, set `CUMULATIVE_FROM_START = False` in `src/dwc/config.py` and re-run steps 4–5.
2. **Operating strategy after 2008 is unchanged** (liquid rate continues its recent trend, no new wells).
3. The economics in `config.ECON` ($70/bbl, $20M/yr fixed opex, $2.5/bbl liquid, 10% discount) are illustrative.

## Repository layout
```
src/dwc/            config, data loading, proxy, history matching, forecast, plotting
scripts/            s01 … s05 pipeline steps (run_all.py runs them in order)
tools/              extract_grid_features.py for the large grid workbooks
tests/              sanity checks
outputs/figures     figures for the slides
outputs/metrics     every table quoted above (CSV)
outputs/models      fitted proxies (joblib) and the selected parameter sets
outputs/deliverable 09_Template_Deliverable.xlsx (filled)
outputs/powerbi     dashboard source tables
docs/               methodology, Power BI guide, slide outline, video script
data/raw            challenge files (not committed: competition data)
```

## References
* Hu Huang et al., *A Deep-Learning-Based Graph Neural Network-Long-Short-Term Memory Model for Reservoir
  Simulation and Optimization with Varying Well Controls*, SPE 2023.
* Zhen Zhang et al., *Robust Method for Reservoir Simulation History Matching Using Bayesian Inversion and
  Long-Short-Term Memory Network-Based Proxy*, SPE 2023.
* Buckley & Leverett (1942), *Mechanism of fluid displacement in sands*, Trans. AIME 146.
