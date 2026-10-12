# Slide deck outline (documentation deliverable)

Twelve slides, following the README's suggested topics. Figures are in `outputs/figures`; numbers are in
`outputs/metrics`.

| # | Slide | Content | Figure |
|---|---|---|---|
| 1 | **Title** | Team name, university, "ML-Assisted History Matching & 20-Year Forecast" | — |
| 2 | **Executive overview** | 55.9 MMSTB cumulative oil by 2028 (+17.1 MMSTB remaining), water cut 90%, 10 history-matched cases. Proxy R² 0.997. A model-error correction prevents a 29% overstatement of remaining oil | `04_forecast.png` |
| 3 | **Background & context** | What history matching is; why it is slow (many simulator runs); why a proxy helps | `01_history_all_cases.png` |
| 4 | **Objectives** | Proxy → optimal parameters → history-matched model → 20-year forecast (the 4 objectives from the brief) | — |
| 5 | **Data & key EDA insights** | 100 cases × 4 parameters; liquid-rate control; GOR constant → gas = 0.363 × oil; porosity multiplier dominates (ρ = 0.99) | `01_parameter_sensitivity.png` |
| 6 | **Physics insight** | Water cut collapses onto one curve vs cumulative liquid / porosity multiplier; the observed field departs late | `01_watercut_collapse.png` |
| 7 | **Methodology** | Pipeline diagram: PCA-GP proxy → Sobol + L-BFGS-B + MCMC → selection of 10 → backtested decline model → template | (draw a flow diagram) |
| 8 | **Proxy results** | Benchmark of 6 models; blind test on cases 86-100; ±2σ coverage 97% | `02_proxy_benchmark.png`, `02_proxy_test_cases.png` |
| 9 | **History match** | Posterior vs prior; what is identifiable (porosity) and what is not (fault); blind-test recovery | `03_history_match.png`, `03_posterior_parameters.png`, `03_blind_test_recovery.png` |
| 10 | **Forecast method** | Backtest of 5 methods; the power law is 1.2% off on cumulative oil; log-log diagnostic | `04_forecast_backtest.png`, `04_oilcut_powerlaw.png` |
| 11 | **Results & business insight** | 2028 volumes; sensitivity 54.8-56.4 MMSTB; uncorrected vs corrected forecast; illustrative NPV10 ≈ $389M; water handling grows from 7.9 to 10.5 kSTB/d, so water handling becomes the cost driver | `04_forecast.png` + table |
| 12 | **Conclusions & next steps** | Model error is the dominant uncertainty, so widen the prior (e.g. permeability < 0.5, fault = 0), add parameters, and run the simulator on the 10 selected cases. Also: LSTM proxy once longer simulations exist; economics with real costs | — |

**Talking points on innovation**
* The proxy is validated twice: as a regressor (test R²) and as part of the whole inverse workflow (blind test
  that recovers known parameters).
* The forecast method is chosen by evidence (a backtest on 100 cases), not by habit.
* We detected **model error** (β 1.32 observed vs 0.81 model) and corrected for it, instead of trusting the history
  match blindly.
* Uncertainty is reported honestly: the 10 cases agree closely because the history constrains them. The larger,
  structural uncertainty is shown separately (sensitivity range and uncorrected forecast).
