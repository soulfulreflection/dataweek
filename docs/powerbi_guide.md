# Power BI dashboard guide

Use Power BI Desktop **2.154.1260.0 (May 2026) or earlier**, as the challenge rules require.

## 1. Data sources

| Source | Why |
|---|---|
| `outputs/deliverable/09_Template_Deliverable.xlsx` | **Required** primary source: the 10 forecast cases |
| `outputs/powerbi/dashboard_data.xlsx` | Supporting tables: history match, parameters, model quality, economics |

### 1a. Combine the 10 template sheets into one table (Power Query)
*Home → Get data → Blank query → Advanced editor*, paste the query, change the path, and name the query `Deliverable`:

```powerquery
let
    Source   = Excel.Workbook(File.Contents("C:\DWC\outputs\deliverable\09_Template_Deliverable.xlsx"), null, true),
    Sheets   = Table.SelectRows(Source, each [Kind] = "Sheet" and Text.StartsWith([Name], "Case")),
    WithTbl  = Table.AddColumn(Sheets, "Tbl", each Table.PromoteHeaders([Data], [PromoteAllScalars = true])),
    Keep     = Table.SelectColumns(WithTbl, {"Name", "Tbl"}),
    Expanded = Table.ExpandTableColumn(Keep, "Tbl",
                 {"Date", "Gas production cumulative [MSCF]", "Oil production cumulative [STB]", "Water production cumulative [STB]"},
                 {"Date", "GasCum_MSCF", "OilCum_STB", "WaterCum_STB"}),
    Renamed  = Table.RenameColumns(Expanded, {{"Name", "Case"}}),
    Typed    = Table.TransformColumnTypes(Renamed,
                 {{"Date", type date}, {"GasCum_MSCF", type number}, {"OilCum_STB", type number}, {"WaterCum_STB", type number}},
                 "en-US"),                                   // template dates are MM/DD/YYYY
    CaseNo   = Table.AddColumn(Typed, "CaseNo", each Number.From(Text.AfterDelimiter([Case], " ")), Int64.Type),
    Sorted   = Table.Sort(CaseNo, {{"CaseNo", Order.Ascending}, {"Date", Order.Ascending}})
in
    Sorted
```

### 1b. Supporting tables
*Get data → Excel → `dashboard_data.xlsx`* and tick every sheet:

| Sheet | Content |
|---|---|
| `cases_timeseries` | 10 cases × (history from the proxy + forecast): rates, cumulatives, water cut; `period` = History (proxy) / Forecast |
| `observed_history` | Observed field 1998-2008: rates, cumulatives, water injection, water cut |
| `case_summary` | Parameters of the 10 cases, decline exponent, 2028 volumes, NPV, misfit |
| `provided_cases` | The 100 provided cases: parameters, split (Train/Validation/Test), misfit vs observed |
| `proxy_benchmark`, `forecast_backtest`, `forecast_sensitivity`, `hm_blind_test` | Model-quality evidence |

**Relationships:** `case_summary` is the case dimension (one row per case):
`case_summary[deliverable_case]` 1—* `Deliverable[Case]` and `case_summary[deliverable_case]` 1—*
`cases_timeseries[deliverable_case]`. Put the case slicer on `case_summary[deliverable_case]`. Add a date table if
you want year slicers.

## 2. Calculated columns on `Deliverable` (rates from cumulatives)

```dax
Prev Date =
CALCULATE ( MAX ( Deliverable[Date] ),
    FILTER ( Deliverable, Deliverable[Case] = EARLIER ( Deliverable[Case] )
                          && Deliverable[Date] < EARLIER ( Deliverable[Date] ) ) )

Oil Rate (STB/d) =
VAR prevCum = CALCULATE ( MAX ( Deliverable[OilCum_STB] ),
    FILTER ( Deliverable, Deliverable[Case] = EARLIER ( Deliverable[Case] )
                          && Deliverable[Date] = EARLIER ( Deliverable[Prev Date] ) ) )
RETURN IF ( ISBLANK ( Deliverable[Prev Date] ), BLANK (),
            DIVIDE ( Deliverable[OilCum_STB] - prevCum, Deliverable[Date] - Deliverable[Prev Date] ) )
```
Repeat for `Water Rate (STB/d)` (WaterCum_STB) and `Gas Rate (MSCF/d)` (GasCum_MSCF), then add:

```dax
Water Cut = DIVIDE ( Deliverable[Water Rate (STB/d)], Deliverable[Oil Rate (STB/d)] + Deliverable[Water Rate (STB/d)] )
```

## 3. Measures

```dax
Last Date        = MAX ( Deliverable[Date] )
Cum Oil 2028 (MMSTB) =
    DIVIDE ( CALCULATE ( AVERAGE ( Deliverable[OilCum_STB] ), Deliverable[Date] = DATE ( 2028, 1, 1 ) ), 1e6 )
Cum Oil 2008 (MMSTB) =
    DIVIDE ( CALCULATE ( AVERAGE ( Deliverable[OilCum_STB] ), Deliverable[Date] = DATE ( 2008, 4, 1 ) ), 1e6 )
Remaining Oil 2008-2028 (MMSTB) = [Cum Oil 2028 (MMSTB)] - [Cum Oil 2008 (MMSTB)]
Cum Water 2028 (MMSTB) =
    DIVIDE ( CALCULATE ( AVERAGE ( Deliverable[WaterCum_STB] ), Deliverable[Date] = DATE ( 2028, 1, 1 ) ), 1e6 )
Cum Gas 2028 (BSCF) =
    DIVIDE ( CALCULATE ( AVERAGE ( Deliverable[GasCum_MSCF] ), Deliverable[Date] = DATE ( 2028, 1, 1 ) ), 1e6 )
Water Cut 2028 = CALCULATE ( AVERAGE ( Deliverable[Water Cut] ), Deliverable[Date] = DATE ( 2028, 1, 1 ) )
Oil Cum Min = MINX ( VALUES ( Deliverable[Case] ), CALCULATE ( SUM ( Deliverable[OilCum_STB] ) ) )
Oil Cum Max = MAXX ( VALUES ( Deliverable[Case] ), CALCULATE ( SUM ( Deliverable[OilCum_STB] ) ) )
NPV10 (USD M)    = AVERAGE ( case_summary[npv10_musd] )
```

## 4. Suggested pages

| Page | Visuals |
|---|---|
| **1 · Executive summary** | KPI cards: Cum Oil 2028, Remaining Oil 2008-2028, Water Cut 2028, Cum Gas 2028, NPV10. Line chart of cumulative oil 1998-2028 (`observed_history` + `Deliverable`), with a case slicer |
| **2 · Forecast** | Line charts of oil, water and gas rate vs date (legend = Case). Area or band from `Oil Cum Min`/`Oil Cum Max`. Water cut vs date |
| **3 · History match** | Observed (markers) vs history-matched oil & water rate (`cases_timeseries` where period = History). Table of the 10 cases' parameters and misfit. Scatter of the 100 provided cases: porosity multiplier vs `nrmse_total`, coloured by split |
| **4 · Model quality** | Bar charts: proxy benchmark (test R² / NRMSE), forecast backtest (error by method). Scatter: blind-test true vs recovered porosity multiplier |
| **5 · Business view** | Sensitivity table (`forecast_sensitivity`). Remaining oil by scenario. NPV per case. A note that economics are illustrative |

Design tips: one blue for forecasts, dark markers for observed data, grey for the 100 provided cases (the same
palette as the figures in `outputs/figures`). Don't put two y-axes on one chart: put oil and water side by side
instead.
