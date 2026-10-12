"""Paths, constants and modelling choices shared by every pipeline step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"
METRICS = OUTPUTS / "metrics"
DELIVERABLE = OUTPUTS / "deliverable"
POWERBI = OUTPUTS / "powerbi"

# Keyword used to locate each raw workbook in data/raw (robust to prefixes / spaces).
RAW_FILES = {
    "intro": "Introduction",
    "production": "Field_Production",
    "template": "Template",
}

# Uncertainty parameters, in the column order of "01 Introduction / Uncertainty Values".
PARAMS = ["Fault Transmissibility", "Porosity Multiplier", "Permeability Multiplier", "Aquifer Pore Volume"]
PARAM_SHORT = ["fault_trans", "poro_mult", "perm_mult", "aquifer_pv"]
# Prior bounds = the range spanned by the 100 provided cases.
PARAM_BOUNDS = {
    "fault_trans": (0.05, 0.15),
    "poro_mult": (0.80, 1.50),
    "perm_mult": (0.50, 10.0),
    "aquifer_pv": (50.0, 200.0),
}

# Official split from the challenge README.
TRAIN_CASES = list(range(1, 71))
VALID_CASES = list(range(71, 86))
TEST_CASES = list(range(86, 101))

OBSERVED_COLUMN = "RAMP_History"   # observed field history inside 02 Field Production
PHASES = ["Oil", "Water", "Gas"]

# History: 41 quarterly dates 1998-01-01 .. 2008-01-01 -> 40 quarterly average rates.
# Forecast: 80 quarterly dates 2008-04-01 .. 2028-01-01 (rows of the template).
FORECAST_START = "2008-04-01"
FORECAST_END = "2028-01-01"

# Template convention: cumulative volumes keep counting from first production (1998),
# exactly like a simulator's field totals (FOPT/FWPT/FGPT). Set to False to report
# volumes produced since 2008-01-01 instead.
CUMULATIVE_FROM_START = True

# ---- Proxy -----------------------------------------------------------------
PROXY_N_COMPONENTS = 12
PROXY_KERNEL = "matern"      # "matern" or "rbf"
PROXY_RESTARTS = 5

# ---- History matching --------------------------------------------------------
HM_N_SAMPLES = 2 ** 17       # Sobol samples evaluated through the proxy
HM_OIL_REL_SIGMA = 0.03      # measurement-error model: 3 % of observed rate ...
HM_WATER_REL_SIGMA = 0.05    # ... 5 % for water ...
HM_ABS_SIGMA_FLOOR = 150.0   # ... with a floor in STB/d
HM_N_SELECT = 10             # cases in the deliverable
HM_MIN_DISTANCE = 0.06       # min. distance between selected cases (unit-cube space)

# ---- Forecast ---------------------------------------------------------------
FC_FIT_QUARTERS = 9          # last 2 years of history + Q1-2008 for trend fitting
FC_LIQUID_MODE = "trend"     # "trend" (exponential fit) or "constant"

# ---- Illustrative economics (replace with your own assumptions) --------------
ECON = {
    "oil_price_usd_bbl": 70.0,
    "gas_price_usd_mscf": 3.0,
    "opex_fixed_usd_year": 20e6,
    "opex_usd_bbl_liquid": 2.5,     # lifting + water handling, per barrel of liquid
    "discount_rate": 0.10,
}

SEED = 2026
