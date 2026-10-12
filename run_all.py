"""Run the full pipeline: python run_all.py  (about 10 minutes on a laptop)."""
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

STEPS = [
    ("s01_prepare_data", "Step 1 - data preparation & EDA"),
    ("s02_train_proxy", "Step 2 - ML proxy training & benchmark"),
    ("s03_history_match", "Step 3 - history matching"),
    ("s04_forecast", "Step 4 - 20-year forecast"),
    ("s05_export_deliverable", "Step 5 - deliverable & Power BI export"),
]

if __name__ == "__main__":
    only = sys.argv[1:]
    for mod, title in STEPS:
        if only and not any(o in mod for o in only):
            continue
        print(f"\n=== {title} ===")
        t0 = time.time()
        importlib.import_module(mod).main()
        print(f"--- done in {time.time() - t0:.0f}s")
