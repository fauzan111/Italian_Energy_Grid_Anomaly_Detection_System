import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Automatically generate mock raw data CSVs in CI if they are missing
raw_data_dir = ROOT / "data" / "raw" / "terna"
raw_data_dir.mkdir(parents=True, exist_ok=True)

total_load_file = raw_data_dir / "TotalLoad_2025.csv"
actual_generation_file = raw_data_dir / "ActualGeneration_2025.csv"

if not total_load_file.exists():
    import pandas as pd
    dates = pd.date_range(start="2025-01-01 00:00:00", periods=100, freq="15min")
    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%dT%H:%M:%S"),
        "Total Load [MW]": [20000.0 + i * 10 for i in range(100)],
        "Forecast Total Load [MW]": [20100.0 + i * 10 for i in range(100)],
        "Bidding Zone": "Italy"
    })
    df.to_csv(total_load_file, index=False)

if not actual_generation_file.exists():
    import pandas as pd
    dates = pd.date_range(start="2025-01-01 00:00:00", periods=100, freq="15min")
    rows = []
    for d in dates:
        date_str = d.strftime("%Y-%m-%dT%H:%M:%S")
        rows.append({"Date": date_str, "Actual Generation": 15000.0, "Primary Source": "Thermal"})
        rows.append({"Date": date_str, "Actual Generation": 5000.0, "Primary Source": "Wind"})
    df = pd.DataFrame(rows)
    df.to_csv(actual_generation_file, index=False)
