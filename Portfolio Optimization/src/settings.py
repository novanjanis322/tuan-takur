from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOCAL_DATA_DIR = DATA_DIR / "local"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# "bigquery" (default) or "local" (CSV files written by scripts/ingest_idx_data.py)
DATA_SOURCE = os.getenv("DATA_SOURCE", "bigquery").lower()
# Local mode only: optional CSV with a "ticker" column limiting the stock universe
UNIVERSE_FILE = os.getenv("UNIVERSE_FILE") or None

# Gurobi WLS credentials. When unset, the default license on this machine is used.
WLS_ACCESS_ID = os.getenv("WLS_ACCESS_ID") or None
WLS_SECRET = os.getenv("WLS_SECRET") or None
LICENSE_ID = int(os.getenv("LICENSE_ID")) if os.getenv("LICENSE_ID") else None

# Default optimization parameters
INITIAL_CAPITAL = 100_000_000
DEFAULT_GRANULARITY = 30  # trading days in the training window

# Covariance estimation
TRADING_DAYS_PER_YEAR = 252  # annualises daily variance
MIN_WINDOW_COVERAGE = 0.9    # a ticker needs prices on >= 90% of the window's trading dates
# "ledoit_wolf_cc" (shrink toward constant correlation) or "sample"
COV_ESTIMATOR = os.getenv("COV_ESTIMATOR", "ledoit_wolf_cc")

# Expected-return signal: "trailing" (return over the training window) or "momentum_12_1"
RETURN_ESTIMATOR = os.getenv("RETURN_ESTIMATOR", "trailing")
MOMENTUM_LOOKBACK_DAYS = 252  # ~12 months
MOMENTUM_SKIP_DAYS = 21       # skip the most recent ~1 month
