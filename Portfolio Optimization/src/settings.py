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
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Gurobi credentials
WLS_ACCESS_ID = os.getenv("WLS_ACCESS_ID")
WLS_SECRET = os.getenv("WLS_SECRET")
LICENSE_ID = int(os.getenv("LICENSE_ID"))

# Default optimization parameters
INITIAL_CAPITAL = 100_000_000
DEFAULT_GRANULARITY = 30
