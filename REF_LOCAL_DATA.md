# REF: local IDX data for tuan-takur

Context for agents working on the local (non-BigQuery) run. Created 2026-11-27.

## SCOPE: personal project, no deployment or production (stated by the user, 2026-11-27)
- The user is continuing this project after the end of their internship. **Nothing will be deployed.** There is no production environment to protect or keep compatible with.
- **"The algorithm"** means the Gurobi portfolio optimization in `src/optimizer.py`, together with its data prep and backtest. That is the focus of all work.
- `API/`, `gke_deployment/`, Pub/Sub, GKE and the BigQuery writer are reference only. Do not spend effort on them, do not flag "production impact", and do not keep their contracts (response shape, sector names, the `CASH` row) unless the user asks.
- The open review items in those areas (C1, C2, H4, H5, H7, M6–M13) are out of scope.

## Why
Production reads `km-data-dev.L0_yahoofinance.adj_close_price` / `.info` from BigQuery. There are no GCP credentials on this PC, so the same Yahoo Finance data is pulled locally with `Portfolio Optimization/scripts/ingest_idx_data.py`.

## Environment
- `Portfolio Optimization/.venv-local`: Python 3.11.15, built with uv. The committed `.venv` is a broken Python 3.9 environment from another machine; do not use it.
- Key packages: pandas 2.2.2, yfinance 1.8.0, gurobipy 14.0.0 (free pip license), pyarrow, duckdb.
- Run from `Portfolio Optimization/`: `.venv-local\Scripts\python.exe scripts\ingest_idx_data.py [--only ...]`

## Sources
| Data | Source | Notes |
|---|---|---|
| Ticker universe | `yf.screen(EquityQuery("eq", ["exchange","JKT"]))` | 840 equities on 2026-11-27. IDX lists ~950. The idx.co.id API returns 403 (Cloudflare) to scripts. |
| Prices | `yf.download(auto_adjust=False, actions=True)` from 2010-01-01 | Batches of 50, threads=6. Empty tickers are retried one at a time. |
| Industry | `yf.Ticker(t).info` → `sector`, `industry` | Yahoo strings, e.g. "Banks - Regional". |
| Benchmark | `^JKLQ45` (Close) and `^JKSE` | LQ45 history starts 2010-01-04. |

Network note: Yahoo calls from this PC often time out (curl 28) on yfinance's 10 s timezone lookup. Fix: the script pre-seeds the yfinance tz cache with `Asia/Jakarta` for every symbol. After that, 0 tickers failed. Index downloads can still return empty results intermittently, so the benchmark step retries until it gets data.

## Ingest result (run of 2026-11-27, 20.5 min)
- 840/840 tickers have prices. 2,182,270 rows from 2010-01-04 to 2026-11-26. 0 duplicates, 0 null, 0 non-positive adj_close.
- 313 tickers start in 2010; the others listed later (peak 76 new in 2023).
- Industry: 839/840 (LAPD.JK has none). Yahoo sectors: Industrials 147, Consumer Cyclical 131, Consumer Defensive 118, Financial Services 99, Real Estate 83, Basic Materials 81, Energy 53, Communication Services 42, Healthcare 40, Technology 34, Utilities 11.
- `^JKLQ45`: 4,103 rows from 2010-01-04 to 2026-11-26.
- Local-mode smoke test, `PortfolioOptimizer(30, "2024-01-01", 0.2).load_data()`: passes validation with no GCP credentials. The parity filter keeps 781,218 of 1,037,536 rows in the 3-year window (25% dropped as unchanged-price days).
- `create_sector_mapping()` on Yahoo industries: Other 480, Financial 91, Materials 69, Energy 59, Industrial 55, Technology 48, Healthcare 38. The keywords miss consumer, real-estate and most industrial names.

## Files (`Portfolio Optimization/data/`, gitignored)
| File | Schema |
|---|---|
| `raw/idx_tickers.csv` | ticker, code, name, quote_type, currency, market_cap, first_trade_date |
| `raw/prices_ohlcv.parquet` | date, ticker, open, high, low, close, adj_close, volume, dividends, splits |
| `raw/industry_info.csv` | ticker, name, sector, industry, status |
| `raw/benchmark_lq45.csv` | Date, Price (the format `load_benchmark_data` expects) |
| `raw/benchmarks_raw.parquet` | OHLCV for ^JKLQ45 and ^JKSE |
| `raw/ingest_report.csv` | per ticker: first_date, last_date, rows, null/non-positive adj_close, zero-volume days, sector, industry, status |
| `raw/_chunks/` | resumable batch files; delete them or pass `--refresh` to re-download |
| `local/adj_close_price.csv` | date, ticker, adj_close (matches the BigQuery table; NaN and ≤0 rows dropped) |
| `local/info.csv` | ticker, industry (matches the BigQuery `info` table) |

Tickers keep the `.JK` suffix in every file.

## Code changes for local runs
- `src/settings.py`: `DATA_SOURCE` (env var, default `bigquery`), `LOCAL_DATA_DIR`. The WLS credentials and `LICENSE_ID` are optional (the old `int(None)` crashed at import).
- `src/data_loader.py`: `DataLoader(source=...)`. In local mode no BigQuery client is created. `_load_local_stock_data` uses the SQL window: `date >= start-3y`, `date < 1st of the current Asia/Jakarta month`. Since the H1 fix on 2026-11-27, the `QUALIFY` unchanged-price filter is removed from **both** the local loader and the BigQuery SQL. Industry comes from `local/info.csv`, the benchmark from `raw/benchmark_lq45.csv`. There is no caching in local mode.
- `UNIVERSE_FILE` env var (local mode only): a CSV with a `ticker` column that limits prices and industry. `data/local/universe_lq45.csv` holds the 45 LQ45 members effective 2026-11-02 to 2027-01-29 (source: Kontan, 2026-10-31).
- `src/optimizer.py`: `sector_limits or {}`. `setup_gurobi_env` falls back to `gp.Env()` when the WLS credentials are absent. The `__main__` block passes arguments.
- `main.py`: `--volatility` (default 0.2) and `--sector-limits` (JSON).
- `.env` (local, gitignored): `DATA_SOURCE=local`.

## Known limits
- Survivorship bias: only currently listed tickers. Stocks delisted in 2010–2026 are absent.
- About 110 IDX listings are missing from Yahoo's screener.
- Yahoo IDX data quality: gaps, stale prices, odd adjustments on small caps. Check `ingest_report.csv`.
- The parity filter for H1 (dropping unchanged-price days) was removed on 2026-11-27, after the baseline run.

## Algorithm fixes (2026-11-27)
The baseline uses the pre-fix code, with a snapshot in `data/processed/baseline_lq45/code_snapshot/`. The fixes in `src/optimizer.py`:
- **C3:** `cov = daily_returns.cov() * TRADING_DAYS_PER_YEAR` (252, in settings).
- **H1:** `prepare_training_data` takes the last `granularity` trading dates before start (shared calendar). `calculate_cov_matrix` pivots on date, keeps tickers with ≥ `MIN_WINDOW_COVERAGE` (0.9) of the window's dates, fills short gaps forward, drops tickers with missing data at the window start, and drops zero-variance tickers. `calculate_metrics` takes its returns from the same `window_prices`, with an assert that the order matches the covariance columns.
- **M2:** the `isin(monthly_return_ticker)` look-ahead filter is removed. A selected stock with no price in the holding month earns 0%.
- **M4:** the portfolio opens at the last close before start. The benchmark uses month-end closes with `shift(1)`. `prepare_benchmark_data` keeps history before start.
- **H3:** `optimization_result` uses `SolCount > 0`. With no solution it writes a `CASH` row (allocation 1, pnl 0) and sets `self.solved = False`, and the loop skips `verify_portfolio_risk`.
- **H6/M1:** `__init__` moves the start to day 1, raises an error if start > the current month's 1st, and requires granularity ≥ 3.
- **L7:** weights are rounded to 4 decimal places, the noise threshold is 1e-6, and the rounding residual goes to the position with the most headroom under `self.max_allocation`.
- **L2:** one `gp.Env` per optimizer; the previous model is disposed each month.
- Not changed: the objective (M3), the binaries / MaxStocks (L5), the N² risk loop (L1), sector mapping, and the API/worker.

LQ45 result, 2021-01 to 2026-10, g=30, vol=0.2 (`data/processed/comparison_lq45.json`):
- Realised volatility: 41.2% before, 25.4% after.
- CAGR: 20.6% before, 16.4% after.
- Max drawdown: −34.5% before, −25.8% after.
- LQ45 benchmark: −41.0% before, −44.9% after (the true index move is −44.9%).
- Cash months (INFEASIBLE, status 3): 2025-05 and 2026-11.
- Rolling 12-month realised volatility after the fixes: median 23.9%. It stays above the target because of the noisy 30-day sample covariance (N=45 > T=29).

## Covariance shrinkage (2026-11-27)
- `src/covariance.py`: `estimate_covariance(returns, method)`, where method is `ledoit_wolf_cc` (Ledoit & Wolf 2003, constant-correlation target, analytic intensity δ = clip((π−ρ)/γ / T, 0, 1)) or `sample`. It returns `(cov, delta)` in the input's units; `calculate_cov_matrix` annualises with ×252.
- `settings.COV_ESTIMATOR` (env var, default `ledoit_wolf_cc`).
- The results CSV gains `ex_ante_volatility` (√(w'Σw) of the solved weights) and `shrinkage_intensity`, both 0 / the month's δ for CASH rows. `summarize_run.py` reports `risk_forecast.bias_statistic` = std(r_t / (σ_t/√12)), where 1.0 means an unbiased forecast.
- Simulation check (1-factor model, N=45, T=29, 300 trials): δ is about 0.91; relative Frobenius error 0.63 (sample) vs 0.36 (LW); min-variance true/believed vol 2.33 vs 1.39; δ=0.03 at T=5000. The shrunk matrix is always positive definite.
- LQ45 grid (vol 0.2, 2021-01 to 2026-10), bias statistic for sample → LW: g30 1.28 → 1.14, g60 1.29 → 1.05, g90 1.29 → 1.16. Realised vol: 25.4 → 22.6, 25.7 → 20.9, 25.7 → 23.0. Mean δ: 0.72 / 0.62 / 0.55. Full table in `data/processed/covariance_comparison_lq45.csv`.
- The window was chosen in-sample, so the g=60 edge is weak evidence (vol SE about ±1.8 pts with 70 months). `DEFAULT_GRANULARITY` stays 30 (confirmed by the user on 2026-11-27).

## Model rewrite: L1, L5, L6 (2026-11-27)
- `define_optimization_model` uses one `addMVar(n, vtype=GRB.SEMICONT, lb=0.01, ub=0.3)`. The objective is `mu @ w`. Constraints: `w @ Σ @ w <= vol²`, `sqrt(diag Σ) * w <= vol`, `w.sum() == 1`, and sector masks `@ w <= limit`. The binaries, the redundant MaxStocks constraint and the duplicate constraint names are gone. Weights are read from `self.weights.X` (array ordered like `stock_names`); `self.allocations` and `self.select_vars` no longer exist.
- The gurobipy matrix API needs **scipy** (added to `.venv-local` and `requirements-local.txt`).
- Equivalence check against the binary model (LW, g30, LQ45): 70/71 periods identical, max per-period weight shift 0.23%, total return 87.28% vs 87.26%.
- Free (size-limited) license, synthetic test: semi-continuous solves N=150 and N=200 and is refused at N=250; binary is refused at N=150. Gurobi presolve turns semi-continuous into binaries internally, but the license counts the model as built.
- Build time at N=840: 0.10 s (matrix API) vs 8.12 s (old Python loop).
- Free Gurobi pip license: 200 variables when the model has quadratic terms, and the model uses 2 variables per stock, so at most ~100 stocks per month. The full universe will fail with "Model too large". The user wants to discuss this later; no universe filter has been added.
