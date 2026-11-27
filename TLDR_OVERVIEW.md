# tuan-takur: Overview

**Last updated:** 2026-11-27

> **This is a personal project. Nothing here goes to deployment or production.**
> It continues the work left at the end of the internship. The goal is to fix and improve
> **the algorithm** (the Gurobi portfolio optimization in `src/optimizer.py`) and to run it locally.
> The API, the Pub/Sub worker, GKE and BigQuery are kept only as reference. They are not maintained or deployed,
> and their review findings are out of scope.

## Status at a glance
| Item | Status |
|---|---|
| Local IDX price data (all listed stocks, 2010 to today) | Done: 840 stocks, 2.18 M daily rows, 2010-01-04 to 2026-11-26 |
| Industry data and the LQ45 / IHSG benchmarks | Done |
| The algorithm reads local data (no BigQuery needed) | Done and tested |
| LQ45 baseline run (old algorithm) | Done. Saved in `data/processed/baseline_lq45/` |
| Algorithm fixes (C3, H1, M2, M4, H3, H6/M1, L7, L2) | **Done.** Re-run on the same LQ45 settings (`data/processed/fixed_lq45/`) |
| Full 840-stock run | Not yet. The Gurobi license needs to be decided first |
| API / security / worker / deployment items | **Out of scope:** not deploying |

## Before vs after the fixes
LQ45 stocks, 2021-01 to 2026-10, 30-day window, 20% volatility target, no sector limits.

| | Old algorithm | Fixed algorithm | LQ45 index |
|---|---|---|---|
| Total return | +198% | +142% | −45% |
| Per year (CAGR) | 20.6% | 16.4% | −9.7% |
| Actual volatility (target 20%) | **41.2%** | **25.4%** | 17.4% |
| Worst drop from a peak | −34.5% | −25.8% | −52.5% |
| Sharpe ratio (0% risk-free) | 0.64 | 0.72 | −0.49 |
| Average number of stocks | 5.1 | 7.8 | 45 |
| Largest single weight (cap 30%) | 31% | 30% | n/a |

What this means:
- **The risk target now roughly works.** Before, the portfolio ran at twice its target. It is still about 5 points over, because a 30-day sample covariance with 45 stocks underestimates risk. Shrinking the covariance would help; that is still open.
- **Returns are lower but cleaner.** Some of the old return came from bugs: lining up prices from different days, and the look-ahead filter.
- **The LQ45 numbers are now correct.** The old code reported −41%; the real index fell 44.9%.
- **Both runs share one bias.** They use today's 45 LQ45 members for the whole period, including stocks that only joined after big rallies (for example CUAN). That makes both runs look better than a real 2021 investor could have done.
- **Some months now hold cash.** In 2 months (2025-05 and the current month, 2026-11), no LQ45 portfolio could meet 20% volatility, so the run holds cash instead of crashing. **The current-month recommendation at 20% is therefore "cash".** Try `--volatility 0.25` or `0.3`.

## What the algorithm fixes changed
- **Risk scale (C3):** daily variance × 252 days, instead of an arbitrary square-root factor.
- **Dates line up (H1):** returns are matched by date, and days with an unchanged price are kept. This also changed the BigQuery query.
- **No look-ahead (M2):** stocks are chosen using only data before each month.
- **Full monthly return (M4):** P&L runs from the previous month-end close, for both the portfolio and LQ45.
- **No crash on a bad month (H3):** if no portfolio meets the constraints, that month holds cash.
- **Start dates (H6/M1):** any start date is moved to the 1st of its month. A future date gives a clear error.
- **Weights (L7):** reported weights stay true to the solver (0.01% precision) and never exceed the 30% cap.
- **Gurobi session (L2):** one license session per run instead of one per month.
- **Granularity (L9):** now means "trading days in the window", the same for every stock.

## Where things are
Folder: `Portfolio Optimization/data/`
- `raw/prices_ohlcv.parquet`: the full raw price history. `raw/ingest_report.csv`: data quality per stock.
- `processed/baseline_lq45/`: the old-algorithm run (results, log, summary, and a copy of the old code).
- `processed/fixed_lq45/`: the fixed-algorithm run. `processed/comparison_lq45.json`: the comparison above.

## How to run
All commands run from `Portfolio Optimization/`:
- LQ45 only:
  ```
  $env:UNIVERSE_FILE="data/local/universe_lq45.csv"
  .venv-local\Scripts\python.exe main.py --start-date 2021-01-01 --granularity 30 --volatility 0.2
  ```
- Compare two runs: `.venv-local\Scripts\python.exe scripts\summarize_run.py OLD.csv --compare NEW.csv`
- Refresh the data (about 20 minutes): `.venv-local\Scripts\python.exe scripts\ingest_idx_data.py --refresh`

## Open questions
- **Gurobi license, for the full 840 stocks.** The free license allows about 100 stocks. Possible fixes:
  - switch to "semi-continuous" variables (about 200 stocks on the free license);
  - a free academic license;
  - a WLS license.
- **Volatility estimate.** Add covariance shrinkage, or use a longer window, so actual risk lands closer to the target?
- **Sector mapping.** 480 of 840 stocks fall into "Other". The sector names can be changed freely, since no API depends on them.
- **Expected return (M3).** It is one trailing 30-day return, and turnover is 74% a month. Keep it, or switch to something steadier?
