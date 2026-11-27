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
| Covariance shrinkage (Ledoit–Wolf) | **Done.** Now the default. The volatility target is close to accurate (see below) |
| Model rewrite (L1, L5, L6) | **Done.** Same results; free license now handles 200 stocks (was 100); model builds 80× faster |
| Sector mapping | **Done.** Uses Yahoo's 11 real sectors (was 480 of 840 stocks in "Other"); misspelled limits now raise an error |
| Full 840-stock run | Not yet. Needs a full Gurobi license, or a universe of ≤ 200 stocks |
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

## Covariance shrinkage (2026-11-27)
**Problem:** a 30-day window gives only 29 daily returns for 45 stocks. The risk estimate is then very noisy, and the algorithm leans into combinations that look safe but are not. The result was 25% actual volatility against a 20% target.

**Fix:** the algorithm now uses Ledoit–Wolf shrinkage toward a constant-correlation target. Each stock keeps its own volatility, the noisy stock-to-stock correlations are pulled toward the average, and the data decides how hard to pull. A simulation test confirmed the method: it roughly halved the estimation error.

LQ45, 2021-01 to 2026-10, 20% target:

| Risk estimate | Window | Actual volatility | Bias (1.0 = right) | CAGR | Worst drop | Sharpe | Turnover/month |
|---|---|---|---|---|---|---|---|
| Old (sample) | 30 days | 25.4% | 1.28 | 16.4% | −25.8% | 0.72 | 74% |
| Shrinkage | 30 days | 22.6% | 1.14 | 11.4% | −21.8% | 0.59 | 73% |
| Old (sample) | 60 days | 25.7% | 1.29 | 24.0% | −29.5% | 0.96 | 54% |
| **Shrinkage** | **60 days** | **20.9%** | **1.05** | 24.4% | −21.5% | 1.15 | 52% |
| Old (sample) | 90 days | 25.7% | 1.29 | 17.8% | −37.4% | 0.76 | 44% |
| Shrinkage | 90 days | 23.0% | 1.16 | 18.0% | −33.6% | 0.83 | 42% |

What this means:
- **Shrinkage helps at every window length.** Actual risk always moves closer to the target. That is the reliable result.
- **60 days with shrinkage is the closest to the target** (20.9% against 20%). Treat the window choice with care: with 70 months of data, differences of about 2 points in volatility are within noise. The return numbers are noisier still, so don't pick a window because of its CAGR.
- **A longer window lowers turnover** (74% → about 50%), because the risk estimate changes less from month to month.
- The default window is still 30 days. Use `--granularity 60` to try the best setting.

## Model rewrite (2026-11-27)
- **Change:** each stock's weight is now one "semi-continuous" variable: either 0, or between 1% and 30%. It replaces a weight plus a yes/no "selected" variable and a stock-count limit that never did anything. Mathematically it is the same model.
- **Same results:** 70 of 71 months had identical portfolios; the remaining month moved 0.23% of weight, within solver tolerance.
- **Free Gurobi license:** it now solves up to **200 stocks**. The old model was refused at 150.
- **Speed:** the model is built with Gurobi's matrix functions instead of a Python loop, so at 840 stocks building takes 0.1 s instead of 8 s per month.
- **New dependency:** `scipy`, needed by Gurobi's matrix functions. The local environment is now listed in `Portfolio Optimization/requirements-local.txt`.

## Sector mapping (2026-11-27)
- **Real sectors:** the algorithm now uses each stock's Yahoo sector directly. Before, it guessed the sector from keywords in the industry name, and 480 of 840 stocks ended up in "Other".
- **Sector names for `--sector-limits`:** Basic Materials, Communication Services, Consumer Cyclical, Consumer Defensive, Energy, Financial Services, Healthcare, Industrials, Real Estate, Technology, Utilities.
  ```
  main.py --start-date 2021-01-01 --sector-limits '{"Energy": 0.2, "Basic Materials": 0.2}'
  ```
- **Misspellings caught:** a misspelled sector name now stops the run with the list of valid names. Before, it was silently ignored.
- **Exact caps:** reported weights never go over a stock or sector cap, not even by 0.01%.
- **Sector in the results:** each position in the results now has a `sector` column.
- **Why limits matter:** without them, the algorithm put up to **79%** in Energy in one month. With Energy and Basic Materials each capped at 20% (LQ45, 30-day window): return 9.8%/yr against 11.4%, volatility 22.2% against 22.6%, worst drop −31.8% against −21.8%.

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
- `processed/fixed_lq45/`: the fixed-algorithm run.
- `processed/cov_<estimator>_g<window>/` and `processed/covariance_comparison_lq45.csv`: the shrinkage runs. `processed/comparison_lq45.json`: the comparison above.

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
- **Gurobi license, for the full 840 stocks.** The free license now allows up to 200 stocks. For all 840, get a free academic license or a WLS license. Otherwise, pick a universe of at most 200 stocks, for example IDX80 or the 200 most liquid.
- **Default window.** Switch the default from 30 to 60 days? It came out best on risk accuracy, but that was judged on this same backtest.
- **Expected return (M3).** It is one trailing 30-day return, and turnover is 74% a month. Keep it, or switch to something steadier?
