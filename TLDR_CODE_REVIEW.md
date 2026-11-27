# tuan-takur: Code Review

**Date:** 2026-11-23 · **Scope:** all project source and deploy config (the `.venv` is excluded) · **Method:** I read the code only. Nothing was run or changed.

## Status at a glance

| Severity | Count | Headline |
|---|---|---|
| Critical | 3 | SQL injection, any logged-in user can read any user's data, and the risk (volatility) math is wrong |
| High | 7 | The CLI is broken. Failed jobs disappear without a trace. One infeasible month crashes the whole run. The cache is never refreshed. |
| Medium | 14 | Backtest bias and misalignment, worker concurrency, secrets kept in a ConfigMap, fragile Docker build |
| Low | 12 | Dead code, stale docstrings, hygiene (a 5,947-file `.venv` is committed), no tests |

**Fix status (2026-11-27):**
- **Scope:** this is now a personal project with **no deployment**. Only the algorithm (`src/optimizer.py` and its data and backtest) is being worked on.
- **Fixed in the algorithm:** C3, H1, H2, H3, H6, M1 (the date part), M2, M4, L2, L7, L9, and L10 (the `LICENSE_ID` part). Before/after results are in `TLDR_OVERVIEW.md`.
- **Still open, algorithm:** M3 (objective choice), L1, L5, L6, the sector mapping, and covariance noise.
- **Out of scope (deployment/API, not being deployed):** C1, C2, H4, H5, H7, M5 to M14.

**Bottom line:** fix the three Critical items before any public use. The volatility target the user sets does not currently control the real portfolio risk, so both the recommendations and the backtest numbers are unreliable.

---

## Project overview

The project produces portfolio recommendations for Indonesian equities (benchmark: LQ45) using a Gurobi optimizer.

```
Client ─► API (FastAPI on Cloud Run, API/app.py)
            ├─ POST /optimize/v2   → publishes job to Pub/Sub
            ├─ GET  /optimizers/{id}           → reads result from BigQuery
            └─ GET  /users/{id}/portfolio-history
Pub/Sub ─► Worker (GKE + KEDA, gke_deployment/app.py)
            └─ run_optimization_pipeline()  (src/optimizer.py)
                 ├─ DataLoader: prices + industries from BigQuery, LQ45 from yfinance (CSV cache)
                 ├─ For each month from start_date to today:
                 │    train on last N price rows → covariance + trailing return
                 │    Gurobi: max return s.t. vol ≤ target, 1–30%/stock, sector caps
                 │    record next month's realised P&L
                 └─ Compound backtest vs LQ45 → write one nested row to BigQuery
CLI: main.py → same pipeline → CSV
```

---

## Critical

| # | Where | Problem | Impact | Fix |
|---|---|---|---|---|
| C1 | `API/app.py:161`, `:344` | `generation_id` (URL path) and `user_id` (URL path) are pasted into the SQL with f-strings. | SQL injection against BigQuery. A crafted path such as `x' OR '1'='1` returns every user's rows. | Use `bigquery.QueryJobConfig(query_parameters=[ScalarQueryParameter(...)])`. |
| C2 | `API/app.py:403-522` | The Firebase token is verified, but `user_claims` is never used. `/users/{user_id}/...` and `/optimizers/{task_id}` do not check that the data belongs to the caller. `/optimize/v2` accepts any `user_id` in the request body. | Any logged-in user can read or create portfolios as any other user (IDOR). | Take the user from `user_claims["email"]` or `uid` instead of the path or body, and filter queries by it. |
| C3 | `src/optimizer.py:303-307` | The covariance matrix is annualised with `cov * sqrt(annual_factor)`. Variance scales linearly (`* annual_factor`), not by the square root. On top of that, `time_delta` is taken over rows sorted by date across **all tickers**, so most gaps are 0 and the factor grows with the number of tickers. | The `RiskLimit` constraint and the printed "annual volatility" are on an arbitrary scale that changes with the size of the ticker universe. The user's `volatility` input does not mean what it says. | Compute returns aligned by date and use `cov * periods_per_year`, with periods measured per ticker (or a fixed 252). |

## High

| # | Where | Problem | Impact | Fix |
|---|---|---|---|---|
| H1 | `src/optimizer.py:293-300` + `src/data_loader.py:127-130` | Returns are pivoted on `sequential_index` (row position per ticker) instead of on date. The SQL `QUALIFY` also drops every day the price did not change, so each ticker's series has different dates. | The covariance pairs up returns from different days. Correlations are wrong and diversification is mis-estimated. | Pivot on `date`, keep zero-return days, and handle NaN explicitly. |
| H2 | `main.py:115-118`, `src/optimizer.py:571`, `:862` | The CLI calls `run_optimization_pipeline` without the required `volatility` argument, which raises TypeError. Even with it supplied, `sector_limits=None` crashes at `self.sector_limits.items()`. The `__main__` block in `optimizer.py` passes no arguments at all. | The local CLI entry point cannot run. | Add `--volatility`, and default `sector_limits` to `{}` or guard against `None`. |
| H3 | `src/optimizer.py:616/676` → `:844-845` (`verify_portfolio_risk`, line 416) | When Gurobi does not reach `OPTIMAL` (infeasible, time limit, suboptimal), the code prints a message and continues. `verify_portfolio_risk` then reads `.X`, which raises an exception. | One infeasible month aborts the whole multi-year job. Infeasibility is easy to trigger: sector limits that sum to less than 1, a low volatility target, or fewer than 4 eligible stocks (30% cap). | Skip or carry forward the month when there is no solution. Accept `SUBOPTIMAL` when `SolCount > 0`. Validate inputs up front (for example, the sector limits must sum to at least 1). |
| H4 | `gke_deployment/app.py:225-228` + `API/app.py:477-482` | The worker acks the message on any failure and only logs it. No failure status is ever written. The API returns 404 "Task not found" whether the job is still running, has failed, or never existed. | Users poll forever and get no error. Failures are invisible unless someone reads the logs. | Write a status row (`processing`, `failed` with a reason, or `success`) and return it from `GET /optimizers/{id}`. Consider a Pub/Sub dead-letter topic. |
| H5 | `src/data_loader.py:30-45, 101-104, 162` | The stock-data cache is keyed only on `start_date`. The industry cache has no key at all. Neither ever expires, even though the query depends on `CURRENT_DATE`. | A long-lived pod (or a CLI rerun) keeps serving stale prices after the month rolls over, so the latest recommendation is trained on old data. | Include the data cut-off month in the key, or add a TTL. |
| H6 | `src/optimizer.py:705` (with `validators.py`) | A `start_date` inside the current month but after day 1 passes validation. The main loop then never runs, and `optimized_portfolio` stays an empty DataFrame with no columns. | `sort_values("period")` raises KeyError and the job crashes. | Require `start_date` to be before `end_date`, or normalise it to the 1st. |
| H7 | `gke_deployment/app.py:257-260` | `subscribe()` is called without `FlowControl`. The defaults allow about 1,000 outstanding messages on 10 callback threads. | One pod (2 CPU, 4 Gi) can run up to 10 Gurobi and pandas jobs at once. Expect OOM kills, Gurobi WLS session limits, and redelivery loops. KEDA's "1 message per pod" sizing has no effect. | Pass `flow_control=pubsub_v1.types.FlowControl(max_messages=1)`. |

## Medium

| # | Where | Problem | Fix |
|---|---|---|---|
| M1 | `src/optimizer.py:814, 849-855, 653` | A `start_date` that is not the 1st gives mid-month to mid-month periods labelled `YYYY-MM`. The benchmark uses calendar months (`resample('ME')`), so the comparison is misaligned. The "latest" period then also contains the LQ45 row, and `/optimizers/{id}` returns **LQ45 as a recommended ticker**, because it does not filter `industry == 'Benchmark'` the way the history endpoint does. | Force day 1. Filter the benchmark consistently in `get_latest_allocation`. |
| M2 | `src/optimizer.py:267-268` | The training universe is limited to tickers that have price data in the *coming* month. | Look-ahead (survivorship) bias inflates backtest results. Use only information available at `start_date`. |
| M3 | `src/optimizer.py:348-352, 520-526` | "Expected return" is the raw trailing point-to-point return over N rows, and the objective maximises it. | Very noisy, momentum-chasing allocations. Consider a mean of returns, shrinkage, or a risk-adjusted objective. (This is a modelling choice; it is flagged, not a bug.) |
| M4 | `src/optimizer.py:169-182` | Monthly P&L is first close to last close **within** the month. The move from the previous month-end to the first trading day is never counted. | Every month drops about one day of return, for both the portfolio and the benchmark. Use the previous month-end close as the open. |
| M5 | `src/data_loader.py:203-219, 249` | yfinance 0.2.49 (the locked version) returns MultiIndex columns by default. The code assumes flat columns. The first write to the cache produces a two-row header (the `str.replace(',', '')` at `optimizer.py:237-238` appears to be a workaround for this), and incremental updates concatenate flat and MultiIndex frames, so **new benchmark rows likely become NaN**. *I could not verify this without running the code.* | Pass `multi_level_index=False` to `yf.download`. |
| M6 | `API/app.py:278` | When the `API_KEY` env var is not set and the header is missing, `None != None` is False, so the check passes. The comparison is also not constant-time. | Fail at startup if `API_KEY` is unset. Use `secrets.compare_digest`. |
| M7 | `API/app.py:248-253`, `:305-307` | Blocking calls (`future.result()`, `auth.verify_id_token`) run inside `async def`. | They stall the event loop. Make these handlers plain `def` or use `run_in_threadpool`. |
| M8 | `gke_deployment/app.py:166-172` | BigQuery load retry uses `if_exception_type(Exception)` with a 600 s timeout. | Schema or permission errors are retried for 10 minutes, and a retry after a timeout can insert duplicate rows. Retry only transient errors. |
| M9 | `gke_deployment/configmap-template.yaml:9-11` | `WLS_SECRET` and the other Gurobi license secrets are stored in a ConfigMap. | Move them to a Kubernetes Secret or Secret Manager. |
| M10 | `gke_deployment/Dockerfile:11-15, 17, 24, 26` | `uv sync` ignores the activated `/opt/venv` and installs into `/app/.venv`. The build only works because of the hand-written `PYTHONPATH`; `/opt/venv`, the `PATH` line and the `find` cleanup do nothing. The base images `uv:latest` and `distroless/python3` are unpinned, while the entrypoint hard-codes `/usr/bin/python3.11`. | The build breaks when distroless moves to a newer Python. Pin the images and use `UV_PROJECT_ENVIRONMENT=/opt/venv`. |
| M11 | `API/app.py:341-346`, `:151-165` | Every poll and history call runs `SELECT *` on the whole table and pulls the full nested backtest arrays, even though only the last period is used. | Full table scans on every request cost money and add latency. Cluster or partition the table on `user_id` and `generation_id`, and select only the needed fields. |
| M12 | `API/app.py:55-61`, `:382-389` | CORS is set to `"*"` with `allow_credentials=True`, which makes Starlette mirror any origin, so the explicit origins in the list mean nothing. The catch-all `OPTIONS` route is shadowed by the middleware. | List the real origins and delete the custom preflight handler. |
| M13 | `API/app.py:260, 315, 445, 522`, `:306`, `:177` | Raw exception text is returned to clients, the first 20 characters of the auth token are logged, and full user rows are `print`ed. | Return generic messages and drop the token and row logging. |
| M14 | Whole repo | There are no tests (`tests/*` is even gitignored), and API requirements and image tags are unpinned. | Add a few unit tests for the covariance, P&L, and validator logic, and pin dependencies. |

## Low (hygiene, dead code, small inefficiencies)

| # | Where | Note |
|---|---|---|
| L1 | `src/optimizer.py:529-533, 522` | The risk term is built in a Python N² loop, and `stock_names.index()` is called inside a loop (O(N²)). Use Gurobi's matrix API (`addMVar`, `x @ Q @ x`). |
| L2 | `src/optimizer.py:391-397` | A new Gurobi `Env` (a new WLS session) is created every month and never disposed. Create one `Env` per job and call `model.dispose()`. |
| L3 | `src/optimizer.py:707-722` | `iterrows` plus per-row `.loc` writes in the backtest. This can be vectorised with `groupby` and `cumprod`. |
| L4 | `API/app.py:392-393` | A new `PublisherClient` (a new gRPC channel) is created on every request. Make it a module-level singleton. |
| L5 | `src/optimizer.py:563-566` | The `MaxStocks` constraint (≤ 100) has no effect: the 1% minimum with weights summing to 1 already implies it. Constraint names `MaxAllocation` and `MinAllocation` are repeated for every stock. |
| L6 | `src/optimizer.py:499-505` | The docstring describes a 20% volatility cap, a minimum of 2 stocks and a 10% regional-bank floor. None of these exist in the code. |
| L7 | `src/optimizer.py:651-666` | Weights are rounded to whole percents, positions under 0.5% are dropped, and the leftover is added to the largest weight. The reported weights can drift from the solved weights and slightly breach the sector or risk caps. |
| L8 | `src/optimizer.py:633, 637`, `API/app.py:36`, `optimizer.py:101` | Bare `except:` clauses, and `raise Exception(...)` without `from e`, lose the error type and traceback. |
| L9 | `src/utils/validators.py:61-69` | The specific granularity messages ("must be positive", "≤ 365") are swallowed and replaced with "must be a valid integer". "Granularity" means price rows, not days, despite the docs; `main.py` even says 1–252. |
| L10 | `src/settings.py:19`, `.env.example` | `int(os.getenv("LICENSE_ID"))` raises a cryptic TypeError at import when the variable is unset. `.env.example` is missing `PROJECT_ID`, `PUBSUB_TOPIC`, `GOOGLE_CLOUD_PROJECT` and `PUBSUB_SUBSCRIPTION`. |
| L11 | Dead code | The commented-out block at `main.py:124-182`; unused imports `uuid` and `current_timestamp`; `find_tickers`, `positive_returns`, `static_start_date`, `optimization_results`, `security`, `fastapi_thread` and `Future`; the unused `start_date`/`volatility` locals at `gke_deployment/app.py:201-202`; and the `sector_limits` parameter, which is accepted but never saved (`:78`). Star imports (`from .settings import *`, `from src.utils.validators import *`) silently provide `load_dotenv` and `datetime`. Library modules call `logging.basicConfig`, and `print` is mixed with `logger`. |
| L12 | Repo hygiene | `Portfolio Optimization/.venv` is committed: 5,947 files from a Python 3.9 environment on another machine, while the project needs Python 3.11 or later. `.gitignore` lists `venv/` but not `.venv/`. It also ignores `pdb.yaml`, which is tracked anyway. The project name in `pyproject.toml` has a typo (`optimizatiom`) and points to a `README.md` that does not exist. The worker image also pulls in dependencies only the API needs (`fastapi[all]`, `granian`, `firebase-admin`). The SQL has a useless `ORDER BY` inside the CTE and a hard-coded **dev** project (`km-data-dev`). |

---

## Suggested fix order

1. **C1, C2, M6:** close the security holes (parameterised SQL, derive the user from the token, require the API key).
2. **C3, H1, M2, M4:** correct the risk and return math, then re-run the backtest. The published backtest chart was produced with these bugs in place.
3. **H3, H4, H6, H7:** make the worker robust (handle infeasible months, add a job status, validate inputs, set flow control to 1).
4. **H2, H5, M5:** restore the CLI, cache invalidation and the benchmark fetch.
5. Everything else as cleanup, together with a small test suite.

## Open questions

- Is dropping unchanged-price days in the SQL (`QUALIFY`) meant to filter out suspended or illiquid stocks? If so, that needs a separate filter; it should not change the return series.
- Should granularity mean trading days or calendar days? The code, the docs and the CLI disagree.
- Is `km-data-dev` the intended production source?
