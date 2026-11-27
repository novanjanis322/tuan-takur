"""
Ingest Indonesia Stock Exchange (IDX) data from Yahoo Finance for local runs.

Steps (run all by default, or pick with --only):
  tickers    list every equity on Yahoo's JKT exchange     -> data/raw/idx_tickers.csv
  prices     daily OHLCV + adj close + actions per ticker  -> data/raw/prices_ohlcv.parquet
  info       sector / industry per ticker                  -> data/raw/industry_info.csv
  benchmark  ^JKLQ45 and ^JKSE daily                       -> data/raw/benchmark_lq45.csv, data/raw/benchmarks_raw.parquet
  report     per-ticker coverage                           -> data/raw/ingest_report.csv
  export     files in the shape the pipeline reads         -> data/local/adj_close_price.csv, data/local/info.csv

Prices and info are resumable: finished batches / tickers are skipped on rerun.

Usage:
  python scripts/ingest_idx_data.py
  python scripts/ingest_idx_data.py --only prices --batch-size 50
  python scripts/ingest_idx_data.py --only report export
"""
import argparse
import json
import logging
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from yfinance import EquityQuery

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LOCAL_DIR = DATA_DIR / "local"
CHUNK_DIR = RAW_DIR / "_chunks"

TICKERS_CSV = RAW_DIR / "idx_tickers.csv"
PRICES_PARQUET = RAW_DIR / "prices_ohlcv.parquet"
INFO_CSV = RAW_DIR / "industry_info.csv"
BENCHMARK_CSV = RAW_DIR / "benchmark_lq45.csv"
BENCHMARKS_RAW = RAW_DIR / "benchmarks_raw.parquet"
REPORT_CSV = RAW_DIR / "ingest_report.csv"
LOCAL_PRICES_CSV = LOCAL_DIR / "adj_close_price.csv"
LOCAL_INFO_CSV = LOCAL_DIR / "info.csv"

PRICE_COLUMNS = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Adj Close": "adj_close", "Volume": "volume",
    "Dividends": "dividends", "Stock Splits": "splits",
}
ALL_STEPS = ["tickers", "prices", "info", "benchmark", "report", "export"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ingest_idx")


def fetch_tickers() -> pd.DataFrame:
    """Page through Yahoo's equity screener for exchange JKT."""
    query = EquityQuery("eq", ["exchange", "JKT"])
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        res = _retry(lambda: yf.screen(query, size=250, offset=offset, sortField="ticker", sortAsc=True))
        total = res.get("total", 0)
        quotes = res.get("quotes", [])
        if not quotes:
            break
        for q in quotes:
            first_ms = q.get("firstTradeDateMilliseconds")
            rows.append({
                "ticker": q.get("symbol"),
                "code": str(q.get("symbol", "")).removesuffix(".JK"),
                "name": q.get("longName") or q.get("shortName"),
                "quote_type": q.get("quoteType"),
                "currency": q.get("currency"),
                "market_cap": q.get("marketCap"),
                "first_trade_date": pd.to_datetime(first_ms, unit="ms").date() if first_ms else None,
            })
        offset += len(quotes)
        logger.info(f"Screener: {offset}/{total} tickers")
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df = df[df["ticker"].str.endswith(".JK") & (df["quote_type"] == "EQUITY")]
    return df.sort_values("ticker").reset_index(drop=True)


def _retry(fn, attempts: int = 4, base_sleep: float = 5.0):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise
            wait = base_sleep * (2 ** i)
            logger.warning(f"{type(e).__name__}: {str(e)[:150]} - retrying in {wait:.0f}s")
            time.sleep(wait)


def _download(tickers: list[str], start: str, end: str, threads) -> pd.DataFrame:
    """Download a batch and return it in long format."""
    raw = yf.download(
        tickers, start=start, end=end, interval="1d",
        auto_adjust=False, actions=True, group_by="ticker",
        multi_level_index=True, threads=threads, progress=False, timeout=30,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "ticker", *PRICE_COLUMNS.values()])
    long = raw.stack(level=0, future_stack=True).reset_index()
    long = long.rename(columns={"Date": "date", "Ticker": "ticker", **PRICE_COLUMNS})
    for col in PRICE_COLUMNS.values():
        if col not in long.columns:
            long[col] = pd.NA
    long = long[["date", "ticker", *PRICE_COLUMNS.values()]]
    long = long.dropna(subset=["open", "high", "low", "close", "adj_close"], how="all")
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
    return long


def fetch_prices(tickers: list[str], start: str, end: str, batch_size: int) -> None:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    missing: list[str] = []

    for n, batch in enumerate(batches, 1):
        chunk = CHUNK_DIR / f"batch_{n:04d}.parquet"
        meta = CHUNK_DIR / f"batch_{n:04d}.json"
        if chunk.exists() and meta.exists():
            missing += json.loads(meta.read_text())["missing"]
            continue
        logger.info(f"Prices batch {n}/{len(batches)} ({batch[0]} .. {batch[-1]})")
        df = _retry(lambda: _download(batch, start, end, threads=6))
        got = set(df["ticker"].unique())
        batch_missing = [t for t in batch if t not in got]
        df.to_parquet(chunk, index=False)
        meta.write_text(json.dumps({"tickers": batch, "missing": batch_missing}))
        missing += batch_missing
        time.sleep(1.5)

    # Second pass: retry tickers that came back empty, one at a time.
    retry_file = CHUNK_DIR / "retry.parquet"
    if missing and not retry_file.exists():
        logger.info(f"Retrying {len(missing)} empty tickers individually")
        frames = []
        for i, t in enumerate(missing, 1):
            try:
                df = _retry(lambda: _download([t], start, end, threads=False), attempts=3, base_sleep=3)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.warning(f"{t}: {e}")
            if i % 25 == 0:
                logger.info(f"Retry {i}/{len(missing)}")
            time.sleep(1.0)
        retry_df = pd.concat(frames, ignore_index=True) if frames else _download([], start, end, False)
        retry_df.to_parquet(retry_file, index=False)

    parts = [pd.read_parquet(p) for p in sorted(CHUNK_DIR.glob("*.parquet"))]
    prices = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    prices = prices.drop_duplicates(["date", "ticker"], keep="last").sort_values(["ticker", "date"])
    prices.to_parquet(PRICES_PARQUET, index=False)
    logger.info(f"Saved {len(prices):,} price rows for {prices['ticker'].nunique()} tickers -> {PRICES_PARQUET}")


def fetch_info(tickers: list[str]) -> None:
    done = pd.read_csv(INFO_CSV) if INFO_CSV.exists() else pd.DataFrame(columns=["ticker", "status"])
    finished = set(done.loc[done["status"] == "ok", "ticker"])
    todo = [t for t in tickers if t not in finished]
    logger.info(f"Industry info: {len(finished)} done, {len(todo)} to fetch")
    rows = done[done["status"] == "ok"].to_dict("records")

    for i, t in enumerate(todo, 1):
        row = {"ticker": t, "name": None, "sector": None, "industry": None, "status": "error"}
        try:
            info = _retry(lambda: yf.Ticker(t).info, attempts=3, base_sleep=3) or {}
            row.update(name=info.get("longName") or info.get("shortName"),
                       sector=info.get("sector"), industry=info.get("industry"), status="ok")
        except Exception as e:
            logger.warning(f"{t}: {str(e)[:150]}")
        rows.append(row)
        if i % 25 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(INFO_CSV, index=False)
            logger.info(f"Industry info {i}/{len(todo)}")
        time.sleep(0.4)
    pd.DataFrame(rows).to_csv(INFO_CSV, index=False)


def fetch_benchmarks(start: str, end: str) -> None:
    def download_nonempty(symbol: str) -> pd.DataFrame:
        df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False,
                         multi_level_index=False, progress=False, timeout=30)
        if df.empty:
            raise ValueError(f"empty result for {symbol}")
        return df

    frames = []
    for symbol in ["^JKLQ45", "^JKSE"]:
        try:
            df = _retry(lambda: download_nonempty(symbol), attempts=5)
        except ValueError:
            logger.warning(f"No data for {symbol}")
            continue
        df = df.reset_index().rename(columns={"Date": "date", **PRICE_COLUMNS})
        df["ticker"] = symbol
        frames.append(df)
        if symbol == "^JKLQ45":
            out = pd.DataFrame({"Date": pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"), "Price": df["close"]})
            out.dropna().to_csv(BENCHMARK_CSV, index=False)
            logger.info(f"Saved {len(out)} LQ45 rows -> {BENCHMARK_CSV}")
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(BENCHMARKS_RAW, index=False)


def build_report(tickers: pd.DataFrame) -> pd.DataFrame:
    prices = pd.read_parquet(PRICES_PARQUET)
    stats = prices.groupby("ticker").agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        rows=("date", "size"),
        null_adj_close=("adj_close", lambda s: int(s.isna().sum())),
        nonpositive_adj_close=("adj_close", lambda s: int((s <= 0).sum())),
        zero_volume_days=("volume", lambda s: int((s == 0).sum())),
    ).reset_index()
    report = tickers[["ticker", "name", "first_trade_date"]].merge(stats, on="ticker", how="left")
    if INFO_CSV.exists():
        info = pd.read_csv(INFO_CSV)[["ticker", "sector", "industry"]]
        report = report.merge(info, on="ticker", how="left")
    report["status"] = report["rows"].notna().map({True: "ok", False: "no_data"})
    report.to_csv(REPORT_CSV, index=False)

    ok = report[report["status"] == "ok"]
    dupes = prices.duplicated(["date", "ticker"]).sum()
    print("\n=== Ingest summary ===")
    print(f"Tickers listed (Yahoo JKT): {len(report)}")
    print(f"Tickers with prices:        {len(ok)}")
    print(f"Tickers with no data:       {(report['status'] == 'no_data').sum()}")
    print(f"Price rows:                 {len(prices):,}")
    print(f"Date range:                 {prices['date'].min().date()} .. {prices['date'].max().date()}")
    print(f"Tickers starting by 2010:   {(ok['first_date'] <= '2010-12-31').sum()}")
    print(f"Duplicate (date, ticker):   {dupes}")
    print(f"Null adj_close rows:        {int(ok['null_adj_close'].sum()):,}")
    print(f"Non-positive adj_close:     {int(ok['nonpositive_adj_close'].sum()):,}")
    if "industry" in report:
        print(f"Tickers missing industry:   {report['industry'].isna().sum()}")
    print(f"Report: {REPORT_CSV}")
    return report


def export_local() -> None:
    """Write files matching the BigQuery tables the pipeline reads."""
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(PRICES_PARQUET, columns=["date", "ticker", "adj_close"])
    prices = prices.dropna(subset=["adj_close"])
    prices = prices[prices["adj_close"] > 0]
    prices["date"] = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")
    prices.sort_values(["ticker", "date"]).to_csv(LOCAL_PRICES_CSV, index=False)
    logger.info(f"Saved {len(prices):,} rows -> {LOCAL_PRICES_CSV}")

    info = pd.read_csv(INFO_CSV)[["ticker", "industry"]]
    info = info[info["ticker"].isin(prices["ticker"].unique())].drop_duplicates()
    info.to_csv(LOCAL_INFO_CSV, index=False)
    logger.info(f"Saved {len(info)} rows -> {LOCAL_INFO_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest IDX stock data from Yahoo Finance")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=date.today().isoformat(), help="exclusive")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--only", nargs="+", choices=ALL_STEPS, default=ALL_STEPS)
    parser.add_argument("--refresh", action="store_true", help="discard resumable progress for prices/info")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(DATA_DIR / ".yf_cache"))

    if args.refresh:
        for p in CHUNK_DIR.glob("*"):
            p.unlink()
        INFO_CSV.unlink(missing_ok=True)

    if "tickers" in args.only or not TICKERS_CSV.exists():
        tickers = fetch_tickers()
        tickers.to_csv(TICKERS_CSV, index=False)
        logger.info(f"Saved {len(tickers)} tickers -> {TICKERS_CSV}")
    tickers = pd.read_csv(TICKERS_CSV)
    symbols = tickers["ticker"].tolist()

    # Every IDX symbol trades in Asia/Jakarta. Seeding yfinance's timezone cache
    # skips one lookup request per ticker, which often times out from here.
    tz_cache = yf.cache.get_tz_cache()
    for t in symbols + ["^JKLQ45", "^JKSE"]:
        if tz_cache.lookup(t) is None:
            tz_cache.store(t, "Asia/Jakarta")

    started = datetime.now()
    if "prices" in args.only:
        fetch_prices(symbols, args.start, args.end, args.batch_size)
    if "info" in args.only:
        fetch_info(symbols)
    if "benchmark" in args.only:
        fetch_benchmarks(args.start, args.end)
    if "report" in args.only:
        build_report(tickers)
    if "export" in args.only:
        export_local()
    logger.info(f"Done in {datetime.now() - started}")


if __name__ == "__main__":
    main()
