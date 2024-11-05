from google.cloud import bigquery
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .settings import RAW_DATA_DIR

class DataLoader:
    def __init__(self, cache_dir: Path = Path('data/cache')):
        """
        Initialize DataLoader with BigQuery client and cache directory.

        Args:
            cache_dir (Path, optional): Directory for caching data.
                Defaults to 'data/cache'.
        """
        self.client = bigquery.Client()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, start_date: str) -> Path:
        """
        Generate cache file path based on start date.

        Args:
            start_date (str): Date string in 'YYYY-MM-DD' format

        Returns:
            Path: Path object pointing to the cache file
        """
        if not start_date:
            raise ValueError("start_date cannot be None or empty")
        return self.cache_dir / f"stock_data_{start_date}.csv"

    def _save_to_cache(self, df: pd.DataFrame, start_date: str) -> None:
        """Save data to cache file
        Args:
            df (pd.DataFrame): DataFrame saved into cache
            start_date (str): Date string in 'YYYY-MM-DD' format
        Returns:
            None
        """
        cache_path = self._get_cache_path(start_date)
        df.to_csv(cache_path, index=False)

    def _load_from_cache(self, start_date: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if it exists
        Args:
            start_date (str): Date string in 'YYYY-MM-DD' format
        Returns:
            Optional[pd.DataFrame]: DataFrame loaded from cache or None if not found

        """
        cache_path = self._get_cache_path(start_date)
        if cache_path.exists():
            return pd.read_csv(cache_path)
        return None

    def load_stock_data(self, start_date: str) -> pd.DataFrame:
        """
        Load stock data with caching mechanism.

        Args:
            start_date (str): Date string in 'YYYY-MM-DD' format

        Returns:
            pd.DataFrame: DataFrame containing stock data with columns:
                - date (str): Date in 'YYYY-MM-DD' format
                - ticker (str): Stock ticker
                - adj_close (float): Adjusted closing price

        Raises:
            Exception: If there's an error querying BigQuery or processing data
        """
        if not start_date:
            raise ValueError("start_date cannot be None or empty")
        # Try to load from cache first
        cached_data = self._load_from_cache(start_date)
        if cached_data is not None:
            print("Loading data from cache...")
            return cached_data

        print("Querying fresh data from BigQuery...")
        # If not in cache, query from BigQuery
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        date_query = (start_dt - relativedelta(years=2)).strftime('%Y-%m-%d')

        query = f"""
          WITH get_data AS (
            SELECT
              *
            FROM
              km-data-dev.L0_yahoofinance.adj_close_price
            WHERE
              `date` >= '{date_query}'
              AND `date` < DATE_TRUNC(CURRENT_DATE('Asia/Jakarta'), MONTH)
            ORDER BY
              ticker, date
          )
          SELECT
            *
          FROM
            get_data
          QUALIFY
            adj_close != LAG(adj_close) OVER (
            PARTITION BY ticker ORDER BY `date`
            )
          ORDER BY
            ticker, date ASC
          """
        df = self.client.query(query).to_dataframe()
        df["date"] = pd.to_datetime(df["date"])
        df["date"] = df["date"].dt.strftime('%Y-%m-%d')
        df["adj_close"] = df["adj_close"].astype(float)

        # Save to cache
        self._save_to_cache(df, start_date)

        return df

    def load_industry_data(self) -> pd.DataFrame:
        """
        Load industry classification data with caching mechanism.

        Returns:
            pd.DataFrame: DataFrame containing industry data with columns:
                - ticker (str): Stock ticker
                - industry (str): Industry classification

        Raises:
            Exception: If there's an error querying BigQuery or processing data
        """
        cache_path = self.cache_dir / "industry_data.csv"

        # Try to load from cache
        if cache_path.exists():
            print("Loading industry data from cache...")
            return pd.read_csv(cache_path)

        print("Querying industry data from BigQuery...")
        query = """
          SELECT
            ticker, industry
          FROM
            km-data-dev.L0_yahoofinance.info
          group by ticker, industry
          """
        df = self.client.query(query).to_dataframe()

        # Save to cache
        df.to_csv(cache_path, index=False)

        return df

    def load_benchmark_data(self) -> pd.DataFrame:
        """
        Load benchmark (LQ45) data with caching mechanism.

        Returns:
            pd.DataFrame: DataFrame containing benchmark data with columns:
                - Date (str): Date in 'YYYY-MM-DD' format
                - Price (float): Benchmark price

        Raises:
            FileNotFoundError: If benchmark CSV file is not found
            Exception: If there's an error processing the benchmark data
        """
        cache_path = self.cache_dir / "benchmark_data.csv"

        # Try to load from cache
        if cache_path.exists():
            print("Loading benchmark data from cache...")
            return pd.read_csv(cache_path)

        print("Loading benchmark data from CSV...")
        try:
            benchmark_path = RAW_DATA_DIR / "LQ45_benchmark.csv"
            if not benchmark_path.exists():
                raise FileNotFoundError(f"Benchmark file not found at: {benchmark_path}")

            df = pd.read_csv(benchmark_path)

            # Handle date formatting
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
            except:
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as e:
                    raise Exception(f"Could not parse dates in benchmark data: {str(e)}")

            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

            # Clean up Price column
            if df['Price'].dtype == object:
                df['Price'] = df['Price'].str.replace(',', '').astype(float)

            # Save to cache
            df.to_csv(cache_path, index=False)

            return df
        except Exception as e:
            raise Exception(f"Error loading benchmark data: {str(e)}")
