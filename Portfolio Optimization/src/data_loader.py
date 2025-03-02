import logging
import pandas as pd
import yfinance as yf
from google.cloud import bigquery
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Optional

from .settings import RAW_DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, cache_dir: Path = Path('data/cache')):
        """
        Initialize DataLoader with BigQuery client and cache directory.

        Args:
            cache_dir (Path, optional): Directory for caching data.
                Defaults to 'data/cache'.
        """
        logger.info(f"Initializing DataLoader with cache directory: {cache_dir}")
        self.client = bigquery.Client()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DataLoader initialization complete")

    def _get_cache_path(self, start_date: str) -> Path:
        """
        Generate cache file path based on start date.

        Args:
            start_date (str): Date string in 'YYYY-MM-DD' format

        Returns:
            Path: Path object pointing to the cache file
        """
        if not start_date:
            logger.error("start_date cannot be None or empty")
            raise ValueError("start_date cannot be None or empty")
        cache_path = self.cache_dir / f"stock_data_{start_date}.csv"
        logger.debug(f"Generated cache path: {cache_path}")
        return cache_path

    def _save_to_cache(self, df: pd.DataFrame, start_date: str) -> None:
        """Save data to cache file
        Args:
            df (pd.DataFrame): DataFrame saved into cache
            start_date (str): Date string in 'YYYY-MM-DD' format
        Returns:
            None
        """
        cache_path = self._get_cache_path(start_date)
        logger.info(f"Saving data to cache: {cache_path}")
        logger.debug(f"DataFrame shape being cached: {df.shape}")
        df.to_csv(cache_path, index=False)
        logger.info("Data successfully saved to cache")


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
            logger.info(f"Loading data from cache: {cache_path}")
            df = pd.read_csv(cache_path)
            logger.debug(f"Loaded DataFrame shape: {df.shape}")
            return df
        logger.info(f"Cache not found at: {cache_path}")
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
        logger.info(f"Loading stock data for start_date: {start_date}")
        if not start_date:
            logger.error("start_date cannot be None or empty")
            raise ValueError("start_date cannot be None or empty")

        cached_data = self._load_from_cache(start_date)
        if cached_data is not None:
            logger.info("Using cached stock data")
            return cached_data

        logger.info("Cache miss - querying fresh data from BigQuery")
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        date_query = (start_dt - relativedelta(years=3)).strftime('%Y-%m-%d')
        logger.debug(f"Query date range: {date_query} to current")
        query = f"""
          WITH get_data AS (
            SELECT
              *,
              ROW_NUMBER() OVER (PARTITION BY ticker, DATE_TRUNC(`date`, MONTH) ORDER BY `date`) AS row_num
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
            row_num = 1 OR adj_close != LAG(adj_close) OVER (
              PARTITION BY ticker ORDER BY `date`
            )
          ORDER BY
            ticker, date ASC;
          """
        logger.debug("Executing BigQuery query")
        df = self.client.query(query).to_dataframe()
        logger.info(f"Retrieved {len(df)} rows from BigQuery")

        logger.debug("Processing date format")
        df["date"] = pd.to_datetime(df["date"])
        df["date"] = df["date"].dt.strftime('%Y-%m-%d')
        df["adj_close"] = df["adj_close"].astype(float)
        logger.debug(f"Final DataFrame shape: {df.shape}")

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
        logger.info("Loading industry data")
        cache_path = self.cache_dir / "industry_data.csv"

        if cache_path.exists():
            logger.info("Loading industry data from cache")
            df = pd.read_csv(cache_path)
            logger.debug(f"Loaded industry data shape: {df.shape}")
            return df

        logger.info("Cache miss - querying industry data from BigQuery")
        query = """
                  SELECT
                    ticker, industry
                  FROM
                    km-data-dev.L0_yahoofinance.info
                  group by ticker, industry
                  """
        df = self.client.query(query).to_dataframe()
        logger.info(f"Retrieved {len(df)} industry records from BigQuery")

        logger.info("Saving industry data to cache")
        df.to_csv(cache_path, index=False)
        return df

    def load_benchmark_data(self, start_date: str) -> pd.DataFrame:
        """
        Load daily benchmark (LQ45) data and automatically update if current data is missing.

        Args:
            start_date (str): Date string in 'YYYY-MM-DD' format

        Returns:
            pd.DataFrame: DataFrame containing daily benchmark data with columns:
                - Date (str): Date in 'YYYY-MM-DD' format
                - Price (float): Daily benchmark closing price
        """
        logger.info("Loading benchmark data")
        cache_path = self.cache_dir / f"benchmark_data_{start_date}.csv"
        current_date = pd.Timestamp.now()

        def fetch_yfinance_data(start: str) -> pd.DataFrame:
            """Helper function to fetch daily data from yfinance"""
            logger.info(f"Fetching yfinance data from {start}")
            try:
                df = yf.download("^JKLQ45", start=start, end=current_date, interval="1d")
                if df.empty:
                    logger.error("No data retrieved from yfinance")
                    raise ValueError("No data retrieved from yfinance")

                df = df.reset_index()

                # Ensure we have the expected columns
                if 'Date' not in df.columns or 'Close' not in df.columns:
                    logger.error("Expected columns not found in yfinance data")
                    raise ValueError("Missing expected columns in yfinance data")

                # Select and rename columns
                df = df[['Date', 'Close']].rename(columns={
                    'Date': 'Date',
                    'Close': 'Price'
                })

                # Convert and format dates
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

                return df

            except Exception as e:
                logger.error(f"Error in fetch_yfinance_data: {str(e)}")
                raise Exception(f"Failed to fetch yfinance data: {str(e)}")

        try:
            if cache_path.exists():
                logger.info("Loading benchmark data from cache")
                df = pd.read_csv(cache_path)
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

                # Check for updates
                df['Date'] = pd.to_datetime(df['Date'])
                latest_date = df['Date'].max()
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                days_difference = (current_date - latest_date).days
                if days_difference > 1:
                    logger.info(f"Data missing for last {days_difference} days. Fetching updates.")
                    new_data = fetch_yfinance_data(latest_date.strftime('%Y-%m-%d'))

                    if not new_data.empty:
                        new_data = new_data[~new_data['Date'].isin(df['Date'])]
                        if not new_data.empty:
                            logger.info(f"Adding {len(new_data)} new data points")
                            df = pd.concat([df, new_data], ignore_index=True)
                            df = df.sort_values('Date').reset_index(drop=True)
                            df.to_csv(cache_path, index=False)
                            logger.info("Cache updated with new data")
                        else:
                            logger.info("No new unique data points to add")
                    else:
                        logger.info("No new data retrieved")
            else:
                logger.info("No cached data found. Fetching complete dataset.")
                df = fetch_yfinance_data(start_date)
                df = df.dropna()
                df.to_csv(cache_path, index=False)
                logger.info("New data cached successfully")

            # Final processing
            df = df.sort_values('Date').reset_index(drop=True)
            logger.info(f"Final dataset shape: {df.shape}")
            return df[['Date', 'Price']]

        except Exception as e:
            logger.error(f"Error in benchmark data processing: {str(e)}")
            raise