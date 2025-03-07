import gurobipy as gp
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from gurobipy import GRB, quicksum
import logging
from .data_loader import DataLoader
from .settings import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PortfolioOptimizer:
    def __init__(self,
                 granularity: int,
                 start_date: Union[str, datetime, date],
                 volatility: float,
                 sector_limits: Optional[Dict[str, float]] = None) -> None:
        """
        Initialize PortfolioOptimizer with optimization parameters.

        Args:
            granularity (int): Number of days for the optimization window
            start_date (Union[str, datetime, date]): Start date for optimization.
                If string, must be in 'YYYY-MM-DD' format.
            volatility (float): Maximum annual volatility target for the portfolio (e.g., 0.2 for 20%)

        Attributes set:
            start_date (str): Start date in 'YYYY-MM-DD' format
            last_date (str): End of current period in 'YYYY-MM-DD' format
            end_date (str): Final optimization date in 'YYYY-MM-DD' format
            initial_capital (float): Initial investment amount
            optimized_portfolio (pd.DataFrame): Empty DataFrame for optimization results
        """
        load_dotenv()

        # Initialize Gurobi credentials
        self.WLSACCESSID = WLS_ACCESS_ID
        self.WLSSECRET = WLS_SECRET
        self.LICENSEID = LICENSE_ID

        self.granularity = granularity
        self.volatility = volatility

        # Convert dates to string format
        if isinstance(start_date, (datetime, date)):
            self.start_date = start_date.strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
        self.static_start_date = self.start_date

        # Calculate last date
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        self.last_date = (start_dt + relativedelta(months=1)).strftime('%Y-%m-%d')

        # Calculate end date
        self.end_date = datetime.today().replace(day=1).strftime('%Y-%m-%d')

        self.initial_capital = INITIAL_CAPITAL
        self.optimized_portfolio = pd.DataFrame()

        # Initialize data loader
        self.data_loader = DataLoader()

        self.sector_limits = sector_limits

    def load_data(self) -> None:
        """
        Load all necessary data for portfolio optimization.

        This method loads stock data, industry data, and benchmark data using the DataLoader.
        It also validates the loaded data using _validate_loaded_data method.

        Raises:
            ValueError: If any of the loaded datasets are empty
            Exception: If there's an error during data loading or validation

        Example:
            optimizer = PortfolioOptimizer(30, "2023-01-01")
            optimizer.load_data()
        """
        try:
            self.df = self.data_loader.load_stock_data(self.start_date)
            if self.df.empty:
                raise ValueError("No stock data retrieved from BigQuery")

            self.industry_df = self.data_loader.load_industry_data()
            if self.industry_df.empty:
                raise ValueError("No industry data retrieved from BigQuery")

            self.df_LQ45_benchmark = self.data_loader.load_benchmark_data(self.start_date)
            if self.df_LQ45_benchmark.empty:
                raise ValueError("No benchmark data loaded from CSV")

            self._validate_loaded_data()

        except Exception as e:
            raise Exception(f"Error in loading data: {str(e)}")

    def _validate_loaded_data(self) -> None:
        """
        Validate the structure and content of loaded data.

        Checks for required columns in stock data, industry data, and benchmark data.

        Raises:
            ValueError: If any required columns are missing from the datasets
        """
        if not all(col in self.df.columns for col in ['date', 'ticker', 'adj_close']):
            raise ValueError("Stock data missing required columns")

        if not all(col in self.industry_df.columns for col in ['ticker', 'industry']):
            raise ValueError("Industry data missing required columns")

        if not all(col in self.df_LQ45_benchmark.columns for col in ['Date', 'Price']):
            raise ValueError("Benchmark data missing required columns")

    def prepare_backtesting_data(self) -> None:
        """
        Prepare data for backtesting by filtering the relevant date range.

        Filters the stock data based on start_date and last_date. Only includes
        data points that fall within this window.

        Attributes set:
            prep_backtesting_data_df (pd.DataFrame): Filtered DataFrame containing:
                - date (str): Date in 'YYYY-MM-DD' format
                - ticker (str): Stock ticker
                - adj_close (float): Adjusted closing price
        """
        _df = self.df[
            (self.df["date"] >= self.start_date) &
            (self.df["date"] < self.last_date)
            ]
        self.prep_backtesting_data_df = _df

    def pnl_backtesting_simulation(self, is_benchmark: bool = False) -> Optional[pd.DataFrame]:
        """
        Perform PNL backtesting simulation for portfolio or benchmark.

        Args:
            is_benchmark (bool, optional): Whether to perform simulation for benchmark data.
                Defaults to False.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing backtesting results if is_benchmark=True,
                None otherwise. The DataFrame contains columns:
                - period (str): Year-month in 'YYYY-MM' format
                - datapoints (int): Number of data points used
                - ticker (str): Stock ticker or 'LQ45' for benchmark
                - allocations (float): Portfolio weight
                - pnl_percentage (float): Percentage return

        Example:
            df = optimizer.pnl_backtesting_simulation(is_benchmark=True)
            print(df.columns)
            ['period', 'datapoints', 'ticker', 'allocations', 'pnl_percentage']
        """
        if is_benchmark:
            _df = self.benchmark_df.copy()
            _df['date'] = pd.to_datetime(_df['date'])
        else:
            _df = self.prep_backtesting_data_df.copy()

        # Use resample for benchmark, groupby for portfolio
        _df = (
            (_df.resample('ME', on='date') if is_benchmark else _df.groupby("ticker"))
            .agg(
                open_price=("adj_close", "first"),
                close_price=("adj_close", "last")
            )
            .reset_index()
        )

        # Calculate monthly returns
        _df["monthly_return (%)"] = round(
            100 * (_df["close_price"] - _df["open_price"]) / _df["open_price"],
            2
        )

        if not is_benchmark:
            # Handle portfolio data
            self.monthly_return_list = _df["monthly_return (%)"].tolist()
            self.monthly_return_ticker = _df["ticker"].tolist()
            self.pnl_backtesting = _df
        else:
            # Handle benchmark data
            _df['ticker'] = "LQ45"
            _df['industry'] = "Benchmark"
            _df['period'] = _df['date'].dt.strftime('%Y-%m')
            _df['allocations'] = 1
            _df['datapoints'] = 0
            _df['pnl_percentage'] = _df['monthly_return (%)']
            _df = _df[['period', 'datapoints', 'ticker', 'industry', 'allocations', 'pnl_percentage']]
            return _df

    def prepare_benchmark_data(self) -> None:
        """
        Prepare benchmark data for analysis.

        Processes the LQ45 benchmark data through the following steps:
        1. Converting dates to proper format
        2. Filtering based on start date
        3. Renaming columns for consistency
        4. Cleaning numerical data

        Attributes set:
            benchmark_df (pd.DataFrame): Processed benchmark data with columns:
                - date (str): Date in 'YYYY-MM-DD' format
                - adj_close (float): Adjusted closing price
            benchmark_pnl_backtesting_df (pd.DataFrame): Benchmark PNL results with columns:
                - period (str): Year-month in 'YYYY-MM' format
                - datapoints (int): Number of data points
                - ticker (str): Always 'LQ45'
                - allocations (float): Always 1.0
                - pnl_percentage (float): Monthly return percentage
        """
        benchmark_df = self.df_LQ45_benchmark.copy()
        # Handle the date conversion properly
        benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'])
        _df = benchmark_df[
            (benchmark_df['Date'] >= pd.to_datetime(self.start_date)) &
            (benchmark_df['Date'] < pd.to_datetime(self.end_date))
            ]

        # Select and rename columns
        _df = _df[['Date', 'Price']].copy()
        _df.columns = ['date', 'adj_close']

        # Convert dates back to string format
        _df['date'] = _df['date'].dt.strftime('%Y-%m-%d')

        # Clean up the adj_close column
        if _df['adj_close'].dtype == object:
            _df['adj_close'] = _df['adj_close'].str.replace(',', '').astype(float)

        self.benchmark_df = _df
        self.benchmark_pnl_backtesting_df = self.pnl_backtesting_simulation(
            is_benchmark=True
        )

    def prepare_training_data(self) -> None:
        """
        Prepare training data for optimization.

        Filters and processes the stock data through:
        1. Selecting data before start_date
        2. Sorting by ticker and date
        3. Limiting to specified granularity
        4. Filtering based on available tickers if not at end date

        Attributes set:
            training_data_df (pd.DataFrame): Processed training data with columns:
                - date (str): Date in 'YYYY-MM-DD' format
                - ticker (str): Stock ticker
                - adj_close (float): Adjusted closing price
                - row_number (int): Sequential number within each ticker group
        """
        _df = self.df[self.df["date"] < self.start_date].sort_values(
            by=["ticker", "date"], ascending=[True, False]
        )
        _df["row_number"] = _df.groupby("ticker").cumcount() + 1
        _df = _df[_df["row_number"] <= self.granularity]
        if self.start_date != self.end_date:
            _df = _df[_df["ticker"].isin(self.monthly_return_ticker)]
        self.training_data_df = _df

    def calculate_cov_matrix(self) -> None:
        """
        Calculate the covariance matrix for portfolio optimization.

        Processes the training data through:
        1. Creating a pivot table of adjusted closing prices
        2. Handling missing values
        3. Calculating returns and covariance matrix
        4. Computing volatility statistics

        Attributes set:
            processed_tickers (set[str]): Set of tickers included in analysis
            cov_matrix (pd.DataFrame): Covariance matrix of stock returns
            cov_matrix_val (np.ndarray): Numpy array version of covariance matrix

        Notes:
            Prints volatility statistics including average, maximum, and minimum
            annual stock volatilities.
        """
        _df = self.training_data_df.sort_values(by="date").drop_duplicates(
            subset=["date", "ticker"]
        )
        _df["sequential_index"] = _df.groupby("ticker").cumcount()

        self.processed_tickers = set()

        _df_pivot = _df.pivot(index="sequential_index", columns="ticker", values="adj_close")
        _df_pivot = _df_pivot.dropna(how="all", axis=1)
        # _df_pivot = _df_pivot.ffill().bfill()
        monthly_returns = _df_pivot.pct_change().dropna()
        monthly_returns = monthly_returns.loc[:, (monthly_returns != 0).any(axis=0)]

        time_delta = pd.to_datetime(_df['date']).diff().mean()
        annual_factor = pd.Timedelta('365 days') / time_delta
        self.processed_tickers = set(monthly_returns.columns)

        cov_matrix = monthly_returns.cov() * np.sqrt(annual_factor)

        print("\nVolatility Statistics:")
        annual_vols = np.sqrt(np.diag(cov_matrix)) * 100
        print(f"Average Annual Stock Volatility: {np.mean(annual_vols):.2f}%")
        print(f"Max Annual Stock Volatility: {np.max(annual_vols):.2f}%")
        print(f"Min Annual Stock Volatility: {np.min(annual_vols):.2f}%")
        cov_matrix_val = cov_matrix.values
        self.cov_matrix = cov_matrix
        self.cov_matrix_val = cov_matrix_val

    def calculate_metrics(self) -> None:
        """
        Calculate portfolio metrics and returns for optimization.

        Processes the training data to compute various metrics and stores them
        for use in the optimization model.

        Attributes set:
            monthly_metrics_df (pd.DataFrame): DataFrame with monthly metrics
            positive_returns (pd.DataFrame): Subset of stocks with positive returns
            stock_names (List[str]): List of stock tickers
            stock_returns (List[float]): List of corresponding returns
            stock_industry (List[str]): List of industry classifications

        Notes:
            Performs validation checks for ticker consistency between
            metrics and covariance matrix calculations.
        """
        _df = self.training_data_df.sort_values(by="date")

        # Only process tickers that exist in the covariance matrix
        if hasattr(self, "processed_tickers"):
            _df = _df[_df["ticker"].isin(self.processed_tickers)]

        monthly_metrics = (
            _df.groupby("ticker")
            .agg(open_price=("adj_close", "first"), close_price=("adj_close", "last"))
            .reset_index()
        )

        monthly_metrics["monthly_return (%)"] = (
                100
                * (monthly_metrics["close_price"] - monthly_metrics["open_price"])
                / monthly_metrics["open_price"]
        )

        monthly_metrics = pd.merge(
            monthly_metrics, self.industry_df, on="ticker", how="left"
        )

        if len(monthly_metrics["ticker"].unique()) != len(self.processed_tickers):
            print(f"Warning: Mismatch in ticker counts:")
            print(f"Monthly metrics tickers: {len(monthly_metrics['ticker'].unique())}")
            print(f"Covariance matrix tickers: {len(self.processed_tickers)}")
            print(
                "Missing tickers:",
                set(self.processed_tickers) - set(monthly_metrics["ticker"]),
            )
            print(
                "Extra tickers:",
                set(monthly_metrics["ticker"]) - set(self.processed_tickers),
            )

        monthly_metrics = monthly_metrics.drop_duplicates(subset=["ticker"])
        positive_returns = monthly_metrics[monthly_metrics["monthly_return (%)"] > 0]
        stock_names = monthly_metrics["ticker"].tolist()
        stock_returns = monthly_metrics["monthly_return (%)"].tolist()
        self.stock_industry = monthly_metrics["industry"].tolist()
        self.monthly_metrics_df = monthly_metrics
        self.positive_returns = positive_returns
        self.stock_names = stock_names
        self.stock_returns = stock_returns

    def setup_gurobi_env(self) -> None:
        """
        Set up the Gurobi optimization environment.

        Initializes the Gurobi environment with the configured license parameters
        and creates a new optimization model.

        Raises:
            Exception: If Gurobi initialization fails
        """
        params = {
            "WLSACCESSID": self.WLSACCESSID,
            "WLSSECRET": self.WLSSECRET,
            "LICENSEID": self.LICENSEID,
        }
        env = gp.Env(params=params)
        self.model = gp.Model(env=env)

    def verify_portfolio_risk(self):
        """
        Perform comprehensive portfolio risk verification.

        Calculates and prints detailed risk analytics including:
        1. Portfolio-level volatility
        2. Individual stock contributions to risk
        3. Risk decomposition by position

        Returns:
            float: Annualized portfolio volatility as a decimal
                (e.g., 0.15 represents 15% volatility)

        Notes:
            Prints detailed risk analysis including weights, individual
            volatilities, and risk contributions for each position.
        """
        portfolio_weights = np.array([self.allocations[stock].X for stock in self.stock_names])

        portfolio_variance = portfolio_weights.T @ self.cov_matrix_val @ portfolio_weights
        portfolio_volatility = np.sqrt(portfolio_variance)

        print("\nPortfolio Risk Analysis:")
        print(f"Annual Portfolio Volatility: {portfolio_volatility * 100:.2f}%")

        print("\nRisk Decomposition:")
        total_risk = 0
        for i, (stock, weight) in enumerate(zip(self.stock_names, portfolio_weights)):
            if weight > 0.01:
                stock_vol = np.sqrt(self.cov_matrix_val[i][i])
                contribution = weight * stock_vol
                total_risk += contribution
                print(f"{stock}:")
                print(f"  Weight: {weight * 100:.1f}%")
                print(f"  Individual Volatility: {stock_vol * 100:.1f}%")
                print(f"  Risk Contribution: {contribution * 100:.1f}%")

        print(f"\nTotal Risk Contributions: {total_risk * 100:.2f}%")
        print(f"Portfolio Volatility: {portfolio_volatility * 100:.2f}%")

        return portfolio_volatility

    def create_sector_mapping(self) -> dict:
        """
        Create a mapping of industries to broader sectors using keyword matching.

        Maps each industry to one of the following sectors:
        - Financial
        - Technology
        - Energy
        - Healthcare
        - Industrial
        - Materials
        - Other (default for unmatched industries)

        Returns:
            Dict[str, str]: Dictionary mapping industry names (keys) to
                sector classifications (values)

        Notes:
            Uses keyword matching to categorize industries. Industries that
            don't match any sector keywords are classified as 'Other'.
            Logs the final mapping for review.
        """
        sector_keywords = {
            'Financial': ['bank', 'insurance', 'capital', 'financial', 'invest', 'credit', 'fund', 'securities'],
            'Technology': ['tech', 'software', 'computer', 'semiconductor', 'electronic', 'digital', 'media', 'telecom'],
            'Energy': ['oil', 'gas', 'energy', 'coal', 'mine', 'utility', 'power'],
            'Healthcare': ['health', 'hospital', 'pharma', 'medical', 'drug', 'biotech'],
            'Industrial': ['industry', 'manufacture', 'construct', 'machinery', 'equipment', 'engineering'],
            'Materials': ['material', 'chemical', 'steel', 'cement', 'paper', 'mineral', 'metal']
        }

        unique_industries = set(self.industry_df['industry'].dropna())

        industry_to_sector = {}

        for industry in unique_industries:
            if pd.isna(industry):
                industry_to_sector[industry] = 'Other'
                continue

            industry_lower = str(industry).lower()
            matched_sector = 'Other'

            for sector, keywords in sector_keywords.items():
                if any(keyword in industry_lower for keyword in keywords):
                    matched_sector = sector
                    break

            industry_to_sector[industry] = matched_sector

        logger.info("Industry to Sector mapping created:")
        logger.info(industry_to_sector)
        return industry_to_sector

    def define_optimization_model(self) -> None:
        """
        Define the portfolio optimization model with constraints.

        Sets up a Gurobi optimization model with:
        1. Maximum return objective function
        2. Risk limit constraint (20% annual volatility)
        3. Full investment constraint (weights sum to 1)
        4. Minimum stocks constraint (at least 2 stocks)
        5. Minimum allocation to regional banks (10%)

        Attributes Set:
            allocations: Portfolio weight variables
            select_vars: Binary variables for stock selection

        Raises:
            Exception: If model definition fails
        """
        allocations = self.model.addVars(
            self.stock_names, lb=0.0, ub=1.0, name="allocation"
        )
        select_vars = self.model.addVars(
            self.stock_names, vtype=GRB.BINARY, name="select"
        )

        self.model.setObjective(
            quicksum(
                allocations[stock] * self.stock_returns[self.stock_names.index(stock)]
                for stock in self.stock_names
            ),
            GRB.MAXIMIZE,
        )

        target_annual_volatility = self.volatility
        risk_expression = quicksum(
            self.cov_matrix_val[i][j] * allocations[stock_i] * allocations[stock_j]
            for i, stock_i in enumerate(self.stock_names)
            for j, stock_j in enumerate(self.stock_names)
        )
        self.model.addConstr(
            risk_expression <= target_annual_volatility ** 2,
            "RiskLimit"
        )
        # Individual stock contribution constraints
        for i, stock in enumerate(self.stock_names):
            stock_variance = self.cov_matrix_val[i][i]
            self.model.addConstr(
                allocations[stock] * np.sqrt(stock_variance) <= target_annual_volatility,
                f"IndividualRiskLimit_{stock}"
            )

        min_allocation = 0.01
        max_allocation = 0.3
        for stock in self.stock_names:
            self.model.addConstr(
                allocations[stock] <= max_allocation * select_vars[stock],
                "MaxAllocation"
            )
            self.model.addConstr(
                allocations[stock] >= min_allocation * select_vars[stock],
                "MinAllocation"
            )

        self.model.addConstr(
            quicksum(allocations[stock] for stock in self.stock_names) == 1,
            "TotalInvestment",
        )

        self.model.addConstr(
            quicksum(select_vars[stock] for stock in self.stock_names) <= int(1 / min_allocation),
            "MaxStocks"
        )

        sector_mapping = self.create_sector_mapping()
        stock_sectors = [sector_mapping.get(industry, 'Other') for industry in self.stock_industry]

        for sector, limit in self.sector_limits.items():
            sector_stocks = [stock for i, stock in enumerate(self.stock_names) if stock_sectors[i] == sector]
            if sector_stocks:
                self.model.addConstr(
                    quicksum(allocations[stock] for stock in sector_stocks) <= limit,
                    name=f"Sector_Limit_{sector}"
                )

        self.allocations = allocations
        self.select_vars = select_vars

    def run_optimization(self) -> None:
        """
        Execute the optimization model.

        Runs the Gurobi optimizer on the defined model. The optimization results
        can be accessed through the model's attributes after completion.

        Raises:
            Exception: If optimization fails
        """
        self.model.optimize()

    def optimization_result(self) -> None:
        """
        Process and store optimization results.

        Extracts optimal portfolio weights and corresponding PNL percentages
        for positions with allocation > 0.5%. Only processes results if an
        optimal solution is found (model.status == GRB.OPTIMAL).

        Attributes set:
            optimized_portfolio (pd.DataFrame): Updated with new results containing:
                - period (str): Year-month of optimization
                - datapoints (int): Number of days in granularity
                - risk (float): Target volatility
                - ticker (str): Stock symbol
                - industry (str): Industry classification
                - allocations (float): Portfolio weight (0-1)
                - pnl_percentage (float): Profit/Loss percentage

        Notes:
            Prints optimization status and portfolio allocation summary.
            If no optimal solution is found, prints failure message with model status.
        """
        if self.model.status == GRB.OPTIMAL:
            new_rows = []
            positions = []

            total_allocation = sum(self.allocations[stock].X for stock in self.stock_names)
            print(f"\nOptimization Solution Status:")
            print(f"Total allocation: {total_allocation * 100:.2f}%")
            print(f"Number of selected stocks: {sum(self.select_vars[stock].X > 0.5 for stock in self.stock_names)}")

            min_reporting_threshold = 0.005  # 0.5%
            for stock in self.allocations:
                weight = self.allocations[stock].X
                if weight > min_reporting_threshold:
                    try:
                        pnl_percentage = self.monthly_return_list[
                            self.monthly_return_ticker.index(stock)
                        ]
                    except:
                        pnl_percentage = 0
                    try:
                        industry = self.stock_industry[self.stock_names.index(stock)]
                    except:
                        industry = "Unknown"

                    positions.append({
                        "ticker": stock,
                        "weight": weight,
                        "industry": industry,
                        "pnl_percentage": pnl_percentage
                    })

            if positions:
                total_weight = sum(pos["weight"] for pos in positions)

                for pos in positions:
                    weight = round(pos["weight"] / total_weight, 2)
                    new_rows.append({
                        "period": datetime.strptime(self.start_date, '%Y-%m-%d').strftime("%Y-%m"),
                        "datapoints": int(self.granularity),
                        "risk": float(self.volatility),
                        "ticker": pos["ticker"],
                        "industry": pos["industry"],
                        "allocations": weight,
                        "pnl_percentage": float(round(pos["pnl_percentage"], 2)),
                    })

                df = pd.DataFrame(new_rows)
                total = df["allocations"].sum()
                if total != 1.0:
                    idx_max = df["allocations"].idxmax()
                    df.loc[idx_max, "allocations"] = round(df.loc[idx_max, "allocations"] + (1.0 - total), 2)

                print(f"\nPortfolio Allocation Summary:")
                print(f"Number of positions: {len(df)}")
                print(f"Total allocation in results: {df['allocations'].sum() * 100:.2f}%")

                self.optimized_portfolio = pd.concat(
                    [self.optimized_portfolio, df],
                    ignore_index=True
                )
        else:
            print(f"Optimization failed to find a solution with status: {self.model.status}")

    def backtesting_simulation(self) -> None:
        """
        Perform backtesting simulation for portfolio and benchmark.

        Simulates portfolio performance by:
        1. Tracking capital changes over time
        2. Calculating monthly P&L for each position
        3. Computing end-of-period values
        4. Comparing portfolio performance with benchmark

        Attributes set:
            final_result (pd.DataFrame): Combined results containing:
                - All columns from optimized_portfolio
                - allocations_idr (int): Position size in IDR
                - pnl_idr (int): Profit/Loss in IDR
                - end_of_period_value (int): Position value at period end
                Both portfolio and benchmark results are included, sorted by
                period and ticker.
        """
        df_backtesting = [self.optimized_portfolio, self.benchmark_pnl_backtesting_df]
        for df in df_backtesting:
            capital = self.initial_capital
            month_val = 0
            previous_period = None

            # Sort the dataframe by period to ensure correct chronological processing
            df.sort_values("period", inplace=True)

            for index, row in df.iterrows():
                current_period = row["period"]
                if previous_period is not None and current_period != previous_period:
                    capital += month_val
                    month_val = 0

                # Calculate allocations and P&L
                allocations_idr = int(capital * row["allocations"])
                pnl_percentage = float(row["pnl_percentage"])  # Ensure it's a float
                pnl_idr = round(allocations_idr * (pnl_percentage / 100))
                end_of_period_value = allocations_idr + pnl_idr

                # Update dataframe
                df.loc[index, "allocations_idr"] = allocations_idr
                df.loc[index, "pnl_idr"] = pnl_idr
                df.loc[index, "end_of_period_value"] = end_of_period_value

                # Accumulate month's P&L
                month_val += pnl_idr
                previous_period = current_period

            # Handle the last month's P&L
            capital += month_val

        # Ensure the dataframes are properly sorted before concatenating
        self.optimized_portfolio.sort_values(["period", "ticker"], inplace=True)
        self.benchmark_pnl_backtesting_df.sort_values(
            ["period", "ticker"], inplace=True
        )

        self.final_result = pd.concat(
            [self.optimized_portfolio, self.benchmark_pnl_backtesting_df],
            ignore_index=True,
        )

    def find_tickers(self, find_date: str) -> Optional[pd.DataFrame]:
        """
        Find portfolio allocations for a specific date.

        Args:
            find_date (str): Date string in 'YYYY-MM' format

        Returns:
            Optional[pd.DataFrame]: DataFrame containing portfolio allocations for the
                specified date, or None if no portfolio exists. Columns include:
                - period: Year-month
                - datapoints: Number of days
                - ticker: Stock symbol
                - allocations: Portfolio weight
                - pnl_percentage: Profit/Loss percentage

        Example:
            optimizer.find_tickers("2023-01")
            # Returns DataFrame with January 2023 allocations
        """
        find_date = datetime.strptime(find_date, "%Y-%m")
        if len(self.optimized_portfolio) == 0:
            print("No optimized portfolio yet")
        else:
            optimized_portfolio_df = pd.DataFrame(self.optimized_portfolio)
            optimized_portfolio_df = optimized_portfolio_df[
                optimized_portfolio_df["period"] == find_date.strftime("%Y-%m")
                ]
            return optimized_portfolio_df


def run_optimization_pipeline(granularity: int,
                              volatility: float,
                              start_date: Optional[Union[str, datetime]] = None,
                              sector_limits: Optional[Dict[str, float]] = None) -> 'PortfolioOptimizer':
    """
    Run the complete portfolio optimization pipeline.

    Args:
        granularity (int): Number of days for the optimization window,
        start_date (Optional[Union[str, datetime]], optional): Start date for optimization.
            Can be either a string in 'YYYY-MM-DD' format or a datetime object.
            If None, defaults to first day of previous month. Defaults to None.
        volatility: Maximum annual volatility for the portfolio
        sector_limits: Dictionary of sector limits for portfolio allocation

    Returns:
        PortfolioOptimizer: Optimized portfolio instance containing the results
            and final portfolio allocations

    Raises:
        ValueError: If granularity or start_date are invalid
        Exception: If optimization process fails

    Example:
        optimizer = run_optimization_pipeline(granularity=30, start_date="2023-01-01", volatility=0.2)
        results_df = optimizer.final_result
    """

    if start_date is None:
        start_date = (
                datetime.now().replace(day=1) -
                relativedelta(months=1)
        ).strftime('%Y-%m-%d')
    elif isinstance(start_date, (datetime, date)):
        start_date = start_date.strftime('%Y-%m-%d')

    optimizer = PortfolioOptimizer(granularity, start_date, volatility, sector_limits)
    optimizer.load_data()
    optimizer.prepare_benchmark_data()
    logger.info(f"loading data for {optimizer.start_date} has been completed")
    logger.info(f"starting optimization process")
    while optimizer.start_date <= optimizer.end_date:
        print(f"\nRunning optimization for {optimizer.start_date}\n")

        logger.info(f"prepare backtesting data for {optimizer.start_date}")
        optimizer.prepare_backtesting_data()

        logger.info(f"performing pnl backtesting simulation for {optimizer.start_date}")
        optimizer.pnl_backtesting_simulation()

        logger.info(f"prepare training data for {optimizer.start_date}")
        optimizer.prepare_training_data()

        logger.info(f"calculate covariance matrix for {optimizer.start_date}")
        optimizer.calculate_cov_matrix()

        logger.info(f"calculate metrics for {optimizer.start_date}")
        optimizer.calculate_metrics()

        logger.info(f"setup gurobi environment for {optimizer.start_date}")
        optimizer.setup_gurobi_env()

        logger.info(f"define optimization model for {optimizer.start_date}")
        optimizer.define_optimization_model()

        logger.info(f"run optimization for {optimizer.start_date}")
        optimizer.run_optimization()

        logger.info(f"optimization result for {optimizer.start_date}")
        optimizer.optimization_result()

        logger.info(f"verify portfolio risk for {optimizer.start_date}")
        optimizer.verify_portfolio_risk()

        # Update dates for next iteration
        current_date = datetime.strptime(optimizer.start_date, '%Y-%m-%d')
        optimizer.start_date = (
                current_date + relativedelta(months=1)
        ).strftime('%Y-%m-%d')
        optimizer.last_date = (
                datetime.strptime(optimizer.last_date, '%Y-%m-%d') +
                relativedelta(months=1)
        ).strftime('%Y-%m-%d')

    optimizer.backtesting_simulation()
    return optimizer


if __name__ == "__main__":
    optimizer = run_optimization_pipeline()
