import numpy as np
import pandas as pd
from typing import List, Dict


class PortfolioBacktest:
    def __init__(self, initial_capital: float = 1e8):
        """
        Initialize backtesting with initial capital
        """
        self.initial_capital = initial_capital
        self.start_date = '2019-07-31'
        self.end_date = '2024-10-31'
        
    def load_data(self, prices_file: str, portfolio_file: str, lq45_file: str) -> None:
        """
        Load and prepare all necessary data
        """
        # Load price data
        self.prices = pd.read_csv(prices_file, parse_dates=['date'])
        self.prices = self.prices[
            (self.prices['date'] >= self.start_date) & 
            (self.prices['date'] <= self.end_date)
        ]
        
        # Load portfolio data
        self.portfolio = pd.read_csv(portfolio_file)
        # Convert Month to datetime if it's not already
        self.portfolio['Month'] = pd.to_datetime(self.portfolio['Month'])
        
        # Load LQ45 data and clean numeric columns
        self.lq45 = pd.read_csv(lq45_file)
        self.lq45['date'] = pd.to_datetime(self.lq45['Tanggal'], format='%d/%m/%Y')
        
        # Clean numeric columns
        numeric_cols = ['Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah']
        for col in numeric_cols:
            self.lq45[col] = (self.lq45[col]
                             .str.replace('.', '')
                             .str.replace(',', '.')
                             .astype(float))
        
        self.lq45 = self.lq45[
            (self.lq45['date'] >= self.start_date) & 
            (self.lq45['date'] <= self.end_date)
        ]

    def calculate_monthly_returns(self) -> pd.DataFrame:
        """
        Calculate monthly returns for all stocks
        """
        # Convert date to period string format for consistency
        self.prices['month'] = self.prices['date'].dt.strftime('%Y-%m')
        
        # Flatten the multi-index result to avoid Series objects in the DataFrame
        monthly_returns = (self.prices.groupby(['ticker', 'month'])['adj_close']
                         .agg(['first', 'last'])
                         .reset_index())
        
        # Calculate returns
        monthly_returns['return'] = (
            (monthly_returns['last'] - monthly_returns['first']) / 
            monthly_returns['first'] * 100
        )
        
        return monthly_returns[['ticker', 'month', 'return']]

    def simulate_portfolio(self, lookback_days: List[int]) -> pd.DataFrame:
        """
        Simulate portfolio performance for different lookback periods with compounding returns
        """
        results = []
        
        # Convert Month to string format for consistency
        self.portfolio['Month'] = self.portfolio['Month'].dt.strftime('%Y-%m')
        # Sort portfolio by month to ensure chronological processing
        self.portfolio = self.portfolio.sort_values('Month')
        
        # Group portfolio by lookback days
        for days in lookback_days:
            portfolio_subset = self.portfolio[self.portfolio['Lookback_Days'] == days].copy()
            current_capital = self.initial_capital
            
            # Process each month chronologically
            for _, row in portfolio_subset.iterrows():
                month = row['Month']
                stocks = eval(row['stock'])
                weights = eval(row['weight'])
                
                # Calculate returns for current month
                month_results = []
                for stock, weight in zip(stocks, weights):
                    monthly_return = self.monthly_returns[
                        (self.monthly_returns['ticker'] == stock) & 
                        (self.monthly_returns['month'] == month)
                    ]['return'].values
                    
                    stock_return = monthly_return[0] if len(monthly_return) > 0 else 0
                    position_value = current_capital * weight
                    pnl = position_value * (stock_return / 100)
                    
                    month_results.append({
                        'period': month,
                        'datapoints': days,
                        'asset': stock,
                        'allocation': weight,
                        'allocation_value': position_value,
                        'return_percentage': stock_return,
                        'pnl': pnl,
                        'end_period_value': position_value + pnl
                    })
                
                # Update current capital for next month based on this month's results
                current_capital = sum(result['end_period_value'] for result in month_results)
                results.extend(month_results)
        
        return pd.DataFrame(results)

    def calculate_lq45_returns(self) -> pd.DataFrame:
        """
        Calculate LQ45 index returns with compounding
        """
        # Convert date to string format for consistency
        self.lq45['month'] = self.lq45['date'].dt.strftime('%Y-%m')
        
        # Flatten the multi-index result
        lq45_monthly = (self.lq45.groupby('month')['Terakhir']
                       .agg(['first', 'last'])
                       .reset_index())
        
        # Calculate returns
        lq45_monthly['return'] = (
            (lq45_monthly['last'] - lq45_monthly['first']) / 
            lq45_monthly['first'] * 100
        )
        
        # Calculate compounding returns for LQ45
        lq45_results = []
        current_capital = self.initial_capital
        
        for _, row in lq45_monthly.iterrows():
            pnl = current_capital * (row['return'] / 100)
            end_period_value = current_capital + pnl
            
            lq45_results.append({
                'period': row['month'],
                'datapoints': 0,  # LQ45 has no lookback period
                'asset': 'LQ45',
                'allocation': 1.0,
                'allocation_value': current_capital,
                'return_percentage': row['return'],
                'pnl': pnl,
                'end_period_value': end_period_value
            })
            
            # Update capital for next period
            current_capital = end_period_value
        
        return pd.DataFrame(lq45_results)

    def run_backtest(self) -> pd.DataFrame:
        """
        Run the complete backtest and return results
        """
        # Calculate returns
        self.monthly_returns = self.calculate_monthly_returns()
        lookback_days = [30, 60, 90, 180, 365]
        
        # Simulate portfolios
        portfolio_results = self.simulate_portfolio(lookback_days)
        
        # Calculate LQ45 returns
        lq45_results = self.calculate_lq45_returns()
        
        # Combine portfolio and LQ45 results
        final_results = pd.concat([portfolio_results, lq45_results], ignore_index=True)
        
        # Ensure all columns are the correct type before sorting
        final_results['datapoints'] = final_results['datapoints'].astype(int)
        final_results['period'] = final_results['period'].astype(str)
        final_results['asset'] = final_results['asset'].astype(str)
        
        # Sort results
        final_results = final_results.sort_values(
            ['period', 'datapoints', 'asset']
        ).reset_index(drop=True)
        
        # Round numeric columns
        numeric_cols = ['allocation', 'allocation_value', 'return_percentage', 'pnl', 'end_period_value']
        final_results[numeric_cols] = final_results[numeric_cols].round(2)
        
        return final_results[
            ['period', 'datapoints', 'asset', 'allocation', 'allocation_value', 
             'return_percentage', 'pnl', 'end_period_value']
        ]

if __name__ == "__main__":
    backtest = PortfolioBacktest(initial_capital=1e8)
    backtest.load_data(
        prices_file="adjusted_close_prices.csv",
        portfolio_file="optimized_portfolio_monthlyrebalance_5years.csv",
        lq45_file="data_historis_jakarta_stock_exchange_lq45.csv"
    )
    
    results = backtest.run_backtest()
    results.to_csv("backtest_results.csv", index=False)
