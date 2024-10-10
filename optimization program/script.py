import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum
import math
import gurobi

# Setting up the initial date limit
start_date = pd.Timestamp('2023-01-01')  # Start at Jan 2023 (will use Dec 2022 data)
end_date = pd.Timestamp('2023-12-31')  # End of Dec 2023

# Load the data
df = pd.read_csv('raw.csv', parse_dates=['date'])

# Sort the data by date and remove duplicates
df = df.sort_values(by='date').drop_duplicates(subset=['date', 'ticker'])

# Loop through each month between the start date and end date
current_date = start_date

# Adding 180 and 365 data points to the options
lookback_days_options = [30, 60, 90, 180, 365]

for lookback_days in lookback_days_options:
    results = []
    print(f"\n### Calculating with {lookback_days} data points ###\n")

    current_date = start_date
    while current_date <= end_date:
        # Filter the DataFrame for the `lookback_days` trading days before the current_date
        df_filtered = df[df['date'] < current_date].copy()

        # Sort the filtered data by date and get the last `lookback_days` trading days
        df_filtered = df_filtered.groupby('ticker').tail(lookback_days)  # Last `lookback_days` trading days for each ticker

        # Pivot the data to have dates as the index and tickers as columns
        df_pivot = df_filtered.pivot(index='date', columns='ticker', values='adj_close')

        # Check for missing data and drop columns or fill missing values as needed
        df_pivot = df_pivot.dropna(how='all', axis=1)  # Drop tickers with no data
        df_pivot = df_pivot.ffill().bfill()

        # Calculate the returns based on the previous `lookback_days`
        monthly_returns = df_pivot.pct_change().dropna()
        monthly_returns = monthly_returns.loc[:, (monthly_returns != 0).any(axis=0)]  # Remove columns with no returns

        # Calculate the covariance matrix
        cov_matrix = monthly_returns.cov()

        ##### Print Covariance Matrix #####
        print(f"\nCovariance matrix for {lookback_days} trading days before {current_date.strftime('%Y-%m-%d')}")
        print(cov_matrix)

        ##### MONTHLY METRICS #####
        # Group by ticker to calculate monthly returns and risk (standard deviation)
        monthly_returns_grouped = df_filtered.groupby('ticker').agg(
            open_price=('adj_close', 'first'),
            close_price=('adj_close', 'last')
        ).reset_index()

        # Calculate the monthly return percentage
        monthly_returns_grouped['monthly_return (%)'] = 100 * (monthly_returns_grouped['close_price'] - monthly_returns_grouped['open_price']) / monthly_returns_grouped['open_price']

        # Calculate the monthly risk (standard deviation of monthly returns)
        stddev = df_filtered.groupby('ticker')['adj_close'].std().reset_index()
        stddev.columns = ['ticker', 'stddev (%)']
        stddev['stddev (%)'] = stddev['stddev (%)'] / 100  # Normalize risk as percentage
        stddev = stddev[stddev['stddev (%)'] != 0]  # Drop rows with zero risk
        stddev = stddev.dropna()

        # Merge the monthly returns and risk into a single DataFrame
        result = pd.merge(monthly_returns_grouped[['ticker', 'monthly_return (%)']], stddev, on='ticker')

        # Load the industry info from info.csv
        industry_info = pd.read_csv('info.csv')

        # Merge the industry info with the result DataFrame
        final_result = pd.merge(result, industry_info, on='ticker', how='left')
        final_result = final_result.drop_duplicates(subset='ticker')

        ##### Print Monthly Metrics #####
        print(f"\nMonthly metrics for the {lookback_days} trading days before {current_date.strftime('%Y-%m-%d')}")
        print(final_result)

        # Extract necessary data for optimization
        stock_names = final_result['ticker'].tolist()
        stock_returns = final_result['monthly_return (%)'].tolist()
        cov_matrix_values = cov_matrix.values

        # Define the set I to include the entire ticker dataset
        I = stock_names

        # Gurobi Optimization:
        params = {
            'WLSACCESSID': gurobi.WLSACCESSID,
            'WLSSECRET': gurobi.WLSSECRET,
            'LICENSEID': gurobi.LICENSEID,
        }
        env = gp.Env(params=params)

        # Create the Gurobi model within the environment
        model = gp.Model(env=env)
        allocation_vars = model.addVars(stock_names, name="allocation", lb=0.0, ub=1.0)

        # Create binary variables for selecting stocks
        select_vars = model.addVars(stock_names, vtype=GRB.BINARY, name="select")

        # Objective function: Maximize total expected return over the set I
        model.setObjective(
            quicksum(allocation_vars[stock] * stock_returns[stock_names.index(stock)] for stock in I), GRB.MAXIMIZE
        )

        # Constraint 1: Portfolio risk (using the covariance matrix) <= 20%
        risk_expr = quicksum(
            cov_matrix_values[i][j] * allocation_vars[stock_i] * allocation_vars[stock_j]
            for i, stock_i in enumerate(I)
            for j, stock_j in enumerate(I)
        )
        model.addQConstr(risk_expr <= (0.2 ** 2), "RiskLimit")

        # Constraint 2: At least 10% of the portfolio must be invested in "Banks - Regional"
        model.addConstr(
            quicksum(allocation_vars[row['ticker']] for _, row in final_result.iterrows() if row['industry'] == 'Banks - Regional') >= 0.1,
            "MinBanksRegional"
        )

        # Constraint 3: The sum of all investments must equal the total investment (normalized to 1)
        model.addConstr(quicksum(allocation_vars[stock] for stock in I) == 1, "TotalInvestment")

        # Constraint 4: At least 3 stocks should be allocated
        for stock in I:
            model.addConstr(allocation_vars[stock] <= select_vars[stock], f"Link_{stock}")
        model.addConstr(quicksum(select_vars[stock] for stock in I) >= 3, "MinStocks")

        # Solve the model
        model.optimize()

        # Print solution
        if model.status == GRB.OPTIMAL:
            total_expected_return = 0  # Initialize total expected return
            portfolio_risk_expr = 0  # Initialize the risk expression for the portfolio
            stock_optimized = [stock for stock in stock_names if allocation_vars[stock].X > 1e-6]
            stock_weight = [round(allocation_vars[stock].X, 2) for stock in stock_names if
                            allocation_vars[stock].X > 1e-6]
            allocations = {stock: allocation_vars[stock].X for stock in stock_names if allocation_vars[stock].X > 1e-6}

            for stock in allocations:
                allocation_percentage = allocations[stock] * 100  # Convert to percentage
                expected_return_contribution = allocations[stock] * stock_returns[stock_names.index(stock)]
                total_expected_return += expected_return_contribution

            # Calculate the portfolio risk using the covariance matrix
            portfolio_risk_expr = sum(
                allocations[stock_i] * allocations[stock_j] * cov_matrix_values[stock_names.index(stock_i)][stock_names.index(stock_j)]
                for stock_i in allocations
                for stock_j in allocations
            )
            portfolio_risk = math.sqrt(portfolio_risk_expr) * 100  # Convert to percentage

            # Store the results
            results.append({
                'Month': current_date.strftime('%Y-%m'),
                'Lookback_Days': lookback_days,
                'total_expected_return': total_expected_return,
                'portfolio_risk': portfolio_risk,
                'stock': stock_optimized,
                'weight': stock_weight,
                'allocations': allocations

            })

        else:
            print("No feasible solution found.")

        # Move to the next month
        current_date += pd.DateOffset(months=1)

    # Convert results into a DataFrame and format the allocations for easier viewing
    results_df = pd.DataFrame(results)
    results_df['allocations'] = results_df['allocations'].apply(lambda x: ', '.join([f'{k}: {v:.2f}%' for k, v in x.items()]))
    results_df.to_csv(f'optimized_portfolio_monthlyrebalance_{lookback_days}datapoints.csv', index=False)


# Print the final results
print(results_df)