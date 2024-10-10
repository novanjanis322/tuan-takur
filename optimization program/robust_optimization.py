import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum, gurobi
import gurobi_license
cap = 100000000
invest_preference = (input("Enter your investing preference (weekly/monthly/quarterly/semester/yearly:"))
if invest_preference == "weekly":
    granularity = 7
elif invest_preference == "monthly":
    granularity = 30
elif invest_preference == "quarterly":
    granularity = 90
elif invest_preference == "semester":
    granularity = 180
elif invest_preference == "yearly":
    granularity = 365
else:
    raise f"Undefined granularity of {invest_preference}..."
invest_period = int(input("enter the number of periods you want to invest:"))
# Set up the initial date limit
start_date = pd.Timestamp('2022-12-31')

# Load the full dataset once
df = pd.read_csv('raw.csv', parse_dates=['date'])

# Loop through each month between the start and end date with different lookback data points (30, 60, 90)
lookback_days = granularity
results = []
# print(f"\n### Calculating with {lookback_days} data points for {}###\n")
stock_optimized = []
stock_weight = []
print(f"\n### Calculating with {lookback_days} data points for {invest_period} {invest_preference} period ###\n")
current_date = start_date
for i in range(invest_period+1):
    # Set the lower bound based on the lookback period
    date_limit = current_date
    date_limit = date_limit + pd.DateOffset(days=1)
    df_filtered = df[df['date'] < date_limit].sort_values(by=['date'], ascending=False)
    df_filtered = df_filtered.groupby('ticker').head(lookback_days).sort_values(by=['ticker', 'date'])

    # Remove rows where the adjusted close price hasn't changed
    df_filtered['price_diff'] = df_filtered.groupby('ticker')['adj_close'].diff()
    df_filtered = df_filtered[df_filtered['price_diff'] != 0]
    df_filtered.drop(columns=['price_diff'], inplace=True)

    checkdate = current_date + pd.DateOffset(days=1)
    if invest_preference == "weekly":
        lastdate = checkdate + pd.DateOffset(weeks=1)
    elif invest_preference == "monthly":
        lastdate = checkdate + pd.DateOffset(months=1)
    elif invest_preference == "quarterly":
        lastdate = checkdate + pd.DateOffset(months=3)
    elif invest_preference == "semester":
        lastdate = checkdate + pd.DateOffset(months=6)
    elif invest_preference == "yearly":
        lastdate = checkdate + pd.DateOffset(years=1)
    tickers_on_last_date = df[(df['date'] >= checkdate) & (df['date'] <= (lastdate + pd.offsets.MonthEnd(0)))]['ticker'].unique()
    df_filtered = df_filtered[df_filtered['ticker'].isin(tickers_on_last_date)]
    print(f'make sure tickers are in between {checkdate.strftime("%Y-%m-%d")} and {(lastdate).strftime("%Y-%m-%d")}')

    df_pivot = df_filtered.pivot(index='date', columns='ticker', values='adj_close')
    # Forward and backward fill missing values
    df_pivot = df_pivot.dropna(how='all', axis=1)
    # Pivot the data to have dates as the index and tickers as columns
    df_pivot = df_pivot.ffill().bfill()

    # Calculate the monthly returns (based on the lookback data)
    monthly_returns = df_pivot.pct_change().dropna()
    monthly_returns = monthly_returns.loc[:, (monthly_returns != 0).any(axis=0)]

    ##### MONTHLY METRICS #######
    # Group by ticker and calculate monthly open and close prices
    monthly_returns_summary = df_filtered.groupby('ticker').agg(
        open_price=('adj_close', 'first'),
        close_price=('adj_close', 'last')
    ).reset_index()

    # print(monthly_returns_summary[monthly_returns_summary['ticker'] == 'BBCA.JK'])
    # Calculate the monthly return percentage
    monthly_returns_summary['monthly_return (%)'] = 100 * (
                monthly_returns_summary['close_price'] - monthly_returns_summary['open_price']) / monthly_returns_summary['open_price']

    # Calculate the monthly risk (standard deviation of adjusted close prices)
    stddev = df_filtered.groupby('ticker')['adj_close'].std().reset_index()
    stddev.columns = ['ticker', 'stddev (%)']
    stddev['stddev (%)'] = stddev['stddev (%)'] / 100
    stddev = stddev[stddev['stddev (%)'] != 0]  # Drop rows where risk is zero
    stddev = stddev.dropna(axis=0)

    # Merge the returns and standard deviation into one DataFrame
    result = pd.merge(monthly_returns_summary[['ticker', 'monthly_return (%)']], stddev, on='ticker')
    # Load industry info
    industry_info = pd.read_csv('info.csv')
    final_result = pd.merge(result, industry_info, on='ticker', how='left')
    final_result = final_result.drop_duplicates(subset='ticker')


    invest_change = 0
    if len(stock_optimized)>0:
        print('''############Portfolio Calculation############''')
        for j in range (len(stock_optimized)):
            weighted_invest = cap * stock_weight[j]
            print(final_result[final_result['ticker'] == stock_optimized[j]]['monthly_return (%)'].values[0] / 100)
            invest_change += weighted_invest * (1 + final_result[final_result['ticker'] == stock_optimized[j]]['monthly_return (%)'].values[0] / 100)
        cap = invest_change
        print(f"Capital after investing in {date_limit.strftime('%Y-%m-%d')}: {cap}\n\n")
        if i == invest_period:
            print(f"final capital after {i} {invest_preference} period: {cap}")
            break
        # Check if there are enough tickers to calculate the covariance matrix
    if len(monthly_returns.columns) > 1:
        cov_matrix = monthly_returns.cov()
        print(f"Covariance matrix for {date_limit.strftime('%Y-%m-%d')} (Lookback: {lookback_days} days)")
        print(cov_matrix)
    else:
        print(
            f"Not enough data to calculate covariance matrix for {date_limit.strftime('%Y-%m-%d')} (Lookback: {lookback_days} days)")
    # print monthly metrics
    print(f"Monthly metrics for {date_limit.strftime('%Y-%m-%d')} (Lookback: {lookback_days} days)")
    print(final_result)

    # Extract necessary data for optimization
    stock_names = final_result['ticker'].tolist()
    stock_returns = final_result['monthly_return (%)'].tolist()
    cov_matrix = cov_matrix.values if 'cov_matrix' in locals() else None

    # Only run optimization if covariance matrix exists
    if cov_matrix is not None and len(cov_matrix) > 0:
        # Create Gurobi model for portfolio optimization
        env = gp.Env(params={
            'WLSACCESSID': gurobi_license.WLSACCESSID,
            'WLSSECRET': gurobi_license.WLSSECRET,
            'LICENSEID': gurobi_license.LICENSEID
        })
        model = gp.Model(env=env)
        selection_vars = model.addVars(stock_names, vtype=GRB.BINARY, name="selection")
        allocation_vars = model.addVars(stock_names, lb=0, ub=1, name="allocation")
        model.setObjective(
            quicksum(allocation_vars[stock] * stock_returns[stock_names.index(stock)] for stock in stock_names),
            GRB.MAXIMIZE)

        risk_expr = quicksum(
            cov_matrix[i][j] * allocation_vars[stock_i] * allocation_vars[stock_j]
            for i, stock_i in enumerate(stock_names)
            for j, stock_j in enumerate(stock_names)
        )
        model.addQConstr(risk_expr <= (0.2 ** 2), "RiskLimit")
        model.addConstr(quicksum(allocation_vars[stock] for stock in stock_names) == 1, "TotalInvestment")
        model.addConstr(quicksum(allocation_vars[row['ticker']] for _, row in final_result.iterrows() if
                                 row['industry'] == 'Banks - Regional') >= 0.1, "MinBanksRegional")
        model.addConstr(quicksum(selection_vars[stock] for stock in stock_names) <= 5, "portfoliolimit")

        # Solve the model
        model.optimize()

        # Store results
        if model.status == GRB.OPTIMAL:
            stock_optimized = [stock for stock in stock_names if allocation_vars[stock].X > 1e-6]
            stock_weight = [round(allocation_vars[stock].X,2) for stock in stock_names if allocation_vars[stock].X > 1e-6]
            portfolio_risk = (sum(
                cov_matrix[i][j] * allocation_vars[stock_i].X * allocation_vars[stock_j].X
                for i, stock_i in enumerate(stock_names)
                for j, stock_j in enumerate(stock_names)
            )) ** 0.5 * 100  # Convert to percentage

            # allocations = {stock: f"{allocation_vars[stock].X:.2f}" for stock in stock_names if allocation_vars[stock].X > 1e-6}
            results.append({
                'Period': checkdate.strftime('%Y-%m-%d'),
                'Lookback_Days': lookback_days,
                'portfolio_risk': portfolio_risk,
                'stock' : stock_optimized,
                'weight' : stock_weight,
            })
            print(f"Optimized portfolio for {date_limit.strftime('%Y-%m-%d')} (Lookback: {lookback_days} days)")
            for i in range(len(stock_optimized)):
                print(f"{stock_optimized[i]}: {stock_weight[i] * 100:.2f}%")
            print(f"Portfolio risk: {portfolio_risk:.2f}%")
            print('\n')
    # Move to the next month
    if invest_preference == "weekly":
        current_date = current_date + pd.DateOffset(weeks=1)
    elif invest_preference == "monthly":
        current_date = current_date + pd.DateOffset(months=1)
        current_date = current_date.replace(day=1) + pd.offsets.MonthEnd(0)
    elif invest_preference == "quarterly":
        current_date = current_date + pd.DateOffset(months=3)
        current_date = current_date.replace(day=1) + pd.offsets.MonthEnd(0)
    elif invest_preference == "semester":
        current_date = current_date + pd.DateOffset(months=6)
        current_date = current_date.replace(day=1) + pd.offsets.MonthEnd(0)
    elif invest_preference == "yearly":
        current_date = current_date + pd.DateOffset(years=1)
        current_date = current_date.replace(day=1) + pd.offsets.MonthEnd(0)
    else:
        raise f"Undefined granularity of {invest_preference}..."

# Convert results into a DataFrame and save
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('results_with_various_lookback.csv', index=False)
capital = 100000000
