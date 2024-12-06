
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# Define the list of stock tickers
# List of tickers
tickers = [
    "AJG", "CDNS", "CASY", "CTAS", "CSU.TO",
    "CPRT", "FTNT", "HEI", "KNSL", "MEDP",
    "MELI", "ROL", "WM", "WSO", "WKL.AS"
]

# Define the date range
start_date = "2020-01-01"
end_date = "2024-12-06"

# Download the data
data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
data.dropna(axis=0, inplace=True)

has_nan = data.isnull().values.any()
print(f"Does the DataFrame contain NaN values? {has_nan}")

# Calculate daily returns
daily_returns = data.pct_change().dropna()

# Calculate annualized return and annualized standard deviation
trading_days = 252  # Approximate number of trading days in a year
annualized_return = daily_returns.mean() * trading_days
annualized_volatility = daily_returns.std() * np.sqrt(trading_days)


# Create a DataFrame for the metrics
metrics = pd.DataFrame({
    'Ticker': tickers,
    'Annualized Return': annualized_return.values,
    'Annualized Volatility': annualized_volatility.values
})


def generate_plot_risk_return(metrics):
    # Plot the risk-return scatter plot with Plotly
    fig = px.scatter(
        metrics,
        x='Annualized Volatility',
        y='Annualized Return',
        text='Ticker',
        title='Risk-Return Stocks Arvy QCore Mini-TCI',
        labels={'Annualized Volatility': 'Annualized Volatility (Risk)', 'Annualized Return': 'Annualized Return'},
        template='plotly',
        size_max=15
    )

    # Improve layout
    fig.update_traces(textposition='top center', marker=dict(size=10, color='blue'))
    fig.update_layout(
        xaxis=dict(title='Annualized Volatility (Risk)',tickformat=".1%", showgrid=True),
        yaxis=dict(title='Annualized Return',tickformat=".1%" ,showgrid=True),
        title=dict(x=0.5),
        showlegend=False
    )

    return fig

fig = generate_plot_risk_return(metrics=metrics)
fig.show()

#######

def cum_returns_EW_daily_rebal(tickers, portfolio_daily_returns):
    # Assume an equally weighted portfolio
    weights = np.full(len(tickers), 1 / len(tickers))  # Equal weights

    # Calculate portfolio daily returns
    portfolio_daily_returns = daily_returns.dot(weights)

    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_daily_returns).cumprod()

    return cumulative_returns

def cum_returns_EW_no_rebal(tickers, data):

    # 2. Initial weights (equal weighting)
    num_stocks = len(tickers)
    initial_weights = np.array([1 / num_stocks] * num_stocks)

    # 3. Initial portfolio value
    initial_portfolio_value = 100000  # Starting portfolio value in currency units
    initial_investment = initial_portfolio_value * initial_weights
    initial_investment = np.floor(initial_investment[0] / data.iloc[0, :])
    initial_capital = initial_investment.dot(data.iloc[0,:])

    # 4. Adjust portfolio value without rebalancing
    # (Contributions are based on initial investments and individual stock performance)
    portfolio_values = data.dot(initial_investment)

    # 5. Calculate total portfolio value (sum contributions)
    cumulative_returns = portfolio_values/initial_capital
    return cumulative_returns

cumulative_returns_daily_rebal = cum_returns_EW_daily_rebal(tickers, daily_returns)
cumulative_returns_no_rebal = cum_returns_EW_no_rebal(tickers,data)

# Create a DataFrame for plotting
performance = pd.DataFrame({
    'Date': data.index,
    'Daily Rebalancing': cumulative_returns_daily_rebal,
    'No Rebalancing': cumulative_returns_no_rebal
})

# Melt the DataFrame for easier plotting with Plotly
performance_melted = performance.melt(id_vars='Date', var_name='Strategy', value_name='Portfolio Value')

# Plot portfolio performance
fig = px.line(
    performance_melted,
    x='Date',
    y='Portfolio Value',
    color='Strategy',
    title='Performance of an Equally Weighted Portfolio',
    labels={'Portfolio Value': 'Portfolio Value (Cumulative Return)', 'Date': 'Date'},
    template='plotly'
)

# Improve layout
fig.update_layout(
    yaxis=dict(tickformat=".1%", title="Cumulative Return"),
    xaxis=dict(title="Date"),
    title=dict(x=0.5)
)

# Show the plot
fig.show()

#######

# Define the factor ETFs and their labels
factor_etfs = {
    'SPY': 'Market Risk Premium',
    'IWM': 'Size (SMB)',
    'VTV': 'Value (HML)',
    'QUAL': 'Profitability (RMW)',
    'SPHD': 'Investment (CMA)',
    'MTUM': 'Momentum'
}

factor_data = yf.download(list(factor_etfs.keys()), start=start_date, end=end_date, progress=False)['Adj Close']

factor_daily_returns = factor_data.pct_change().dropna()

# benchmarks
bcmk_etfs = {
    'SPY': 'SPY',
    'RSP': 'RSP',
    'IWM': 'IWM',
    'QQQ': 'QQQ'
}

bcmk_data = yf.download(list(bcmk_etfs.keys()), start=start_date, end=end_date, progress=False)['Adj Close']

bcmk_daily_returns = bcmk_data.pct_change().dropna()


# Combine portfolio returns and factor returns into a single DataFrame

combined_returns = pd.DataFrame({
    'Equally Weighted Portfolio': cumulative_returns_no_rebal/cumulative_returns_no_rebal.shift(1)-1
}).join(factor_daily_returns.rename(columns=factor_etfs), how='inner')

# Perform correlation analysis
correlation_matrix = combined_returns.corr()

# Display the correlation matrix
print(correlation_matrix)

########

# Calculate cumulative returns for the ETFs
factor_cumulative_returns = (1 + factor_daily_returns).cumprod()

# Combine portfolio cumulative returns with ETFs
combined_cumulative_returns = pd.DataFrame({
    'Arvy QCore Mini-TCI': cumulative_returns_no_rebal
}).join(factor_cumulative_returns.rename(columns=factor_etfs), how='inner')

# Melt the DataFrame for Plotly compatibility
cumulative_returns_melted = combined_cumulative_returns.reset_index().melt(
    id_vars='Date',
    var_name='Asset',
    value_name='Cumulative Return'
)

# Plot the cumulative returns
fig = px.line(
    cumulative_returns_melted,
    x='Date',
    y='Cumulative Return',
    color='Asset',
    title='Performance of Equally Weighted Portfolio (Arvy QCore Mini-TCI) vs Factor ETFs',
    labels={'Cumulative Return': 'Cumulative Return (Growth of $1)', 'Date': 'Date'},
    template='plotly'
)

# Improve layout
fig.update_layout(
    yaxis=dict(tickformat=".1%", title="Cumulative Return"),
    xaxis=dict(title="Date"),
    title=dict(x=0.5)
)

# Show the plot
fig.show()

#####

# Calculate drawdowns
rolling_max = cumulative_returns_no_rebal.cummax()
drawdowns = (cumulative_returns_no_rebal - rolling_max) / rolling_max
# Find the drawdown periods: a drawdown ends when the portfolio value exceeds the previous peak
drawdown_periods = []

# Iterate through the drawdowns to find distinct peak-to-trough periods
in_drawdown = False
drawdown_start = None
for i in range(1, len(drawdowns)):
    if drawdowns[i] < 0 and not in_drawdown:
        # A new drawdown starts
        drawdown_start = i
        in_drawdown = True
    elif drawdowns[i] >= 0 and in_drawdown:
        # Drawdown ends when the value exceeds the previous peak
        drawdown_end = i
        drawdown_periods.append({
            'Start': cumulative_returns_no_rebal.index[drawdown_start],
            'Trough': cumulative_returns_no_rebal.index[drawdown_start + np.argmin(drawdowns[drawdown_start:drawdown_end])],
            'End': cumulative_returns_no_rebal.index[drawdown_end],
            'Drawdown': drawdowns[drawdown_start + np.argmin(drawdowns[drawdown_start:drawdown_end])]
        })
        in_drawdown = False

# Convert the drawdown periods into a DataFrame
drawdown_df = pd.DataFrame(drawdown_periods)

# Sort by the largest drawdown and get the top 5 worst
worst_drawdowns = drawdown_df.sort_values(by='Drawdown').head(5)

# Print the results
print("Worst 5 Drawdowns of the Equally Weighted Strategy (Arvy QCore Mini-TCI):")
print(worst_drawdowns[['Start', 'Trough', 'End', 'Drawdown']])

####

# Define the date range (last 90 days)
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
start_date = (pd.Timestamp.today() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")

# Download the data (only 'Volume' column)
volume_data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Volume']


# Calculate the average daily volume for the past 90 days in millions
average_daily_volume_millions = volume_data.mean() / 1_000_000

# Sort by decreasing value
average_daily_volume_millions_sorted = average_daily_volume_millions.sort_values(ascending=False)

# Print the results
print("Average Daily Volume (Past 90 Days) in Millions (Sorted):")
print(average_daily_volume_millions_sorted)


#### correlation analysis
# Calculate cumulative returns for the ETFs
bcmk_cumulative_returns = (1 + bcmk_daily_returns).cumprod()

# Combine portfolio cumulative returns with ETFs
combined_cumulative_returns = pd.DataFrame({
    'Portfolio': cumulative_returns_no_rebal
}).join(bcmk_cumulative_returns.rename(columns=bcmk_etfs), how='inner')

# Melt the DataFrame for Plotly compatibility
cumulative_returns_melted = combined_cumulative_returns.reset_index().melt(
    id_vars='Date',
    var_name='Asset',
    value_name='Cumulative Return'
)

# Plot the cumulative returns
fig = px.line(
    cumulative_returns_melted,
    x='Date',
    y='Cumulative Return',
    color='Asset',
    title='Performance of Arvy QCore Mini-TCI vs benchmark ETFs',
    labels={'Cumulative Return': 'Cumulative Return (Growth of $1)', 'Date': 'Date'},
    template='plotly'
)

# Improve layout
fig.update_layout(
    yaxis=dict(tickformat=".1%", title="Cumulative Return"),
    xaxis=dict(title="Date"),
    title=dict(x=0.5)
)

# Show the plot
fig.show()

# rolling beta
# Define rolling window (e.g., 30 days)
window_size = 60

combined_returns = combined_cumulative_returns/combined_cumulative_returns.shift(1) -1
combined_returns.dropna(inplace=True)

# Annualized return: (1 + daily return) ^ 252 - 1
annualized_returns = combined_cumulative_returns.iloc[-1,:] ** (252 / len(combined_cumulative_returns)) - 1
annualized_returns = pd.DataFrame(annualized_returns).transpose()

# Annualized volatility: Standard deviation of daily returns * sqrt(252)
annualized_volatility = combined_returns.std() * np.sqrt(252)

# Print results for each asset
for asset in combined_returns.columns:
    print(f"Asset: {asset}")
    print(f"  Annualized Return: {float(annualized_returns[asset]):.4f}")
    print(f"  Annualized Volatility: {annualized_volatility[asset]:.4f}")
    print("-" * 50)


# Initialize a DataFrame to store rolling beta values
rolling_betas = pd.DataFrame({'Date': combined_returns.index})

for bcmk in bcmk_etfs.keys():
    # Calculate rolling covariance and variance
    rolling_cov = combined_returns['Portfolio'].rolling(window=window_size).cov(combined_returns[bcmk])
    rolling_var = combined_returns[bcmk].rolling(window=window_size).var()

    # Calculate rolling beta
    rolling_betas[bcmk] = (rolling_cov / rolling_var).values

rolling_betas.dropna(inplace=True)

# Melt the rolling betas DataFrame for plotting
rolling_betas_long = pd.melt(
    rolling_betas,
    id_vars=['Date'],
    var_name='Benchmark',
    value_name='Rolling Beta'
)

# Plot rolling betas using Plotly
fig = px.line(
    rolling_betas_long,
    x='Date',
    y='Rolling Beta',
    color='Benchmark',
    title=f'Rolling Beta (Window: {window_size} Days)',
    labels={'Rolling Beta': 'Beta', 'Date': 'Date'},
    template='plotly_white'
)

# Add a horizontal reference line for Beta = 1
fig.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="Beta = 1", annotation_position="bottom left")

# Show the plot
fig.show()

# beta over all period
for bcmk in bcmk_etfs.keys():
    cov = combined_returns['Portfolio'].cov(combined_returns[bcmk])
    var = combined_returns[bcmk].var()
    beta=cov/var
    print("Beta " + str(bcmk) + ": " + str(beta))