import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import yfinance as yf
import plotly.graph_objects as go

# Define the ticker symbol and the time period
ticker = "AAPL"  # Example: Apple Inc.
start_date = "2023-01-01"
end_date = "2024-01-01"

# Download historical data from Yahoo Finance
data = yf.download(ticker, start=start_date, end=end_date)
data.columns = [i[0] for i in data.columns] #renaming columns

lookback = 5*20

"""
Detect a triangle pattern in stock price data.

Parameters:
    data (pd.DataFrame): Stock price data with 'High' and 'Low' columns.
    lookback (int): Number of data points to consider for the pattern.

Returns:
    bool: True if a triangle pattern is detected, False otherwise.
    dict: A dictionary containing support and resistance trendlines.
"""
# Extract the relevant data
highs = data['High'][-lookback:].reset_index(drop=True)
lows = data['Low'][-lookback:].reset_index(drop=True)
x = np.arange(len(highs))

# Fit trendlines
res_slope, res_intercept, _, _, _ = linregress(x, highs)
sup_slope, sup_intercept, _, _, _ = linregress(x, lows)

# Compute the trendlines
resistance = res_slope * x + res_intercept
support = sup_slope * x + sup_intercept





# Ensure the data has 'High' and 'Low' columns
if 'High' not in data.columns or 'Low' not in data.columns:
    raise ValueError("The input data must contain 'High' and 'Low' columns.")



# Create a candlestick chart using Plotly
fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    increasing_line_color='green',
    decreasing_line_color='red'
)])

# Customize the layout with a black background
fig.update_layout(
    title=f"Candlestick Chart for {ticker}",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=False,  # Hide the range slider
    template="plotly_dark",           # Dark theme
    plot_bgcolor="black",             # Plot background color
    paper_bgcolor="black",            # Overall figure background color
    font=dict(color="white")          # Font color to contrast with black
)

# Add dynamic support line (linear regression on Low prices)
fig.add_trace(go.Scatter(
    x=data.index[-lookback:],
    y=support + data.Low[-lookback:].min()-support[int(np.where(data.Low[-lookback:] == data.Low[-lookback:].min())[0])],
    mode="lines",
    line=dict(color="white", width=2, dash="dash"),
    name="Support (Trend)"
))

# Add dynamic resistance line (linear regression on High prices)
fig.add_trace(go.Scatter(
    x=data.index[-lookback:],
    y=resistance + data.High[-lookback:].max()-resistance[int(np.where(data.High[-lookback:] == data.High[-lookback:].max())[0])],
    mode="lines",
    line=dict(color="white", width=2, dash="dash"),
    name="Resistance (Trend)"
))



# Show the plot
fig.show()