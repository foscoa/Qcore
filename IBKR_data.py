from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import pandas as pd
import threading
import time
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go


class IBApi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []  # Store historical data here

    def historicalData(self, reqId, bar):
        """Called for each historical bar received."""
        print(f"Time: {bar.date}, Open: {bar.open}, High: {bar.high}, Low: {bar.low}, Close: {bar.close}")
        self.data.append(bar)

    def historicalDataEnd(self, reqId, start, end):
        """Called when historical data download is complete."""
        print(f"Historical data download complete. Start: {start}, End: {end}")
        self.disconnect()


def run_loop():
    """Keeps the API running in a separate thread."""
    app.run()


if __name__ == "__main__":
    # Step 1: Create an instance of the IBApi class

    # Define file path
    file_path = ".\\futures_contract_specs.csv"  # Update with your file location

    # Read the CSV file
    fut_specs = pd.read_csv(file_path).to_dict(orient='records')

    # Create a contract for E-mini S&P 500 futures

    fut = [i for i in fut_specs if i['symbol']=='CT'][0]

    app = IBApi()

    # Step 2: Connect to TWS or IB Gateway
    app.connect("127.0.0.1", 7496, clientId=1)

    # Step 3: Start the API in a separate thread
    api_thread = threading.Thread(target=run_loop, daemon=True)
    api_thread.start()

    time.sleep(1)  # Give the connection some time to establish

    contract = Contract()
    contract.symbol = fut['symbol']  # Symbol for E-mini S&P 500 futures
    contract.secType = fut['secType']  # Futures security type
    contract.exchange = fut['exchange']  # CME's GLOBEX exchange
    contract.currency = fut['currency']  # Currency in USD
    contract.lastTradeDateOrContractMonth = fut['lastTradeDateOrContractMonth']  # Expiry in January 2025 (yyyy-mm)
    contract.strike = fut['strike']  # Futures contracts do not have a strike price
    contract.multiplier = fut['multiplier']  # Multiplier for E-mini S&P 500 futures (e.g., 50)

    # Step 5: Request historical data
    app.reqHistoricalData(
        reqId=1,                     # Unique ID for the request
        contract=contract,           # The contract object
        endDateTime="",# End time ("" = current time)
        durationStr="180 D",           # Duration (e.g., "1 D" = 1 day)
        barSizeSetting="1 day",      # Granularity (e.g., "1 min", "5 mins")
        whatToShow="TRADES",         # Data type: "TRADES", "BID", etc.
        useRTH=1,                    # Regular Trading Hours only
        formatDate=1,                # Date format: 1 = human-readable, 2 = UNIX
        keepUpToDate=False,          # Keep receiving live updates (False for static)
        chartOptions=[]
    )

    # Step 6: Wait for the data to arrive
    time.sleep(5)

    # Step 7: Convert data to DataFrame (optional)
    df = pd.DataFrame([{
        'date': bar.date,
        'open': bar.open,
        'high': bar.high,
        'low': bar.low,
        'close': bar.close,
        'volume': bar.volume
    } for bar in app.data])
    print(df)

    # Disconnect and clean up
    app.disconnect()

    lookback = 5 * 25
    df.index = pd.to_datetime(df.date, format='%Y%m%d')
    data = df

    """
    Detect a triangle pattern in stock price data.

    Parameters:
        data (pd.DataFrame): Stock price data with 'high' and 'low' columns.
        lookback (int): Number of data points to consider for the pattern.

    Returns:
        bool: True if a triangle pattern is detected, False otherwise.
        dict: A dictionary containing support and resistance trendlines.
    """
    # Extract the relevant data
    highs = data['high'][-lookback:].reset_index(drop=True)
    lows = data['low'][-lookback:].reset_index(drop=True)
    x = np.arange(len(highs))

    # Fit trendlines
    res_slope, res_intercept, _, _, _ = linregress(x, highs)
    sup_slope, sup_intercept, _, _, _ = linregress(x, lows)

    # Compute the trendlines
    resistance = res_slope * x + res_intercept
    support = sup_slope * x + sup_intercept

    # Ensure the data has 'high' and 'low' columns
    if 'high' not in data.columns or 'low' not in data.columns:
        raise ValueError("The input data must contain 'high' and 'low' columns.")

    # Create a candlestick chart using Plotly
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    # Customize the layout with a black background
    fig.update_layout(
        title=f"Candlestick Chart for {contract.symbol}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,  # Hide the range slider
        template="plotly_dark",  # Dark theme
        plot_bgcolor="black",  # Plot background color
        paper_bgcolor="black",  # Overall figure background color
        font=dict(color="white")  # Font color to contrast with black
    )

    # Add dynamic support line (linear regression on low prices)
    # fig.add_trace(go.Scatter(
    #     x=data.index[-lookback:],
    #     y=support + data.low[-lookback:].min() - support[
    #         int(np.where(data.low[-lookback:] == data.low[-lookback:].min())[0])],
    #     mode="lines",
    #     line=dict(color="white", width=2, dash="dash"),
    #     name="Support (Trend)"
    # ))

    c = 0
    for i in range(1):

        val = np.sort(data.low[-lookback:])[i]

        c += val - support[int(np.where(data.low[-lookback:] == val)[0][0])]

    s = support + c/(i+1)

    # Add a horizontal line at y=150 (example)
    fig.add_shape(
        type="line",
        x0=data.index[-lookback:][0],  # Start at the first date
        x1=data.index[-lookback:][-1],  # End at the last date
        y0=s[0],  # Y position of the line
        y1=s[-1],  # Y position of the line (same for horizontal line)
        line=dict(color="white", width=1),
    )

    m = resistance + data.high[-lookback:].max() - resistance[
            int(np.where(data.high[-lookback:] == data.high[-lookback:].max())[0])]

    # Add dynamic resistance line (linear regression on high prices)
    # Add a horizontal line at y=150 (example)
    fig.add_shape(
        type="line",
        x0=data.index[-lookback:][0],  # Start at the first date
        x1=data.index[-lookback:][-1],  # End at the last date
        y0=m[0],  # Y position of the line
        y1=m[-1],  # Y position of the line (same for horizontal line)
        line=dict(color="white", width=1),
    )

    # Show the plot
    fig.show()
