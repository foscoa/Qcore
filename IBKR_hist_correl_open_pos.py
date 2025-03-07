from ib_insync import *
import pandas as pd
import time

# Connect to IB Gateway or TWS
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=1)

# Fetch open positions
positions = ib.positions()

# Extract unique contracts
contracts = {pos.contract.symbol: pos.contract for pos in positions}

# Ensure exchange is set for futures and other contracts
for contract in contracts.values():
    if not contract.exchange:
        contract.exchange = "SMART"

# Request historical data
hist_data = {}
for symbol, contract in contracts.items():
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='1 Y',  # One year of data
        barSizeSetting='1 day',
        whatToShow='ADJUSTED_LAST',
        useRTH=True,
        formatDate=1
    )

    if bars:
        df = pd.DataFrame(bars)
        df.set_index("date", inplace=True)
        hist_data[symbol] = df['close']

    time.sleep(1)  # Prevent rate limits

# Disconnect from IBKR
ib.disconnect()

# Create a DataFrame with closing prices
price_df = pd.DataFrame(hist_data)

# Compute the correlation matrix
correlation_matrix = price_df.pct_change().corr()

# Display the correlation matrix
print("Historical Correlation Matrix of Open Positions:")
print(correlation_matrix)
