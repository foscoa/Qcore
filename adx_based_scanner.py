from ib_insync import *
import pandas as pd
import ta
import time

# Connect to IB Gateway or TWS
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=1)  # Paper trading

# Define file path
file_path = "./futures_contract_specs.csv"  # Update with your file location

# Read the CSV file
fut_specs = pd.read_csv(file_path).to_dict(orient='records')

# Create a contract for E-mini S&P 500 futures

fut_dict = [i for i in fut_specs if i['symbol'] == 'CT'][0]

# Convert date to string if needed
fut_dict['lastTradeDateOrContractMonth'] = str(fut_dict['lastTradeDateOrContractMonth'])

# Create a general contract
contract = Contract(**fut_dict)



# Helper function to get historical data
def get_data(contract):
    bars = ib.reqHistoricalData(
        contract=contract,  # The contract object
        endDateTime="",  # End time ("" = current time)
        durationStr="1 Y",  # Duration (e.g., "1 D" = 1 day)
        barSizeSetting="1 day",  # Granularity (e.g., "1 min", "5 mins")
        whatToShow="TRADES",  # Data type: "TRADES", "BID", etc.
        useRTH=1,  # Regular Trading Hours only
        formatDate=1,  # Date format: 1 = human-readable, 2 = UNIX
        keepUpToDate=False,  # Keep receiving live updates (False for static)
        chartOptions=[]
    )

    df = util.df(bars)
    return df if not df.empty else None


# Function to check ADX entry setup
def check_adx_signal(df, threshold=25):
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    df['di_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)
    df['di_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)

    latest = df.iloc[-1]

    if latest['adx'] > threshold:
        if latest['di_pos'] > latest['di_neg']:
            return 'LONG'
        elif latest['di_neg'] > latest['di_pos']:
            return 'SHORT'
    return None


# Main scan loop
results = []

for fut in fut_specs:
    df = get_data(fut)
    if df is not None and len(df) > 20:
        signal = check_adx_signal(df)
        if signal:
            results.append({
                'symbol': fut.localSymbol,
                'exchange': fut.exchange,
                'signal': signal
            })
        time.sleep(1.5)  # Respect IB rate limits

# Print results
print("Entry Opportunities Found:")
for r in results:
    print(f"{r['symbol']} ({r['exchange']}): {r['signal']}")

ib.disconnect()
