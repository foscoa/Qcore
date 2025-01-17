from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import pandas as pd
import threading
import time


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
    app = IBApi()

    # Step 2: Connect to TWS or IB Gateway
    app.connect("127.0.0.1", 7496, clientId=1)

    # Step 3: Start the API in a separate thread
    api_thread = threading.Thread(target=run_loop, daemon=True)
    api_thread.start()

    time.sleep(1)  # Give the connection some time to establish

    # Step 4: Define the contract for the stock
    from ibapi.contract import Contract

    # Create a contract for E-mini S&P 500 futures
    contract = Contract()
    contract.symbol = "CT"  # Symbol for E-mini S&P 500 futures
    contract.secType = "FUT"  # Futures security type
    contract.exchange = "NYBOT"  # CME's GLOBEX exchange
    contract.currency = "USD"  # Currency in USD
    contract.lastTradeDateOrContractMonth = "202503"  # Expiry in January 2025 (yyyy-mm)
    contract.strike = 0.0  # Futures contracts do not have a strike price
    contract.multiplier = 50000  # Multiplier for E-mini S&P 500 futures (e.g., 50)

    # Step 5: Request historical data
    app.reqHistoricalData(
        reqId=1,                     # Unique ID for the request
        contract=contract,           # The contract object
        endDateTime="20250116 14:20:00 US/Eastern",              # End time ("" = current time)
        durationStr="1 D",           # Duration (e.g., "1 D" = 1 day)
        barSizeSetting="1 min",      # Granularity (e.g., "1 min", "5 mins")
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
