import asyncio
import pandas as pd
from ib_async import IB, Contract
from utils.utils import plot_candles
import plotly.io as pio

# Replace with your actual IBKR contract ID
CONTRACT_ID = 731454279  # Example: SPY

# Force Plotly to open plots in your web browser
pio.renderers.default = "browser"

# Connect to IBKR Gateway or TWS
ib = IB()
ib.connect('localhost', 7496, clientId=1)  # Use 4002 for IB Gateway paper trading

conid = int(CONTRACT_ID)
contract = ib.qualifyContracts(Contract(conId=conid))[0]
durationStr= '1 Y'
barSizeSetting= '1 day'


# Request historical data
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',  # '' means the latest available data
    durationStr=durationStr,  # Duration: 1 day (options: '1 W', '1 M', '1 Y', etc.)
    barSizeSetting=barSizeSetting,  # Bar size: 1 hour (options: '1 min', '5 min', etc.)
    whatToShow='TRADES',  # Can be 'TRADES', 'BID', 'ASK', 'MIDPOINT'
    useRTH=True,  # Regular Trading Hours only
    formatDate=1
)
ib.sleep(1)  # Allow time to fetch market data

# Convert the historical data to a pandas DataFrame
ts_df = pd.DataFrame(bars)

# Extract only the date and close price
ts_df['date'] = pd.to_datetime(ts_df['date'],utc=True)
ts_df.set_index('date', inplace=True)

ts_close = ts_df[['close']]

# plot figure

entry = 135
exit = 132

fig = plot_candles(df=ts_df,title="a", interval="1D", my_entry=entry, my_exit=exit)
fig.show()

ib.disconnect()