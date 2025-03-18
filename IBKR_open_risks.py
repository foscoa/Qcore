from ib_insync import *
import pandas as pd

from IBKR_open_orders import open_positions

# Connect to IBKR Gateway or TWS
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=1)  # Use 4002 for IB Gateway paper trading

# Fetch open positions
positions = ib.positions()
positions_df = pd.DataFrame([
    {
        'ConID': pos.contract.conId,
        'Symbol': pos.contract.symbol,
        'Local Symbol': pos.contract.localSymbol,
        'SecType': pos.contract.secType,
        'Exchange': pos.contract.exchange,
        'Currency': pos.contract.currency,
        'Multiplier': pos.contract.multiplier if hasattr(pos.contract, 'multiplier') else 1,
        'Position': pos.position,
        'Avg Cost': pos.avgCost
    } for pos in positions
])

# Request all open orders (manual + API)
ib.reqAllOpenOrders()
trades = ib.trades()  # Fetch updated order list with contracts
# Convert trades to DataFrame
orders_df = pd.DataFrame([
    {
        'Order ID': trade.order.orderId,
        'ConID': trade.contract.conId,
        'Symbol': trade.contract.symbol,
        'Local Symbol': trade.contract.localSymbol,
        'SecType': trade.contract.secType,
        'Exchange': trade.contract.exchange,
        'Currency': trade.contract.currency,
        'Multiplier': trade.contract.multiplier if hasattr(trade.contract, 'multiplier') else 1,
        'Order Type': trade.order.orderType,
        'Quantity': trade.order.totalQuantity,
        'Action': trade.order.action,  # BUY or SELL
        'Limit Price': trade.order.lmtPrice if hasattr(trade.order, 'lmtPrice') else None,
        'Stop Price': trade.order.auxPrice if hasattr(trade.order, 'auxPrice') else None,
        'Status': trade.orderStatus.status
    } for trade in trades if trade.order  # Ensure order exists
])

# Net Liquidation Value
account_summary = ib.accountSummary()
# Convert to DataFrame
# Convert the list of TagValues to a DataFrame
account_summary_df = pd.DataFrame([
    {'Tag': item.tag, 'Value': item.value, 'Currency': item.currency}
    for item in account_summary
])
NLV = float(account_summary_df[account_summary_df['Tag'] == 'NetLiquidation'].Value.values[0])

# Merge positions and orders on ConID, Symbol, SecType, Exchange, Currency, Multiplier
risk_df = positions_df.merge(orders_df, on=['ConID', 'Symbol', 'Local Symbol',
                                            'SecType', 'Exchange', 'Currency', 'Multiplier'],
                                        how='outer', suffixes=('_Position', '_Order'))

def positionsHistPrices(df, durationStr, barSizeSetting):
    # Define a dictionary to store the close prices
    close_prices_dict = {}

    for conid in df['ConID'].unique():# Define contract using ConID (e.g., AAPL: 265598)

        conid = int(conid)
        exchange = df.query('ConID == @conid and Exchange != ""').Exchange.unique()[0]
        contract = Contract(conId=conid, exchange=exchange)

        # Request historical data
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',        # '' means the latest available data
            durationStr=durationStr,     # Duration: 1 day (options: '1 W', '1 M', '1 Y', etc.)
            barSizeSetting=barSizeSetting,  # Bar size: 1 hour (options: '1 min', '5 min', etc.)
            whatToShow='MIDPOINT',  # Can be 'TRADES', 'BID', 'ASK', 'MIDPOINT'
            useRTH=True,           # Regular Trading Hours only
            formatDate=1
        )
        ib.sleep(1)  # Allow time to fetch market data

        # Convert the historical data to a pandas DataFrame
        ts_df = util.df(bars)

        # Extract only the date and close price
        ts_df = ts_df[['date', 'close']]
        ts_df.set_index('date', inplace=True)

        # Store the close prices in the dictionary with the symbol as the key
        close_prices_dict[df.query('ConID == @conid and Exchange != ""').Symbol.unique()[0]] = ts_df['close']

    # Combine all close price DataFrames into one DataFrame
    close_prices_df = pd.DataFrame(close_prices_dict)

    return  close_prices_df

def addBaseCCYfx(df, ccy):
    # Fetch FX conversion rates to base currency
    base_currency = ccy  # Set your base currency here
    fx_rates = {}
    fx_time = {}
    for currency in df['Currency'].unique():
        if currency != base_currency:
            fx_contract = Forex(f'{base_currency}{currency}')
            # Request historical data for EUR/USD
            historical_data = ib.reqHistoricalData(
                fx_contract,
                endDateTime='',
                durationStr='1 D',  # 1 Day of historical data
                barSizeSetting='5 mins',  # 5-minute bars
                whatToShow='MIDPOINT',  # MIDPOINT gives the average bid/ask price
                useRTH=False,  # Use regular trading hours
                formatDate=1  # Format the date as a string
            )
            ib.sleep(1)  # Allow time to fetch market data
            fx_rates[currency] = historical_data[-1].close
            fx_time[currency] = historical_data[-1].date
        else:
            fx_rates[currency] = 1

    # Apply FX conversion rates
    df['FX Rate to Base'] = df['Currency'].map(fx_rates)

    return df

risk_df = addBaseCCYfx(risk_df, 'EUR')

# contracts for Money market purposed
contracts_MM = [11625311, 74991935, 281534370, 568953593]
risk_df = risk_df.copy().query("ConID not in @contracts_MM")

def addLastPX(df):
    # Fetch FX conversion rates to base currency
    LastPX = {}
    LastPX_time = {}

    for conid in df['ConID'].unique():# Define contract using ConID (e.g., AAPL: 265598)

        conid = int(conid)
        exchange = df.query('ConID == @conid and Exchange != ""').Exchange.unique()[0]
        contract = Contract(conId=conid, exchange=exchange)

        # Request historical data
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',        # '' means the latest available data
            durationStr='1 D',     # Duration: 1 day (options: '1 W', '1 M', '1 Y', etc.)
            barSizeSetting='1 min',  # Bar size: 1 hour (options: '1 min', '5 min', etc.)
            whatToShow='MIDPOINT',  # Can be 'TRADES', 'BID', 'ASK', 'MIDPOINT'
            useRTH=False,           # Regular Trading Hours only
            formatDate=1
        )
        ib.sleep(1)  # Allow time to fetch market data
        LastPX[conid] = bars[-1].close
        LastPX_time[conid] = bars[-1].date

    # Apply FX conversion rates
    df['LastPX'] = df['ConID'].map(LastPX)

    return df

risk_df = addLastPX(risk_df)
ts = positionsHistPrices(df = risk_df, durationStr= '1 Y', barSizeSetting= '1 day')
ts_ret = ts.copy().pct_change().dropna()
corr = ts_ret.corr()

last_risk = pd.DataFrame(columns=['Status','Symbol', 'Local Symbol', 'Currency', 'Contracts',
                                  'Risk (EUR)', 'Risk NLV (bps)', 'Exposure (EUR)', 'Expos. NLV (%)', 'FX', 'multiplier',
                                  'Last Price', 'ConID'])

for conid in risk_df['ConID'].unique():

    sub_df = risk_df.copy().query('ConID == @conid')

    multiplier = int(sub_df.Multiplier.unique()[0])
    lastPX = float(sub_df.LastPX.unique()[0])
    fx = float(sub_df['FX Rate to Base'].unique()[0])

    if sub_df.Position.isna().sum() == len(sub_df):
        position_status = "working"
    else:
        position_status = "open"

    if position_status == "open":
        open_position = sub_df.Position.dropna().sum()
        exposure = open_position * multiplier * lastPX / fx

        if open_position > 0: # long
            stops = sub_df[(sub_df["Stop Price"] < lastPX) & (sub_df["Limit Price"] < lastPX)].copy()
        elif open_position < 0: # short
            stops = sub_df[(sub_df["Stop Price"] > lastPX) & (sub_df["Limit Price"] > lastPX)].copy()

        stops['dir'] = stops['Action'].map({'SELL': -1, 'BUY': 1})
        stops_exec_exp = stops.Quantity * multiplier * (stops['Stop Price'] + stops['Limit Price']) / fx
        risk = exposure - stops_exec_exp.sum()


    elif position_status == "working":
        exposure = 0
        open_position = 0
        sub_df['dir'] = sub_df['Action'].map({'SELL': -1, 'BUY': 1})
        risk = (sub_df.Quantity * multiplier * (sub_df['Stop Price'] + sub_df['Limit Price']) * sub_df.dir).sum() / fx

    new_row = pd.DataFrame(data={
        'Status': [position_status],
        'Symbol': [sub_df.Symbol.unique()[0]],  # Make sure it's a list
        'Local Symbol': [sub_df['Local Symbol'].unique()[0]],  # Make sure it's a list
        'Currency': [sub_df.Currency.unique()[0]],  # Make sure it's a list
        'Contracts': [open_position],  # Scalar wrapped in list
        'Risk (EUR)': [risk],  # Scalar wrapped in list
        'Risk NLV (bps)':[(risk/NLV) *10000],
        'Exposure (EUR)': [exposure],  # Scalar wrapped in list
        'Expos. NLV (%)': [exposure/NLV*100],
        'FX': [fx],  # Scalar wrapped in list
        'multiplier': [multiplier],  # Scalar wrapped in list
        'Last Price': [lastPX],  # Scalar wrapped in list
        'ConID': [conid]  # Scalar wrapped in list
    })

    last_risk = pd.concat([last_risk, new_row], ignore_index=True)

last_risk.to_csv("Q_Pareto_Transaction_History/Data/open_risks.csv")

# Disconnect from IBKR
ib.disconnect()
