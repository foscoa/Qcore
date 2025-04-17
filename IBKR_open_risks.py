from ib_insync import *
import pandas as pd
import numpy as np

from IBKR_open_orders import open_positions

# Connect to IBKR Gateway or TWS
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=1)  # Use 4002 for IB Gateway paper trading

def get_realized_PnL():
    # Define the file path
    file_path = "Q_Pareto_Transaction_History/Data/U15721173_TradeHistory_04172025.csv"
    # Read the CSV file
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.replace("/", "_", regex=False)

    master_df = df.copy().query(
        'LevelOfDetail == "EXECUTION" & (Open_CloseIndicator == "C" or Open_CloseIndicator == "O")')

    # clean DateTime
    # Remove single quotes and replace ";" with a space
    clean_date = master_df.DateTime.str.replace(";", " ")

    # Convert to datetime in CET timezone
    master_df['DateTime_clean'] = pd.to_datetime(clean_date, format="%Y%m%d %H%M%S").dt.tz_localize(
        'America/New_York').dt.tz_convert('Europe/Berlin').dt.tz_localize(None)

    # Sort by Time
    master_df = master_df.copy().sort_values(by='DateTime_clean', ascending=False)

    # Adding FifoPnlRealzed in Base Currency
    master_df['FifoPnlRealizedToBase'] = master_df.FifoPnlRealized * master_df.FXRateToBase

    # Adding NotionaltoBase
    master_df[
        'NotionaltoBase'] = master_df.Quantity.abs() * master_df.Multiplier * master_df.FXRateToBase * master_df.TradePrice

    # Direction
    master_df['Position'] = master_df.Buy_Sell.map({'SELL': 'SHORT', 'BUY': 'LONG'})

    symbol_mapping = pd.read_csv('Q_Pareto_Transaction_History/Data/mapping/symbol_mapping.csv',
                                 header=0,
                                 index_col=0)
    # Mapping Symbol
    master_df['Name'] = master_df.UnderlyingSymbol.map(symbol_mapping.name.to_dict())
    master_df['Asset Class'] = master_df.UnderlyingSymbol.map(symbol_mapping.assetClass.to_dict())

    # Display the first few rows
    # Define the aggregation dictionary
    agg_dict_IBOrderID = {
        # **Categorical Columns**: Keep the first non-null occurrence (assuming they are the same within a group)
        'ClientAccountID': 'first',
        'CurrencyPrimary': 'first',
        'Symbol': 'first',
        'Description': 'first',
        'Conid': 'first',
        'SecurityID': 'first',
        'ListingExchange': 'first',
        'TradeID': lambda x: ', '.join(x.astype(str).unique()),  # Keep all unique TradeIDs
        'Multiplier': 'first',
        'Strike': 'first',
        'Expiry': 'first',
        'Put_Call': 'first',
        'TradeDate': 'first',
        'TransactionID': 'first',
        'IBExecID': lambda x: ', '.join(x.astype(str).unique()),  # Keep unique executions
        'OrderTime': lambda x: ', '.join(x.astype(str).unique()),  # Keep unique executions
        'FXRateToBase': 'mean',  # Weighted average trade price,  # Averaging the FX rate makes sense
        'AssetClass': 'first',
        'Name': 'first',
        'Asset Class': 'first',
        'SubCategory': 'first',
        'ISIN': 'first',
        'FIGI': 'first',
        'UnderlyingConid': 'first',
        'UnderlyingSymbol': 'first',
        'UnderlyingSecurityID': 'first',
        'UnderlyingListingExchange': 'first',
        'Issuer': 'first',
        'IssuerCountryCode': 'first',
        'ReportDate': 'first',
        'DateTime': 'first',
        'SettleDateTarget': 'first',
        'TransactionType': 'first',
        'Exchange': 'first',
        'Open_CloseIndicator': 'first',
        'Notes_Codes': 'first',
        'Buy_Sell': 'first',
        'BrokerageOrderID': 'first',
        'ExtExecID': 'first',
        'OpenDateTime': 'first',
        'HoldingPeriodDateTime': 'first',
        'LevelOfDetail': 'first',
        'OrderType': 'first',
        'DateTime_clean': 'first',
        'Position': 'first',

        # **Numerical Columns**: Use appropriate aggregations
        'Quantity': 'sum',  # Sum of traded quantities
        'TradePrice': 'mean',  # ,  # Weighted average trade price
        'IBCommission': 'sum',  # Total commission paid
        'CostBasis': 'sum',  # Aggregate cost basis
        'NotionaltoBase': 'sum',  # Aggregate cost basis to base
        'FifoPnlRealized': 'sum',  # Realized profit and loss
        'FifoPnlRealizedToBase': 'sum',  # Realized PnL converted to base currency
        'TradeMoney': 'sum',  # Total trade money
        'Proceeds': 'sum',  # Total proceeds
        'NetCash': 'sum',  # Net cash impact
        'ClosePrice': 'mean',
    }  # Average closing price

    aggregated_df = master_df.copy().groupby(['IBOrderID']).agg(agg_dict_IBOrderID).reset_index().sort_values(
        by='DateTime_clean',
        ascending=True)
    filter_df = aggregated_df[[
        # Basic Trade Information
        'DateTime_clean',
        'Symbol',
        'Description',
        'Name',

        # Trade Metrics
        'Quantity',
        'TradePrice',
        'NotionaltoBase',
        'FifoPnlRealizedToBase',
        'Multiplier',

        # Financial Instrument Information
        'Asset Class',
        'AssetClass',
        'CurrencyPrimary',
        'FXRateToBase',
        'Open_CloseIndicator',
        'Position',

        # Transaction Details
        'IBOrderID',
        'Conid',
        'Exchange',
    ]]

    # open quantity
    open_q = filter_df.groupby('Conid', as_index=False)['Quantity'].sum()
    open_conid = list(open_q.query('Quantity != 0').Conid)
    open_filter_df = filter_df.query('Conid in @open_conid')

    return  open_filter_df
open_rzld_pnl = get_realized_PnL()

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
        'PermID': trade.order.permId,
        'ConID': trade.contract.conId,
        'Symbol': trade.contract.symbol,
        'Local Symbol': trade.contract.localSymbol,
        'SecType': trade.contract.secType,
        'Exchange': trade.contract.exchange,
        'Currency': trade.contract.currency,
        'Multiplier': trade.contract.multiplier if hasattr(trade.contract, 'multiplier') else 1,
        'Order Type': trade.order.orderType,
        'Quantity': trade.order.totalQuantity,
        'Action': trade.order.action,
        'Limit Price': trade.order.lmtPrice if hasattr(trade.order, 'lmtPrice') else None,
        'Stop Price': trade.order.auxPrice if hasattr(trade.order, 'auxPrice') else None,
        'Status': trade.orderStatus.status,
        'Fills': trade.fills
    } for trade in trades if trade.order  # Ensure order exists
])

portfolio = ib.portfolio()
portfolio_df = pd.DataFrame([
    {
        'ConID': pos.contract.conId,
        'Symbol': pos.contract.symbol,
        'Position': pos.position,
        'Unrealized PnL': pos.unrealizedPNL,
        'Realized PnL': pos.realizedPNL, # Directly from IBKR!
        'Market Price': pos.marketPrice,

    } for pos in portfolio
])

# print(portfolio_df)

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

symbol_mapping = pd.read_csv('Q_Pareto_Transaction_History/Data/mapping/symbol_mapping.csv',
                                 header=0,
                                 index_col=0)

risk_df['Name'] = risk_df.Symbol.map(symbol_mapping.name.to_dict())
risk_df['Asset Class'] = risk_df.Symbol.map(symbol_mapping.assetClass.to_dict())

def positionsHistPrices(df, durationStr, barSizeSetting):
    # Define a dictionary to store the close prices
    close_prices_dict = {}
    ATR_30 = {}

    for conid in df.copy().query("ConID not in @defect_ids")['ConID'].unique():

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
        ts_df.set_index('date', inplace=True)
        ts_close = ts_df[['close']]

        period = 30
        # Compute True Range (TR)
        ts_df['Previous Close'] = ts_df['close'].shift(1)
        ts_df['High-Low'] = ts_df['high'] - ts_df['low']
        ts_df['High-PC'] = abs(ts_df['high'] - ts_df['Previous Close'])
        ts_df['Low-PC'] = abs(ts_df['low'] - ts_df['Previous Close'])

        ts_df['TR'] = ts_df[['High-Low', 'High-PC', 'Low-PC']].max(axis=1)

        ts_df['ATR'] = ts_df['TR'].rolling(window=period).mean()
        ATR_30[df.query('ConID == @conid and Exchange != ""').Symbol.unique()[0]] = ts_df['ATR'].values[-1]

        # Store the close prices in the dictionary with the symbol as the key
        close_prices_dict[df.query('ConID == @conid and Exchange != ""').Symbol.unique()[0]] = ts_close['close']

    # Combine all close price DataFrames into one DataFrame
    close_prices_df = pd.DataFrame(close_prices_dict)

    return  [close_prices_df, ATR_30]

def addBaseCCYfx(df, ccy):
    # Fetch FX conversion rates to base currency
    base_currency = ccy  # Set your base currency here
    fx_rates = {}
    fx_time = {}
    for currency in df['Currency'].unique():
        if currency != base_currency:
            if currency != 'KRW':
                fx_contract = Forex(f'{base_currency}{currency}')
            else:
                fx_contract = Forex(f'{currency}{base_currency}')
            # Request historical data for EUR/xxx
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

            if currency != 'KRW':
                fx_rates[currency] = historical_data[-1].close
            else:
                fx_rates[currency] = 1/historical_data[-1].close
            fx_time[currency] = historical_data[-1].date
        else:
            fx_rates[currency] = 1

    # Apply FX conversion rates
    df['FX Rate to Base'] = df['Currency'].map(fx_rates)

    return df

risk_df = addBaseCCYfx(risk_df, 'EUR')

# contracts for Money market purposed
contracts_MM = [11625311, 74991935, 281534370, 301467983, 568953593]
risk_df = risk_df.copy().query("ConID not in @contracts_MM")

risk_df = risk_df.copy().query("SecType not in 'CASH'")
risk_df = risk_df.copy().query("Status not in 'Cancelled'")


nans_lastPX_Ids = {
                    488641260: portfolio_df.query("ConID == 488641260")['Market Price'].values[0], # MDAX cert,
                    230949979 : portfolio_df.query("ConID == 230949979")['Market Price'].values[0], # USD CNH
                    120551943: portfolio_df.query("ConID == 120551943")['Market Price'].values[0]
                   }
defect_ids = list(nans_lastPX_Ids.keys())

contracts_quoted_USd = {526262864: 100,
                        565301283: 100,
                        577421489: 100,
                        532513462: 100,
                        #656391483: 100
}

def addLastPX(df):
    # Fetch FX conversion rates to base currency
    LastPX = nans_lastPX_Ids
    LastPX_time = {}

    for conid in df.copy().query("ConID not in @defect_ids")['ConID'].unique():

        conid = int(conid)
        exchange = df.query('ConID == @conid and Exchange != ""').Exchange.unique()[0]
        contract = Contract(conId=conid, exchange=exchange)

        # Request historical data
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',        # '' means the latest available data
            durationStr='1 Y',     # Duration: 1 day (options: '1 W', '1 M', '1 Y', etc.)
            barSizeSetting='1 day',  # Bar size: 1 hour (options: '1 min', '5 min', etc.)
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
result_query = positionsHistPrices(df = risk_df, durationStr= '1 Y', barSizeSetting= '1 day')
ts = result_query[0]
ATR_30 = result_query[1]
ts_ret = ts.copy().pct_change().dropna()
corr = ts_ret.corr()


risk_df.replace('', np.nan, inplace=True)
risk_df['Multiplier'] = risk_df.Multiplier.fillna(1)

last_risk = pd.DataFrame(columns=[
        'Status',
        'Currency',
        'FX',
        'Symbol',
        'Local Symbol',
        'Name',
        'Asset Class',
        'Position',
        'Contracts',
        'Risk (EUR)',
        'Risk (bps)',
        'Rlzd PnL (EUR)',
        'Rlzd PnL (bps)',
        'UnRlzdPnL(EUR)',
        'UnRlzdPnL(bps)',
        'Tot PnL (EUR)',
        'Tot PnL (bps)',
        'Exposure (EUR)',  # Scalar wrapped in list
        'Expos. (%)',
        'Stop or Trigger',
        'ATR 30D',
        'ATR 30D (%)',
        'multiplier',  # Scalar wrapped in list
        'Last Price',  # Scalar wrapped in list
        'ConID' # Scalar wrapped in list
    ])

df_open_rzld_pnl = open_rzld_pnl.groupby('Conid').FifoPnlRealizedToBase.sum()


# arr = arr[(arr != 620731015)  & (arr != 128832371) & (arr != 526262864)]
for conid in risk_df['ConID'].unique():

    flag_filledANDcanc = risk_df.copy().query('ConID == @conid and (Status != "Cancelled" and Status != "Filled")').empty
    flag_notOpen = conid not in positions_df.ConID

    # adjust for prices quoted in USd (cents)
    if conid in contracts_quoted_USd.keys():
        div = contracts_quoted_USd[conid]
    else:
        div = 1

    if flag_notOpen and flag_filledANDcanc:
        sub_df = risk_df.copy().query('ConID == @conid')
    else:
        sub_df = risk_df.copy().query('ConID == @conid & Status != "Cancelled" & Quantity != 0')

    multiplier = float(sub_df.Multiplier.unique()[0])/div
    lastPX = float(sub_df.LastPX.unique()[0])
    fx = float(sub_df['FX Rate to Base'].unique()[0])


    if sub_df.Position.isna().sum() == len(sub_df):
        position_status = "working"
    else:
        position_status = "open"

    if flag_notOpen and flag_filledANDcanc:
        position_status = "closed"

    if position_status == "open":
        open_position = abs(sub_df.Position.dropna().sum())
        exposure = open_position * multiplier * lastPX / fx
        if risk_df[risk_df.ConID == conid].SecType.values[0] == 'WAR':
            exposure = (sub_df.Position * sub_df.LastPX).values[0]

        rlzd_PnL = portfolio_df[portfolio_df.ConID == conid]['Realized PnL'].values[0]/fx
        if conid in df_open_rzld_pnl.index:
            rlzd_PnL += df_open_rzld_pnl[conid]
        unrlzd_PnL = portfolio_df[portfolio_df.ConID == conid]['Unrealized PnL'].values[0]/fx

        stops = sub_df[sub_df['Order Type'] == 'STP']  # get rid of taking profit orders

        if not stops.empty: # there are stops
            if open_position > 0: # long
                stops = stops.sort_values(by='Stop Price', ascending=False)
            elif open_position < 0: # short
                stops = stops.sort_values(by='Stop Price', ascending=True)

            string_stops = ('P: ' + stops['Stop Price'].round(3).astype(str) + ', Q: ' + stops['Quantity'].astype(int).astype(str) + \
                        ', Dist: ' + (abs(lastPX - stops['Stop Price']) / lastPX * 100).round(2).astype(str) + '%').str.cat(sep=' | ')

            stops['dir'] = stops['Action'].map({'SELL': -1, 'BUY': 1})

            if list(stops.Action)[0] == 'BUY':
                position = 'SHORT'
            else:
                position = 'LONG'

            stops_exec_exp = stops.Quantity * multiplier * (stops['Stop Price']) / fx
            risk = abs(exposure - stops_exec_exp.sum())

        elif stops.empty: # no stops
            string_stops = ''
            if portfolio_df[portfolio_df.ConID == conid].Position.values > 0:
                position = 'LONG'
            else:
                position = 'SHORT'
            risk = exposure

        new_row = pd.DataFrame(data={
            'Status': [position_status],
            'Currency': [sub_df.Currency.unique()[0]],  # Make sure it's a list
            'FX': [fx],  # Scalar wrapped in list
            'Symbol': [sub_df.Symbol.unique()[0]],  # Make sure it's a list
            'Local Symbol': [sub_df['Local Symbol'].unique()[0]],  # Make sure it's a list
            'Name': [sub_df.Name.unique()[0]],
            'Asset Class': [sub_df['Asset Class'].unique()[0]],
            'Position': [position],
            'Contracts': [open_position],  # Scalar wrapped in list
            'Risk (EUR)': [risk],  # Scalar wrapped in list
            'Risk (bps)': [(risk / NLV) * 10000],
            'Rlzd PnL (EUR)': [rlzd_PnL],
            'Rlzd PnL (bps)': [(rlzd_PnL/ NLV) * 10000],
            'UnRlzdPnL(EUR)': [unrlzd_PnL],
            'UnRlzdPnL(bps)': [(unrlzd_PnL / NLV) * 10000],
            'Tot PnL (EUR)': [rlzd_PnL + unrlzd_PnL],
            'Tot PnL (bps)': [((rlzd_PnL + unrlzd_PnL)/ NLV) * 10000],
            'Exposure (EUR)': [exposure],  # Scalar wrapped in list
            'Expos. (%)': [exposure / NLV * 100],
            'Stop or Trigger': [string_stops],
            'ATR 30D': [ATR_30.get(sub_df.Symbol.unique()[0], np.nan)],
            'ATR 30D (%)': [ATR_30.get(sub_df.Symbol.unique()[0], np.nan) / lastPX],
            'multiplier': [multiplier],  # Scalar wrapped in list
            'Last Price': [lastPX],  # Scalar wrapped in list
            'ConID': [conid]  # Scalar wrapped in list
        })

        last_risk = pd.concat([last_risk, new_row], ignore_index=True)

    elif position_status == "working":
        exposure = 0
        open_position = 0
        rlzd_PnL = 0
        unrlzd_PnL = 0

        groups = {k: v for k, v in sub_df.groupby('Quantity')}

        for order in groups.values():

            order = order.sort_values(by='PermID')
            stops = order[(order['Status'] == 'PreSubmitted') & (order['Order Type'] == 'STP')].reset_index()  # get rid of taking profit orders

            if not stops.empty:
                if len(stops.Action.unique()) == 1:
                    triggers = order[((order['Status'] == 'Submitted') & (order['Order Type'].isin(['LMT', 'STP LMT']))) |((order['Status'] == 'PreSubmitted') & (order['Order Type'].isin(['STP LMT'])))].reset_index()
                    type = 'Limit Price'
                else:
                    triggers = stops[stops.Action == stops.Action.unique()[0]].reset_index()
                    stops = stops[stops.Action != stops.Action.unique()[0]].reset_index()
                    type = 'Stop Price'

                stops['dir'] = stops['Action'].map({'SELL': -1, 'BUY': 1})

                if list(stops.Action)[0] == 'SELL':
                    position = 'LONG'
                else:
                    position = 'SHORT'

                string_stops = ('P: ' + triggers[type].round(3).astype(str) + ', Q: ' + triggers['Quantity'].astype(int).astype(str) + \
                        ', Dist: ' + (abs(lastPX - triggers[type]) / lastPX * 100).round(2).astype(str) + '%').str.cat(sep=' | ')

                risk = (stops.Quantity * multiplier * (stops['Stop Price'] - triggers[type]) * stops.dir).sum() / fx

                new_row = pd.DataFrame(data={
                    'Status': [position_status],
                    'Currency': [stops.Currency.unique()[0]],  # Make sure it's a list
                    'FX': [fx],  # Scalar wrapped in list
                    'Symbol': [stops.Symbol.unique()[0]],  # Make sure it's a list
                    'Local Symbol': [stops['Local Symbol'].unique()[0]],  # Make sure it's a list
                    'Name': [stops.Name.unique()[0]],
                    'Asset Class': [stops['Asset Class'].unique()[0]],
                    'Position': [position],
                    'Contracts': [open_position],  # Scalar wrapped in list
                    'Risk (EUR)': [risk],  # Scalar wrapped in list
                    'Risk (bps)': [(risk / NLV) * 10000],
                    'Rlzd PnL (EUR)': [rlzd_PnL],
                    'Rlzd PnL (bps)': [(rlzd_PnL/ NLV) * 10000],
                    'UnRlzdPnL(EUR)': [unrlzd_PnL],
                    'UnRlzdPnL(bps)': [(unrlzd_PnL / NLV) * 10000],
                    'Tot PnL (EUR)': [rlzd_PnL + unrlzd_PnL],
                    'Tot PnL (bps)': [((rlzd_PnL + unrlzd_PnL)/ NLV) * 10000],
                    'Exposure (EUR)': [exposure],  # Scalar wrapped in list
                    'Expos. (%)': [exposure / NLV * 100],
                    'Stop or Trigger': [string_stops],
                    'ATR 30D': [ATR_30.get(stops.Symbol.unique()[0], np.nan)],
                    'ATR 30D (%)': [ATR_30.get(stops.Symbol.unique()[0], np.nan) / lastPX],
                    'multiplier': [multiplier],  # Scalar wrapped in list
                    'Last Price': [lastPX],  # Scalar wrapped in list
                    'ConID': [conid]  # Scalar wrapped in list
                })

                last_risk = pd.concat([last_risk, new_row], ignore_index=True)

    elif position_status == "closed":
        exposure = 0
        open_position = 0
        risk = 0
        unrlzd_PnL = 0
        string_stops = np.nan


        rlzd_PnL = 0
        for fill in orders_df.query('ConID == @conid').Fills:
            if fill == []:
                continue
            rlzd_PnL += fill[0].commissionReport.realizedPNL

        rlzd_PnL += df_open_rzld_pnl[conid]

        if list(sub_df.Action)[-1] == 'SELL':
            position = 'LONG'
        else:
            position = 'SHORT'

        new_row = pd.DataFrame(data={
            'Status': [position_status],
            'Currency': [sub_df.Currency.unique()[0]],  # Make sure it's a list
            'FX': [fx],  # Scalar wrapped in list
            'Symbol': [sub_df.Symbol.unique()[0]],  # Make sure it's a list
            'Local Symbol': [sub_df['Local Symbol'].unique()[0]],  # Make sure it's a list
            'Name': [sub_df.Name.unique()[0]],
            'Asset Class': [sub_df['Asset Class'].unique()[0]],
            'Position': [position],
            'Contracts': [open_position],  # Scalar wrapped in list
            'Risk (EUR)': [risk],  # Scalar wrapped in list
            'Risk (bps)': [(risk / NLV) * 10000],
            'Rlzd PnL (EUR)': [rlzd_PnL],
            'Rlzd PnL (bps)': [(rlzd_PnL/ NLV) * 10000],
            'UnRlzdPnL(EUR)': [unrlzd_PnL],
            'UnRlzdPnL(bps)': [(unrlzd_PnL / NLV) * 10000],
            'Tot PnL (EUR)': [rlzd_PnL + unrlzd_PnL],
            'Tot PnL (bps)': [((rlzd_PnL + unrlzd_PnL)/ NLV) * 10000],
            'Exposure (EUR)': [exposure],  # Scalar wrapped in list
            'Expos. (%)': [exposure / NLV * 100],
            'Stop or Trigger': [string_stops],
            'ATR 30D': [ATR_30.get(sub_df.Symbol.unique()[0], np.nan)],
            'ATR 30D (%)': [ATR_30.get(sub_df.Symbol.unique()[0], np.nan) / lastPX],
            'multiplier': [multiplier],  # Scalar wrapped in list
            'Last Price': [lastPX],  # Scalar wrapped in list
            'ConID': [conid]  # Scalar wrapped in list
        })

        last_risk = pd.concat([last_risk, new_row], ignore_index=True)




last_risk.to_csv("Q_Pareto_Transaction_History/Data/open_risks.csv")
corr.to_csv("Q_Pareto_Transaction_History/Data/corr_matrix.csv")

# Disconnect from IBKR
ib.disconnect()
