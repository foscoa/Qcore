from ib_insync import *
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from pathlib import Path
from typing import List

DATA_DIR = Path("Q_Pareto_Transaction_History_DEV/Data")
TRADE_HISTORY_FILE = DATA_DIR / "U15721173_TradeHistory_05052025.csv"
SYMBOL_MAPPING_FILE = DATA_DIR / "mapping/symbol_mapping.csv"


def load_trade_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.replace("/", "_", regex=False)
    return df

def clean_and_filter_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.query('LevelOfDetail == "EXECUTION" and Open_CloseIndicator in ["C", "O"]')

    # Clean datetime
    df['DateTime_clean'] = pd.to_datetime(
        df['DateTime'].str.replace(";", " "), format="%Y%m%d %H%M%S"
    ).dt.tz_localize('America/New_York').dt.tz_convert('Europe/Berlin').dt.tz_localize(None)

    # Sort by datetime
    df = df.sort_values(by='DateTime_clean', ascending=False)

    # Derived fields
    df['FifoPnlRealizedToBase'] = df['FifoPnlRealized'] * df['FXRateToBase']
    df['NotionaltoBase'] = df['Quantity'].abs() * df['Multiplier'] * df['FXRateToBase'] * df['TradePrice']
    df['Position'] = df['Buy_Sell'].map({'SELL': 'SHORT', 'BUY': 'LONG'})

    return df

def map_symbols(df: pd.DataFrame, symbol_mapping_file: Path) -> pd.DataFrame:
    symbol_mapping = pd.read_csv(symbol_mapping_file, index_col=0)
    df['Name'] = df['UnderlyingSymbol'].map(symbol_mapping.name.to_dict())
    df['Asset Class'] = df['UnderlyingSymbol'].map(symbol_mapping.assetClass.to_dict())
    return df

def aggregate_by_order_id(df: pd.DataFrame) -> pd.DataFrame:
    aggregation_rules = {
        # Categorical
        'ClientAccountID': 'first',
        'CurrencyPrimary': 'first',
        'Symbol': 'first',
        'Description': 'first',
        'Conid': 'first',
        'SecurityID': 'first',
        'ListingExchange': 'first',
        'TradeID': lambda x: ', '.join(x.astype(str).unique()),
        'Multiplier': 'first',
        'Strike': 'first',
        'Expiry': 'first',
        'Put_Call': 'first',
        'TradeDate': 'first',
        'TransactionID': 'first',
        'IBExecID': lambda x: ', '.join(x.astype(str).unique()),
        'OrderTime': lambda x: ', '.join(x.astype(str).unique()),
        'FXRateToBase': 'mean',
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

        # Numeric
        'Quantity': 'sum',
        'TradePrice': 'mean',
        'IBCommission': 'sum',
        'CostBasis': 'sum',
        'NotionaltoBase': 'sum',
        'FifoPnlRealized': 'sum',
        'FifoPnlRealizedToBase': 'sum',
        'TradeMoney': 'sum',
        'Proceeds': 'sum',
        'NetCash': 'sum',
        'ClosePrice': 'mean',
    }

    return df.groupby('IBOrderID').agg(aggregation_rules).reset_index().sort_values(by='DateTime_clean')

def filter_open_positions(df: pd.DataFrame) -> pd.DataFrame:
    open_q = df.groupby('Conid', as_index=False)['Quantity'].sum()
    open_conids = open_q.query('Quantity != 0')['Conid'].tolist()
    return df[df['Conid'].isin(open_conids)]

def get_realized_pnl(file_path: Path = TRADE_HISTORY_FILE,
                     symbol_map_path: Path = SYMBOL_MAPPING_FILE) -> pd.DataFrame:
    raw_df = load_trade_data(file_path)
    cleaned_df = clean_and_filter_data(raw_df)
    mapped_df = map_symbols(cleaned_df, symbol_map_path)
    aggregated_df = aggregate_by_order_id(mapped_df)

    columns_of_interest = [
        'DateTime_clean', 'Symbol', 'Description', 'Name', 'Quantity', 'TradePrice', 'NotionaltoBase',
        'FifoPnlRealizedToBase', 'Multiplier', 'Asset Class', 'AssetClass', 'CurrencyPrimary',
        'FXRateToBase', 'Open_CloseIndicator', 'Position', 'IBOrderID', 'Conid', 'Exchange'
    ]
    filtered_df = aggregated_df[columns_of_interest]
    open_positions_df = filter_open_positions(filtered_df)

    return open_positions_df

open_rzld_pnl = get_realized_pnl()

#####

# === IB Connection ===
def connect_ibkr(host: str = '127.0.0.1', port: int = 7496, client_id: int = 1) -> IB:
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    return ib


# === Data Retrieval ===
def fetch_positions(ib: IB) -> pd.DataFrame:
    positions: List[Position] = ib.positions()
    return pd.DataFrame([
        {
            'ConID': pos.contract.conId,
            'Symbol': pos.contract.symbol,
            'Local Symbol': pos.contract.localSymbol,
            'SecType': pos.contract.secType,
            'Exchange': pos.contract.exchange,
            'Currency': pos.contract.currency,
            'Multiplier': getattr(pos.contract, 'multiplier', 1),
            'Position': pos.position,
            'Avg Cost': pos.avgCost
        }
        for pos in positions
    ])

def fetch_open_orders(ib: IB) -> pd.DataFrame:
    ib.reqAllOpenOrders()
    trades: List[Trade] = ib.trades()

    return pd.DataFrame([
        {
            'PermID': trade.order.permId,
            'ConID': trade.contract.conId,
            'Symbol': trade.contract.symbol,
            'Local Symbol': trade.contract.localSymbol,
            'SecType': trade.contract.secType,
            'Exchange': trade.contract.exchange,
            'Currency': trade.contract.currency,
            'Multiplier': getattr(trade.contract, 'multiplier', 1),
            'Order Type': trade.order.orderType,
            'Quantity': trade.order.totalQuantity,
            'Action': trade.order.action,
            'Limit Price': getattr(trade.order, 'lmtPrice', None),
            'Stop Price': getattr(trade.order, 'auxPrice', None),
            'Status': trade.orderStatus.status,
            'Fills': trade.fills
        }
        for trade in trades if trade.order
    ])

def fetch_portfolio(ib: IB) -> pd.DataFrame:
    portfolio: List[PortfolioItem] = ib.portfolio()
    return pd.DataFrame([
        {
            'ConID': item.contract.conId,
            'Symbol': item.contract.symbol,
            'Position': item.position,
            'Unrealized PnL': item.unrealizedPNL,
            'Realized PnL': item.realizedPNL,
            'Market Price': item.marketPrice
        }
        for item in portfolio
    ])

def fetch_net_liquidation_value(ib: IB) -> float:
    summary: List[TagValue] = ib.accountSummary()
    summary_df = pd.DataFrame([
        {'Tag': tag.tag, 'Value': tag.value, 'Currency': tag.currency}
        for tag in summary
    ])
    return float(summary_df.loc[summary_df['Tag'] == 'NetLiquidation', 'Value'].values[0])

# === Merging and Mapping ===
def merge_position_order_data(positions_df: pd.DataFrame,
                               orders_df: pd.DataFrame) -> pd.DataFrame:
    return positions_df.merge(
        orders_df,
        on=['ConID', 'Symbol', 'Local Symbol', 'SecType', 'Exchange', 'Currency', 'Multiplier'],
        how='outer',
        suffixes=('_Position', '_Order')
    )

def enrich_with_symbol_mapping(df: pd.DataFrame, mapping_file: Path) -> pd.DataFrame:
    symbol_mapping = pd.read_csv(mapping_file, index_col=0)
    df['Name'] = df['Symbol'].map(symbol_mapping.name.to_dict())
    df['Asset Class'] = df['Symbol'].map(symbol_mapping.assetClass.to_dict())
    return df

# === Main Risk Overview Builder ===
def build_ibkr_risk_snapshot() -> pd.DataFrame:
    ib = connect_ibkr()
    positions_df = fetch_positions(ib)
    orders_df = fetch_open_orders(ib)
    portfolio_df = fetch_portfolio(ib)
    nlv = fetch_net_liquidation_value(ib)

    risk_df = merge_position_order_data(positions_df, orders_df)
    risk_df = enrich_with_symbol_mapping(risk_df, SYMBOL_MAPPING_FILE)

    print(f"Net Liquidation Value (Base Currency): {nlv:,.2f}")
    return risk_df

risk_snapshot_df = build_ibkr_risk_snapshot()


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

# contracts for Money market purposes
contracts_MM = [11625311, 74991935, 281534370, 301467983, 568953593]
risk_df = risk_df.copy().query("ConID not in @contracts_MM")

risk_df = risk_df.copy().query("SecType not in 'CASH'")
risk_df = risk_df.copy().query("Status not in 'Cancelled'")


nans_lastPX_Ids = {
                    488641260: 6.51, # MDAX cert,
                    230949979 : 7.2435
                    # 120550477: 533.56, # BKR B
                   }
defect_ids = list(nans_lastPX_Ids.keys())

contracts_quoted_USd = {526262864: 100,
                        565301283: 100,
                        577421489: 100,
                        532513438: 100,
                        573366572: 100,
                        725809839: 100, # Feeder Cattle
                        577421487: 100  # Coffee "C"
}

# map CFDs conid with STK
map_conid = {
    120550477: Contract(secType='STK', conId=72063691, symbol='BRK B', exchange='SMART', primaryExchange='NYSE', currency='USD', localSymbol='BRK B', tradingClass='BRK B'),
    230949979: Forex('USDCNH', conId=113342317, exchange='IDEALPRO', localSymbol='USD.CNH', tradingClass='USD.CNH'),
    166176201: Contract(secType='STK', conId=166090175, symbol='BABA', exchange='SMART', primaryExchange='NYSE', currency='USD', localSymbol='BABA', tradingClass='BABA')
}


def addLastPX(df):
    # Fetch FX conversion rates to base currency
    LastPX = nans_lastPX_Ids
    LastPX_time = {}

    for conid in df.copy().query("ConID not in @defect_ids")['ConID'].unique():

        conid = int(conid)
        if conid in map_conid.keys():
            contract = map_conid[conid]
        else:
            conid = int(conid)
            exchange = df.query('ConID == @conid and Exchange != ""').Exchange.unique()[0]
            contract = Contract(conId=conid, exchange=exchange)

        # Request historical data
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',        # '' means the latest available data
            durationStr='1 D',     # Duration: 1 day (options: '1 W', '1 M', '1 Y', etc.)
            barSizeSetting='1 min',  # Bar size: 1 hour (options: '1 min', '5 min', etc.)
            whatToShow='TRADES',  # Can be 'TRADES', 'BID', 'ASK', 'MIDPOINT'
            useRTH=False,           # Regular Trading Hours only
            formatDate=1
        )
        ib.sleep(1)  # Allow time to fetch market data
        LastPX[conid] = bars[-1].close
        LastPX_time[conid] = bars[-1].date

    # Apply FX conversion rates
    df['LastPX'] = df['ConID'].map(LastPX)

    return df

def positionsHistPrices(df, durationStr, barSizeSetting):
    # Define a dictionary to store the close prices
    close_prices_dict = {}
    ATR_30 = {}

    for conid in df.copy().query("ConID not in @defect_ids")['ConID'].unique():

        conid = int(conid)
        if conid in map_conid.keys():
            contract = map_conid[conid]
        else:
            conid = int(conid)
            exchange = df.query('ConID == @conid and Exchange != ""').Exchange.unique()[0]
            contract = ib.qualifyContracts(Contract(conId=conid))[0]

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


risk_df = addLastPX(risk_df)
result_query = positionsHistPrices(df = risk_df, durationStr= '1 Y', barSizeSetting= '1 day')
ts = result_query[0]
ATR_30 = result_query[1]

# correlation, assumption
def calculate_correlation(durationStr= '1 M', barSizeSetting= '30 mins'):
    time_series_prices = positionsHistPrices(df = risk_df, durationStr= durationStr, barSizeSetting= barSizeSetting)[0]
    time_series_ret = time_series_prices.copy().pct_change(fill_method=None) # fill_method=None to keep NANs
    corr = time_series_ret.corr()

    return corr

# log report time
# Set timezone to Zurich
zurich_tz = pytz.timezone('Europe/Zurich')
# Get current time in Zurich
report_time = datetime.now(zurich_tz)


if flag_update_corr == True:

    durationStr = '1 M'
    barSizeSetting = '30 mins'

    corr = calculate_correlation(durationStr= durationStr, barSizeSetting= barSizeSetting)
    corr.to_csv("Q_Pareto_Transaction_History_DEV/Data/corr_matrix.csv")
    corr.to_csv("C:/Users/FoscoAntognini/DREI-R GROUP/QCORE AG - Documents/Investments/Trading App/PROD/open_risks/corr_matrix.csv")

    # log assumptions
    corr_assumptions = {
        "report_time": report_time.strftime("%A, %d %B %Y - %H:%M:%S %Z"),
        "barSizeSetting": barSizeSetting,
        "durationStr": durationStr
    }

    # write assumptions
    pd.DataFrame([corr_assumptions]).to_csv("Q_Pareto_Transaction_History_DEV/Data/corr_matrix_ass.csv", index=False)
    pd.DataFrame([corr_assumptions]).to_csv("C:/Users/FoscoAntognini/DREI-R GROUP/QCORE AG - Documents/Investments/Trading App/PROD/open_risks/corr_matrix_ass.csv", index=False)


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

    sub_df_iter = [sub_df]

    # hybrid orders
    hybrid_IDs = [#727764322, # GBS
                  # 304037456,  # CL
                  # 532513438, #ZL
                  ]
    if conid in hybrid_IDs:
        open_q = abs(sub_df.Position.dropna().values[0])
        open_sub_df = sub_df[(sub_df.Quantity.isna()) | (sub_df.Quantity == open_q)]
        working_sub_df = sub_df[(sub_df.Quantity.notna()) & (sub_df.Quantity != open_q)]

        # open and working orders have the same quantity
        if conid == 532513438:
            permIDs = [585675972]
            open_sub_df = sub_df.query('PermID not in @permIDs')
            working_sub_df = sub_df.query('PermID in @permIDs')

        sub_df_iter = [open_sub_df, working_sub_df]


    for sub_df in sub_df_iter:

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

            rlzd_PnL = portfolio_df[portfolio_df.ConID == conid]['Realized PnL'].values[0] / fx

            if conid in open_rzld_pnl.Conid.unique():

                conid_rlzd_pnl = open_rzld_pnl.query('Conid == @conid')
                closed_same_contract = (conid_rlzd_pnl.Quantity.cumsum() == 0)
                if closed_same_contract.sum() != 0:
                    last_exec = np.where(conid_rlzd_pnl.Quantity.cumsum() == 0)[0][-1]
                else:
                    last_exec = -1
                rlzd_PnL += conid_rlzd_pnl.FifoPnlRealizedToBase.iloc[(last_exec+1):].sum()

            unrlzd_PnL = portfolio_df[portfolio_df.ConID == conid]['Unrealized PnL'].values[0] / fx

            stops = sub_df[sub_df['Order Type'] == 'STP']  # get rid of taking profit orders

            if not stops.empty:  # there are stops
                if open_position > 0:  # long
                    stops = stops.sort_values(by='Stop Price', ascending=False)
                elif open_position < 0:  # short
                    stops = stops.sort_values(by='Stop Price', ascending=True)

                string_stops = ('P: ' + stops['Stop Price'].round(3).astype(str) + ', Q: ' + stops['Quantity'].astype(
                    int).astype(str) + \
                                ', Dist: ' + (abs(lastPX - stops['Stop Price']) / lastPX * 100).round(2).astype(
                            str) + '%').str.cat(sep=' | ')

                stops['dir'] = stops['Action'].map({'SELL': -1, 'BUY': 1})

                if list(stops.Action)[0] == 'BUY':
                    position = 'SHORT'
                else:
                    position = 'LONG'

                stops_exec_exp = stops.Quantity * multiplier * (stops['Stop Price']) / fx
                risk = abs(exposure - stops_exec_exp.sum())

            elif stops.empty:  # no stops
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
                'Rlzd PnL (bps)': [(rlzd_PnL / NLV) * 10000],
                'UnRlzdPnL(EUR)': [unrlzd_PnL],
                'UnRlzdPnL(bps)': [(unrlzd_PnL / NLV) * 10000],
                'Tot PnL (EUR)': [rlzd_PnL + unrlzd_PnL],
                'Tot PnL (bps)': [((rlzd_PnL + unrlzd_PnL) / NLV) * 10000],
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
            exposure = np.nan
            rlzd_PnL = np.nan
            unrlzd_PnL = np.nan

            groups = {k: v for k, v in sub_df.groupby('Quantity')}

            for order in groups.values():

                order = order.sort_values(by='PermID')
                stops = order[(order['Status'] == 'PreSubmitted') & (
                            order['Order Type'] == 'STP')].reset_index()  # get rid of taking profit orders

                if not stops.empty:
                    if len(stops.Action.unique()) == 1:
                        triggers = order[
                            ((order['Status'] == 'Submitted') & (order['Order Type'].isin(['LMT', 'STP LMT', 'STP']))) | (
                                        (order['Status'] == 'PreSubmitted') & (
                                    order['Order Type'].isin(['STP LMT'])))].reset_index()
                        if (triggers['Limit Price'] != 0).values:
                            type = 'Limit Price'
                        else:
                            type = 'Stop Price'

                    else:
                        triggers = stops[stops.Action == stops.Action.unique()[0]].reset_index()
                        stops = stops[stops.Action != stops.Action.unique()[0]].reset_index()
                        type = 'Stop Price'

                    stops['dir'] = stops['Action'].map({'SELL': -1, 'BUY': 1})

                    if list(stops.Action)[0] == 'SELL':
                        position = 'LONG'
                    else:
                        position = 'SHORT'

                    string_stops = ('P: ' + triggers[type].round(3).astype(str) + ', Q: ' + triggers['Quantity'].astype(
                        int).astype(str) + \
                                    ', Dist: ' + (abs(lastPX - triggers[type]) / lastPX * 100).round(2).astype(
                                str) + '%').str.cat(sep=' | ')

                    open_position = triggers['Quantity'].values[0].astype(int)

                    risk = (stops.Quantity * multiplier * (stops['Stop Price'] - triggers[type]) * stops.dir).sum() / fx
                    exposure = abs((stops.Quantity * multiplier * triggers[type] * stops.dir).sum()) / fx

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
                        'Rlzd PnL (bps)': [(rlzd_PnL / NLV) * 10000],
                        'UnRlzdPnL(EUR)': [unrlzd_PnL],
                        'UnRlzdPnL(bps)': [(unrlzd_PnL / NLV) * 10000],
                        'Tot PnL (EUR)': [rlzd_PnL + unrlzd_PnL],
                        'Tot PnL (bps)': [((rlzd_PnL + unrlzd_PnL) / NLV) * 10000],
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
            exposure = np.nan
            open_position = np.nan
            rlzd_PnL = sum(
                    fill.commissionReport.realizedPNL
                    for fills_list in sub_df['Fills']
                    for fill in fills_list
            ) /fx

            if conid in open_rzld_pnl.Conid.unique():

                conid_rlzd_pnl = open_rzld_pnl.query('Conid == @conid')
                closed_same_contract = (conid_rlzd_pnl.Quantity.cumsum() == 0)
                if closed_same_contract.sum() != 0:
                    last_exec = np.where(conid_rlzd_pnl.Quantity.cumsum() == 0)[0][-1]
                else:
                    last_exec = -1
                rlzd_PnL += conid_rlzd_pnl.FifoPnlRealizedToBase.iloc[(last_exec + 1):].sum()


            unrlzd_PnL = np.nan
            position = 'CLOSED'
            risk = np.nan
            string_stops = np.nan

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
                'Rlzd PnL (bps)': [(rlzd_PnL / NLV) * 10000],
                'UnRlzdPnL(EUR)': [unrlzd_PnL],
                'UnRlzdPnL(bps)': [(unrlzd_PnL / NLV) * 10000],
                'Tot PnL (EUR)': [rlzd_PnL + unrlzd_PnL],
                'Tot PnL (bps)': [((rlzd_PnL + unrlzd_PnL) / NLV) * 10000],
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

last_risk['NLV'] = NLV
last_risk['Report Time'] = report_time.strftime("%A, %d %B %Y - %H:%M:%S %Z")


last_risk['maxL/minP (EUR)'] = np.where(last_risk["Status"] != 'working',
                                        last_risk['Tot PnL (EUR)'] - last_risk['Risk (EUR)'], np.nan)
last_risk['maxL/minP (bps)'] = (last_risk['maxL/minP (EUR)']/NLV)*10000


last_risk = last_risk[['Status', 'Currency', 'FX', 'Symbol', 'Local Symbol', 'Name',
       'Asset Class', 'Position', 'Contracts', 'Risk (EUR)', 'Risk (bps)','maxL/minP (EUR)', 'maxL/minP (bps)',
       'Rlzd PnL (EUR)', 'Rlzd PnL (bps)', 'UnRlzdPnL(EUR)', 'UnRlzdPnL(bps)',
       'Tot PnL (EUR)', 'Tot PnL (bps)', 'Exposure (EUR)', 'Expos. (%)',
       'Stop or Trigger', 'ATR 30D', 'ATR 30D (%)', 'multiplier', 'Last Price',
       'ConID', 'NLV', 'Report Time']]

last_risk.to_csv("Q_Pareto_Transaction_History_DEV/Data/open_risks.csv")
# last_risk.to_csv("C:/Users/FoscoAntognini/DREI-R GROUP/QCORE AG - Documents/Investments/Trading App/PROD/open_risks/open_risks.csv")


# Disconnect from IBKR
ib.disconnect()
