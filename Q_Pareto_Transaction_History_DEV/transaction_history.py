import pandas as pd
import numpy as np

# Define the file path
file_path = "Q_Pareto_Transaction_History_PROD/Data/TradeHistory_raw.csv"

# Read the CSV file
df = pd.read_csv(file_path)
df.columns = df.columns.str.replace("/", "_", regex=False)

master_df = df.copy().query('LevelOfDetail == "EXECUTION" & '
                     'Open_CloseIndicator == "C"')

# clean DateTime
# Remove single quotes and replace ";" with a space
clean_date = master_df.DateTime.str.replace(";", " ")

# Convert to datetime in CET timezone
master_df['DateTime_clean'] = pd.to_datetime(clean_date, format="%Y%m%d %H%M%S").dt.tz_localize('America/New_York').dt.tz_convert('Europe/Berlin')

# Sort by Time
master_df = master_df.copy().sort_values(by='DateTime_clean', ascending=False)

# Adding FifoPnlRealzed in Base Currency
master_df['FifoPnlRealizedToBase'] = master_df.FifoPnlRealized * master_df.FXRateToBase

# Define weighted average function
def weighted_avg_price(x):
    return (x['TradePrice'] * x['Quantity']).sum() / x['Quantity'].sum()


# Display the first few rows
# Define the aggregation dictionary
agg_dict = {
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

    # **Numerical Columns**: Use appropriate aggregations
    'Quantity': 'sum',  # Sum of traded quantities
    'TradePrice': 'mean',#,  # Weighted average trade price
    'IBCommission': 'sum',  # Total commission paid
    'CostBasis': 'sum',  # Aggregate cost basis
    'FifoPnlRealized': 'sum',  # Realized profit and loss
    'FifoPnlRealizedToBase': 'sum',  # Realized PnL converted to base currency
    'TradeMoney': 'sum',  # Total trade money
    'Proceeds': 'sum',  # Total proceeds
    'NetCash': 'sum',  # Net cash impact
    'ClosePrice': 'mean'}  # Average closing price

aggregated_df = master_df.copy().groupby(['IBOrderID']).agg(agg_dict).reset_index().sort_values(by='DateTime_clean', ascending=False)

filter_df = aggregated_df[[
    'DateTime_clean',
    'CurrencyPrimary',
    'Symbol',
    'Description',
    'Conid',
    'Quantity',
    'TradePrice',
    'LevelOfDetail',
    'FifoPnlRealizedToBase',
    'IBOrderID'
]]

