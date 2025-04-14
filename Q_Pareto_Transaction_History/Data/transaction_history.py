import pandas as pd

"""
    Function to load, clean, process, and aggregate trade data from a CSV file.
    Returns a cleaned and aggregated DataFrame.
    """

file_path = "Q_Pareto_Transaction_History/Data/U15721173_TradeHistory_04112025.csv"

# Read the CSV file
df = pd.read_csv(file_path)
df.columns = df.columns.str.replace("/", "_", regex=False)

# Filter for execution-level trades that are either "Open" or "Close"
trade_df = df.query('LevelOfDetail == "EXECUTION" & Open_CloseIndicator in ["C", "O"]').copy()

# Clean and convert DateTime column
trade_df['DateTime_clean'] = (
    pd.to_datetime(trade_df.DateTime.str.replace(";", " "), format="%Y%m%d %H%M%S", errors='coerce')
    .dt.tz_localize('America/New_York')
    .dt.tz_convert('Europe/Berlin')
    .dt.tz_localize(None)
)

# Sort by DateTime in descending order
trade_df = trade_df.sort_values(by='DateTime_clean', ascending=False)

# Calculate Realized PnL and Notional value in Base Currency
trade_df['FifoPnlRealizedToBase'] = trade_df['FifoPnlRealized'] * trade_df['FXRateToBase']
trade_df['NotionalToBase'] = trade_df['Quantity'] * trade_df['Multiplier'] * trade_df['FXRateToBase'] * \
                             trade_df['TradePrice']

# Map buy/sell directions to position types
trade_df['Position'] = trade_df['Buy_Sell'].map({'SELL': 'SHORT', 'BUY': 'LONG'})

# Load symbol mapping file
symbol_mapping = pd.read_csv('Q_Pareto_Transaction_History/Data/mapping/symbol_mapping.csv', index_col=0)

# Map underlying symbols to asset classes and names
trade_df['Name'] = trade_df['UnderlyingSymbol'].map(symbol_mapping['name'].to_dict())
trade_df['Instr.'] = trade_df['AssetClass']
trade_df['AssetClass'] = trade_df['UnderlyingSymbol'].map(symbol_mapping['assetClass'].to_dict())

# Define aggregation rules
agg_dict_IBOrderID = {
    # Categorical fields: keep first non-null occurrence
    'ClientAccountID': 'first',
    'CurrencyPrimary': 'first',
    'Symbol': 'first',
    'Description': 'first',
    'Conid': 'first',
    'Multiplier': 'first',
    'TradeDate': 'first',
    'Position': 'first',
    'AssetClass': 'first',
    'Instr.': 'first',
    'Name': 'first',
    'Open_CloseIndicator': 'first',
    'Exchange': 'first',

    # Numerical fields: apply appropriate aggregations
    'Quantity': 'sum',
    'TradePrice': 'mean',
    'IBCommission': 'sum',
    'NotionalToBase': 'sum',
    'FifoPnlRealizedToBase': 'sum',
    'DateTime_clean': 'first',
    'FXRateToBase': 'mean',  # Weighted average trade price,  # Averaging the FX rate makes sense
}

# Aggregate trades by IBOrderID
aggregated_df = trade_df.groupby('IBOrderID').agg(agg_dict_IBOrderID).reset_index()
aggregated_df = aggregated_df.sort_values(by='DateTime_clean', ascending=False)


def aggregate_closed_positions(df):
    """
    Aggregates closed positions by identifying fully offsetting trades.
    """

    aggregated_positions = pd.DataFrame()

    # Group by contract ID (Conid)
    for conid, group in df.groupby('Conid'):
        group = group.sort_values(by='DateTime_clean', ascending=True)

        # Track cumulative quantity to detect closed positions
        group['CumulativeQuantity'] = group['Quantity'].cumsum()

        # Identify indices where cumulative quantity returns to zero
        zero_indices = np.where(group['CumulativeQuantity'] == 0)[0] + 1
        zero_indices = np.insert(zero_indices, 0, 0)

        for i in range(len(zero_indices) - 1):
            trade_slice = group.iloc[zero_indices[i]:zero_indices[i + 1]].copy()

            if trade_slice.empty:
                continue  # Skip if no trades in this slice

            closed_trades = trade_slice[trade_slice['Open_CloseIndicator'] == "C"]
            open_trades = trade_slice[trade_slice['Open_CloseIndicator'] == "O"]

            # Aggregate metrics
            slice_agg = trade_slice.groupby('Conid').agg({
                'DateTime_clean': 'last',
                'Symbol': 'first',
                'Description': 'first',
                'Name': 'first',
                'Open_CloseIndicator': 'first',
                'IBOrderID': lambda x: ', '.join(x.astype(str).unique()),
                'Exchange': 'first',
                'AssetClass': 'first',
                'CurrencyPrimary': 'first',
                'FXRateToBase': 'last',
                'Position': 'first',
                'Quantity': lambda x: x.cumsum().abs().max(),
                'NotionalToBase': lambda x: x.cumsum().abs().max(),
                'FifoPnlRealizedToBase': 'sum',
                'Multiplier': 'first',
                'Instr.': 'first',
            }).reset_index()

            # Add derived metrics
            slice_agg['FirstEntryDate'] = trade_slice['DateTime_clean'].iloc[0]
            slice_agg['AvgClosePrice'] = (
                    (closed_trades['TradePrice'] * closed_trades['Quantity'].abs()).sum() /
                    closed_trades['Quantity'].abs().sum()
            ) if closed_trades['Quantity'].abs().sum() > 0 else np.nan

            slice_agg['AvgOpenPrice'] = (
                    (open_trades['TradePrice'] * open_trades['Quantity'].abs()).sum() /
                    open_trades['Quantity'].abs().sum()
            ) if open_trades['Quantity'].abs().sum() > 0 else np.nan

            aggregated_positions = pd.concat([aggregated_positions, slice_agg], ignore_index=True)

    aggregated_positions.rename(columns={'DateTime_clean': 'LastExitDate'}, inplace=True)

    return aggregated_positions


# Aggregate closed positions
aggregated_positions_df = aggregate_closed_positions(aggregated_df)
aggregated_positions_df = aggregated_positions_df.sort_values(by='LastExitDate', ascending=False)

# Trim long text fields
aggregated_positions_df['Symbol'] = aggregated_positions_df['Symbol'].astype(str).apply(
    lambda x: x[:10] + '...' if len(x) > 10 else x)
aggregated_positions_df['Description'] = aggregated_positions_df['Description'].astype(str).apply(
    lambda x: x[:15] + '...' if len(x) > 15 else x)

# Compute trade duration
aggregated_positions_df['TradeDuration'] = aggregated_positions_df['LastExitDate'] - aggregated_positions_df[
    'FirstEntryDate']
aggregated_positions_df['Seconds'] = aggregated_positions_df['TradeDuration'].dt.total_seconds()


def format_duration(duration):
    """Format timedelta into a readable string."""
    if pd.isnull(duration):
        return "N/A"

    days = duration.days
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{days} days {hours:02}:{minutes:02}:{seconds:02}"


aggregated_positions_df['TradeDuration'] = aggregated_positions_df['TradeDuration'].apply(format_duration)
aggregated_positions_df.to_csv("Q_Pareto_Transaction_History/Data/aggregated_transaction_history.csv")
