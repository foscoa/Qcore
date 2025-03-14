import dash
from dash import dcc, html, dash_table, Input, Output, State
import pandas as pd
import plotly.express as px
import os
import numpy as np
from dash.dash_table.Format import Format, Scheme, Sign, Group
import isodate

asst_path = os.path.join(os.getcwd(), "\\Q_Pareto_Transaction_History\\app\\assets\\images\\")

# Initialize app
app = dash.Dash(__name__, suppress_callback_exceptions=True, assets_folder=asst_path)
app.title = "Trading Tracker"

# Sample DataFrames
def get_sample_data():
    return pd.DataFrame({
        "Date": ["2025-03-10", "2025-03-11"],
        "Asset": ["AAPL", "TSLA"],
        "Entry Price": [150, 700],
        "Current Price": [155, 680],
        "Risk %": [2, 3],
        "Status": ["Open", "Open"]
    })

def get_journal_data():
    # Define the file path
    file_path = "Q_Pareto_Transaction_History/Data/U15721173_TradeHistory_03132025.csv"

    # Read the CSV file
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.replace("/", "_", regex=False)

    master_df = df.copy().query('LevelOfDetail == "EXECUTION" & (Open_CloseIndicator == "C" or Open_CloseIndicator == "O")')

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
    master_df['NotionaltoBase'] = master_df.Quantity.abs() * master_df.Multiplier * master_df.FXRateToBase * master_df.TradePrice

    # Direction
    master_df['Position'] = master_df.Buy_Sell.map({'BUY': 'SHORT', 'SELL': 'LONG'})

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

    aggregated_df = master_df.copy().groupby(['IBOrderID']).agg(agg_dict_IBOrderID).reset_index().sort_values(by='DateTime_clean',
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


    def aggregator(df_toBeGrouped):

        df_aggregated = pd.DataFrame(columns=np.append(df_toBeGrouped.columns.values,
                                                       ['first_entry_date', 'AvgClosePrice', 'AvgOpenPrice']))

        for conid in df_toBeGrouped.Conid.unique():

            df_toBeGrouped_conid = df_toBeGrouped.query('Conid == @conid')
            # print(df_toBeGrouped_conid)

            if len(df_toBeGrouped_conid.Open_CloseIndicator.unique()) > 1:
                print(df_toBeGrouped_conid.Symbol)
                idx_divisors = np.where(df_toBeGrouped_conid.Open_CloseIndicator == 'O')[0]
                idx_divisors = np.append(idx_divisors, len(df_toBeGrouped_conid))

                for i in range(len(idx_divisors)-1):
                    slice = df_toBeGrouped_conid.iloc[int(idx_divisors[i]):int(idx_divisors[i+1]),]
                    slice_close = slice.copy().query('Open_CloseIndicator == "C"')
                    slice_open = slice.copy().query('Open_CloseIndicator == "O"')

                    if len(slice_close) == 1: # no grouping needed
                        slice_aggr = slice_close
                    else:
                        agg_dict_slice = {
                            # **String Columns**: Use appropriate aggregation
                            'DateTime_clean': 'last',
                            # Concatenate unique values as a string
                            'Symbol': 'first',  # Keep the first unique symbol
                            'Description': 'first',  # Keep the first unique description
                            'Name': 'first',  # Keep the first unique name
                            'Open_CloseIndicator': 'first',  # Keep the first unique Open/Close indicator
                            'IBOrderID': lambda x: ', '.join(x.astype(str).unique()),  # Concatenate unique IBOrderIDs
                            'Conid': 'first',  # Keep the first Conid
                            'Exchange': 'first',  # Keep the first exchange
                            'Asset Class': 'first',  # Keep the first unique Asset Class
                            'AssetClass': 'first',  # Keep the first unique Asset Class
                            'CurrencyPrimary': 'first',  # Keep the first unique CurrencyPrimary
                            'FXRateToBase': 'last',
                            'Position': 'first',
                            # Get the last FX rate to base (could be 'first' depending on your use case)

                            # **Numerical Columns**: Use appropriate aggregation
                            'Quantity': 'sum',  # Sum of the quantities
                            'TradePrice': lambda x: ', '.join(x.astype(str).unique()),
                            # Weighted average TradePrice
                            'NotionaltoBase': 'sum',  # Sum of Notional to Base
                            'FifoPnlRealizedToBase': 'sum',  # Sum of Fifo PnL Realized to Base
                            'Multiplier': 'first',  # Keep the first unique Multiplier (assuming it's constant for the group)
                        }
                        # Apply the aggregation
                        slice_aggr = slice_close.copy().groupby('Open_CloseIndicator').agg(agg_dict_slice)

                    # create new columns
                    slice_aggr['first_entry_date'] = slice_open.DateTime_clean.values[0]
                    #print(slice_aggr['DateTime_clean'])
                    slice_aggr['AvgClosePrice'] = (slice_close.TradePrice * slice_close.Quantity.abs()).sum()/slice_close.Quantity.abs().sum()
                    slice_aggr['AvgOpenPrice'] = (slice_open.TradePrice * slice_open.Quantity.abs()).sum()/slice_open.Quantity.abs().sum()

                    df_aggregated = pd.concat([df_aggregated, slice_aggr], ignore_index=True)

        df_aggregated = df_aggregated.rename(columns={'DateTime_clean': 'last_exit_date'})

        return df_aggregated

    aggr2_df = aggregator(filter_df)

    #



    # Function to trim text to 11 characters and add "..."
    aggr2_df.loc[:, 'Symbol'] = aggr2_df['Symbol'].apply(lambda x: x[:10] + '...' if len(x) > 10 else x)
    aggr2_df.loc[:, 'Description'] = aggr2_df['Description'].apply(lambda x: x[:15] + '...' if len(x) > 15 else x)

    aggr2_df['Trade Duration'] = aggr2_df.last_exit_date - aggr2_df.first_entry_date
    aggr2_df['Seconds'] = aggr2_df['Trade Duration'].dt.total_seconds()

    # Rename columns
    aggr2_df = aggr2_df.rename(columns={   'CurrencyPrimary': 'CCY',
                                           'FXRateToBase':'FX',
                                           'AssetClass':'Instr.',
                                           'TradePrice': 'Trade Price',
                                           'FifoPnlRealizedToBase': 'Real.PnL(EUR)',
                                           'NotionaltoBase': 'Notional(EUR)',
                                           'first_entry_date': 'First Entry Date',
                                           'last_exit_date': 'Last Exit Date',
                                           'AvgClosePrice': 'wAvg Close Price',
                                           'AvgOpenPrice': 'wAvg Open Price'})

    aggr2_df = aggr2_df[[
            "Last Exit Date",
            "First Entry Date",
            "Trade Duration",
            "Name",
            "Description",
            "Asset Class",
            "CCY",
            "FX",
            "Position",
            "Quantity",
            "Notional(EUR)",
            "Real.PnL(EUR)",
            "wAvg Open Price",
            "wAvg Close Price",
            # "Open_CloseIndicator",
            "Multiplier",
            "Instr.",
            "Symbol",
            "Exchange",
            # "IBOrderID",
            "Seconds",
            "Conid"
        ]]

    def format_duration(duration):
        if isinstance(duration, pd.Timedelta):  # If already a timedelta, format directly
            days = duration.days
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
        elif isinstance(duration, str):  # If ISO 8601 string, parse it first
            duration = isodate.parse_duration(duration)
            days = duration.days
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
        else:
            return "Invalid Duration"

        return f"{days} days {hours:02}:{minutes:02}:{seconds:02}"

    aggr2_df['Trade Duration'] = aggr2_df['Trade Duration'].apply(format_duration)

    aggr2_df = aggr2_df.sort_values(by="Last Exit Date", ascending=False)

    return aggr2_df

def get_statistics():
    return {"Win Rate": "60%", "Avg Profit": "$20", "Avg Loss": "$10"}

# Layout
app.layout = html.Div([
    html.Div([
        # TODO: add QCORE logo
        # html.Img(src="file_example_PNG_500kB.png",
        #          style={"width": "150px", "margin": "auto", "display": "block"}
        # )
    ]),
    dcc.Tabs(id='tabs', value='open_risks', children=[
        dcc.Tab(label='Open Risks', value='open_risks', style={"backgroundColor": "#ecf0f1", "padding": "10px"}),
        dcc.Tab(label='Trading Journal', value='journal', style={"backgroundColor": "#ecf0f1", "padding": "10px"}),
        dcc.Tab(label='Statistics', value='stats', style={"backgroundColor": "#ecf0f1", "padding": "10px"})
    ], colors={"border": "#3498db", "primary": "#2980b9", "background": "#ecf0f1"}),
    html.Div(id='tabs-content', style={"padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "10px"})
], style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#ffffff", "padding": "20px", "borderRadius": "10px"})

# Callbacks
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'open_risks':
        df = get_sample_data()
        return html.Div([
            dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={'overflowX': 'auto', "border": "1px solid #ddd"},
                style_header={'backgroundColor': "rgb(18,54,90)", 'color': 'white', 'fontWeight': 'bold'},
                style_data={'backgroundColor': '#ecf0f1', 'color': '#2c3e50'},
                # Conditionally style the 'pnl' column
                style_data_conditional=[
                {
                    'if': {'filter_query': '{Real.PnL(EUR)} < 0', 'column_id': 'Real.PnL(EUR)'},
                    'backgroundColor': 'red',
                    'color': 'white',
                },
                {
                    'if': {'filter_query': '{Real.PnL(EUR)} >= 0', 'column_id': 'Real.PnL(EUR)'},
                    'backgroundColor': 'green',
                    'color': 'white',
                },
                ]
            )
        ])
    elif tab == 'journal':
        df = get_journal_data()
        # Dynamically generate columns, but modify the Date column to include format
        columns = [{"name": i, "id": i} for i in df.columns]

        # Format only the 'Date' column
        for col in columns:
            if col["id"] == "Trade Date":
                col["type"] = "datetime"
                col["format"] = {"specifier": "%Y-%m-%d %H:%M:%S"}  # or any format you want

            elif col["id"] == 'Real.PnL(EUR)':
                col["type"] = "numeric"
                col["format"] = Format(sign=Sign.positive,precision=2, scheme=Scheme.decimal_integer,
                                       group_delimiter="'", group=Group.yes, groups=[3])
            elif col["id"] == 'wAvg Open Price':
                col["type"] = "numeric"
                col["format"] = dict(specifier='.4~f')

            elif col["id"] == 'wAvg Close Price':
                col["type"] = "numeric"
                col["format"] = dict(specifier='.4~f')

            elif col["id"] == 'FX':
                col["type"] = "numeric"
                col["format"] = dict(specifier='.4~f')

            elif col["id"] == 'Notional(EUR)':
                col["type"] = "numeric"
                col["format"] = Format(precision=2, scheme=Scheme.decimal_integer,
                                       group_delimiter="'", group=Group.yes, groups=[3])

        return html.Div([
            dash_table.DataTable(
                data=df.to_dict('records'),
                columns=columns,
                sort_action="native",
                style_table={'overflowX': 'auto', "border": "1px solid #ddd"},
                style_header={'backgroundColor': "rgb(18,54,90)", 'color': 'white', 'fontWeight': 'bold'},
                style_data={'backgroundColor': '#ecf0f1', 'color': '#2c3e50'},
                style_data_conditional=[
                {
                    'if': {'filter_query': '{Real.PnL(EUR)} < 0', 'column_id': 'Real.PnL(EUR)'},
                    'backgroundColor': '#F7B7B7',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{Real.PnL(EUR)} >= 0', 'column_id': 'Real.PnL(EUR)'},
                    'backgroundColor': '#A8E6A1',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{Position} = "LONG"', 'column_id': 'Position'},
                    'color': 'forestgreen',
                },
                {
                    'if': {'filter_query': '{Position} = "SHORT"', 'column_id': 'Position'},
                    'color': 'firebrick',
                },
                # ASSET CLASS CONDITIONAL FORMATTING
                {
                    'if': {'filter_query': '{Asset Class} = "Rates"', 'column_id': 'Asset Class'},
                    'backgroundColor': 'rgb(250, 230, 160)',  # Pastel Yellow
                },
                {
                    'if': {'filter_query': '{Asset Class} = "Forex"', 'column_id': 'Asset Class'},
                    'backgroundColor': 'rgb(128, 177, 240)',  # Softer Sky Blue
                },
                {
                    'if': {'filter_query': '{Asset Class} = "Eqty Idx"', 'column_id': 'Asset Class'},
                    'backgroundColor': 'rgb(102, 242, 204)',  # Minty Green
                },
                {
                    'if': {'filter_query': '{Asset Class} = "Equity"', 'column_id': 'Asset Class'},
                    'backgroundColor': 'rgb(198, 255, 240)',  # Pale Aqua
                },
                {
                    'if': {'filter_query': '{Asset Class} = "Comdty"', 'column_id': 'Asset Class'},
                    'backgroundColor': 'rgb(255, 178, 150)',  # Soft Coral
                },
                {
                    'if': {'filter_query': '{Asset Class} = "Other"', 'column_id': 'Asset Class'},
                    'backgroundColor': 'rgb(200, 180, 255)',  # Lavender
                }
                ]
            )
        ])
    elif tab == 'stats':
        stats = get_statistics()
        df_stats = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
        return html.Div([
            dash_table.DataTable(
                data=df_stats.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df_stats.columns],
                style_table={'overflowX': 'auto', "border": "1px solid #ddd"},
                style_header={'backgroundColor': "rgb(18,54,90)", 'color': 'white', 'fontWeight': 'bold'},
                style_data={'backgroundColor': '#ecf0f1', 'color': '#2c3e50'}
            ),
            dcc.Graph(
                figure=px.line(
                    pd.DataFrame({"Date": ["2025-03-05", "2025-03-06", "2025-03-07"], "Equity": [1000, 1020, 1050]}),
                    x="Date", y="Equity", title="Equity Curve",
                    line_shape='spline', template='plotly_dark'
                )
            )
        ])

# Run server
if __name__ == '__main__':
    app.run_server(debug=False, port=8051)
