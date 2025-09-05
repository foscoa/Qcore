import dash
from dash import dcc, html, dash_table, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import os
import numpy as np
from dash.dash_table.Format import Format, Scheme, Sign, Group
from scipy.stats import gaussian_kde
from math import floor
import plotly.graph_objects as go

# Define the file pathes
file_path_transaction_history = "Q_Pareto_Transaction_History_DEV/Data/aggregated_transaction_history.csv"
file_path_transaction_history_AMC = 'Q_Pareto_Transaction_History_DEV/Data/AMC_transaction_history/aggregated_transaction_history_AMC.csv'
file_path_NAV = 'Q_Pareto_Transaction_History_DEV/Data/NAV_in_base_FA.csv'
file_path_open_risks = "Q_Pareto_Transaction_History_DEV/Data/open_risks.csv"
asst_path = os.path.dirname(os.path.abspath(__name__)) + '/Q_Pareto_Transaction_History_DEV/assets/'
file_path_corr_matrix= "Q_Pareto_Transaction_History_DEV/Data/corr_matrix.csv"
file_path_corr_matrix_ass= "Q_Pareto_Transaction_History_DEV/Data/corr_matrix_ass.csv"
file_path_manual_entries = 'Q_Pareto_Transaction_History_DEV/Data/aggregated_transaction_history_manual_entries.xls'
file_path_2022_stats = 'Q_Pareto_Transaction_History_DEV/Data/AMC_transaction_history/2022_stats/2022_stats.csv'

# Initialize app
app = dash.Dash(__name__, suppress_callback_exceptions=True, assets_folder=asst_path)
app.title = "Q - PT Trading App"


# Sample DataFrames
def get_sample_data(file_path):
    return pd.read_csv(file_path, index_col=0)

def get_journal_data(file_path_transaction_history):

    aggregated_positions_df = pd.read_csv(file_path_transaction_history,
                                             index_col=0)

    manual_entries = pd.read_excel(file_path_manual_entries
        , engine='xlrd')

    # get rid of money market trades
    order_IDs = list(manual_entries[manual_entries.Scope == 'money market'].IBOrderID)
    aggregated_positions_df = aggregated_positions_df.query('IBOrderID not in @order_IDs')

    # load AMC data
    aggregated_positions_df_AMC = pd.read_csv(file_path_transaction_history_AMC, index_col=0)

    # remove DAX money market trades
    mm_AMC = ['DAX 15DEC23', 'DAX 15MAR24', 'DAX 20DEC24']
    aggregated_positions_df_AMC =aggregated_positions_df_AMC.query('Description not in @mm_AMC')

    # merge with AMC data
    aggregated_positions_df = pd.concat([aggregated_positions_df, aggregated_positions_df_AMC], axis=0, ignore_index=True)

    # Read transaction history CSV file
    df_nav = pd.read_csv(file_path_NAV)
    df_nav.columns = df_nav.columns.str.replace("/", "_", regex=False)

    # add PnL in basis points
    df_nav["ReportDate"] = pd.to_datetime(df_nav["ReportDate"], format="%Y%m%d")

    # Truncate time from FirstEntryDate to keep only the date part
    aggregated_positions_df["FirstEntryDate"] = pd.to_datetime(aggregated_positions_df["FirstEntryDate"],
                                                               errors='coerce')

    aggregated_positions_df["EntryDate_D"] = aggregated_positions_df["FirstEntryDate"].dt.floor("D")

    # Step 2: Merge the dataframes on the date
    aggregated_positions_df = aggregated_positions_df.merge(
        df_nav[["ReportDate", "Total"]],
        left_on="EntryDate_D",
        right_on="ReportDate",
        how="left"
    )

    aggregated_positions_df['FifoPnlRealizedToBaseBps'] = (aggregated_positions_df.FifoPnlRealizedToBase / aggregated_positions_df.Total) * 10000

    aggregated_positions_df['Change(%)'] = ((aggregated_positions_df.AvgClosePrice/aggregated_positions_df.AvgOpenPrice)-1)*100

    # Rename columns for clarity
    aggregated_positions_df.rename(columns={
        'CurrencyPrimary': 'CCY',
        'FXRateToBase': 'FX',
        'AssetClass': 'Instr.',
        'TradePrice': 'Trade Price',
        'FifoPnlRealizedToBase': 'Real.PnL(EUR)',
        'NotionalToBase': 'Notional(EUR)',
        'FirstEntryDate': 'First Entry Date',
        'LastExitDate': 'Last Exit Date',
        'AvgClosePrice': 'ExitPrice(wAvg)',
        'AvgOpenPrice': 'EntryPrice(wAvg)',
        'TradeDuration': 'Trade Duration',
        'AssetClass': 'Asset Class',
        'Total': 'NAV at Entry',
        'FifoPnlRealizedToBaseBps': 'Real.PnL(bps)',
        'Multiplier': 'Mult.',


    }, inplace=True)

    # reorder the columns
    aggregated_positions_df = aggregated_positions_df[[
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
        "Real.PnL(bps)",
        "EntryPrice(wAvg)",
        "ExitPrice(wAvg)",
        "Change(%)",
        # "Open_CloseIndicator",
        "Mult.",
        "Instr.",
        "Symbol",
        "Exchange",
        "NAV at Entry",
        # "IBOrderID",
        "Seconds",
        "Conid"
    ]]

    # Return the cleaned and aggregated DataFrame
    return aggregated_positions_df.sort_values(by="Last Exit Date", ascending=False)

def perf_attribution(file_path_transaction_history):

    aggregated_positions_df = pd.read_csv(file_path_transaction_history,
                                             index_col=0)

    manual_entries = pd.read_excel(file_path_manual_entries
        , engine='xlrd')

    trader = 'RR'

    aggregated_positions_df['Trader'] = aggregated_positions_df.IBOrderID.map(
        manual_entries[['IBOrderID', 'Trader']].set_index('IBOrderID').Trader.to_dict())

    aggregated_positions_df = aggregated_positions_df.query('IBOrderID in @order_IDs')

    # Read transaction history CSV file
    df_nav = pd.read_csv(file_path_NAV)
    df_nav.columns = df_nav.columns.str.replace("/", "_", regex=False)

    # add PnL in basis points
    df_nav["ReportDate"] = pd.to_datetime(df_nav["ReportDate"], format="%Y%m%d")

    # Truncate time from FirstEntryDate to keep only the date part
    aggregated_positions_df["FirstEntryDate"] = pd.to_datetime(aggregated_positions_df["FirstEntryDate"],
                                                               errors='coerce')

    aggregated_positions_df["EntryDate_D"] = aggregated_positions_df["FirstEntryDate"].dt.floor("D")

    # Step 2: Merge the dataframes on the date
    aggregated_positions_df = aggregated_positions_df.merge(
        df_nav[["ReportDate", "Total"]],
        left_on="EntryDate_D",
        right_on="ReportDate",
        how="left"
    )

    aggregated_positions_df['FifoPnlRealizedToBaseBps'] = (
                                                                      aggregated_positions_df.FifoPnlRealizedToBase / aggregated_positions_df.Total) * 10000

    aggregated_positions_df['Change(%)'] = ((
                                                        aggregated_positions_df.AvgClosePrice / aggregated_positions_df.AvgOpenPrice) - 1) * 100

    # Rename columns for clarity
    aggregated_positions_df.rename(columns={
        'CurrencyPrimary': 'CCY',
        'FXRateToBase': 'FX',
        'AssetClass': 'Instr.',
        'TradePrice': 'Trade Price',
        'FifoPnlRealizedToBase': 'Real.PnL(EUR)',
        'NotionalToBase': 'Notional(EUR)',
        'FirstEntryDate': 'First Entry Date',
        'LastExitDate': 'Last Exit Date',
        'AvgClosePrice': 'ExitPrice(wAvg)',
        'AvgOpenPrice': 'EntryPrice(wAvg)',
        'TradeDuration': 'Trade Duration',
        'AssetClass': 'Asset Class',
        'Total': 'NAV at Entry',
        'FifoPnlRealizedToBaseBps': 'Real.PnL(bps)',
        'Multiplier': 'Mult.',

    }, inplace=True)

    # reorder the columns
    aggregated_positions_df = aggregated_positions_df[[
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
        "Real.PnL(bps)",
        "EntryPrice(wAvg)",
        "ExitPrice(wAvg)",
        "Change(%)",
        # "Open_CloseIndicator",
        "Mult.",
        "Instr.",
        "Symbol",
        "Exchange",
        "NAV at Entry",
        # "IBOrderID",
        "Seconds",
        "Conid"
    ]]

    aggregated_positions_df.to_csv("C:\\Users\\FoscoAntognini\\Documents\\a.csv")


sample_data = get_sample_data(file_path_open_risks)

report_time = sample_data['Report Time'].unique()[0]
NLV = sample_data['NLV'].unique()[0]

sample_data.drop(['Report Time', 'NLV', 'Daily Contribution (bps)'], axis=1, inplace=True)
available_years = pd.to_datetime(get_journal_data(file_path_transaction_history)['Last Exit Date']).dt.year.unique()
available_years = np.insert(available_years.astype(object), 0, "Since Inception")

def get_number_positions(df):
    nr_positions = df["Status"].value_counts().to_dict()
    if 'working' not in nr_positions.keys():
        nr_positions['working'] = 0

    if 'open' not in nr_positions.keys():
        nr_positions['open'] = 0
    return nr_positions

def plot_corr_matrix():

    corr_matrix = pd.read_csv(file_path_corr_matrix, index_col=0).round(2).transpose()

    # Create Heatmap
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        colorscale="RdBu_r",  # Use a valid Plotly colorscale
        annotation_text=corr_matrix.values,
        showscale=False,
        zmin=-1,  # Set minimum value of color scale
        zmax=1,  # Set maximum value of color scale
        font_colors=["black"]  # Set annotation text color to black
    )

    # Improve Layout
    fig.update_layout(
        xaxis=dict(side="bottom"),
        # width=480,
        # height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#f8f9fa",  # Background color of the entire figure
        plot_bgcolor="#f8f9fa"  # Background color of the plot area
    )

    return fig

def get_string_corr_ass():

    corr_matrix_ass = pd.read_csv(file_path_corr_matrix_ass, index_col=False)

    return ("Data as of " + corr_matrix_ass.report_time.values[0] + " | Time frame: " + corr_matrix_ass.durationStr.values[0] +
            " | Granularity: " + corr_matrix_ass.barSizeSetting.values[0])

def plot_return_distribution(trans_hist, year_string):
    # Plot
    # Set fixed bin width to 1.5k EUR
    bin_width = 5

    # Calculate bin range
    x_min = trans_hist['Real.PnL(bps)'].min()
    x_max = trans_hist['Real.PnL(bps)'].max()

    start = bin_width * np.floor(x_min / bin_width)
    end = bin_width * np.ceil(x_max / bin_width)

    # Split data by sign
    neg_values = trans_hist[trans_hist['Real.PnL(bps)'] < 0]['Real.PnL(bps)']
    pos_values = trans_hist[trans_hist['Real.PnL(bps)'] >= 0]['Real.PnL(bps)']

    # Create figure
    fig = go.Figure()

    # Histogram: PnL < 0
    fig.add_trace(go.Histogram(
        x=neg_values,
        xbins=dict(start=start, end=end, size=bin_width),
        marker=dict(color='#F7B7B7', line=dict(color='black', width=1)),
        name='PnL < 0',
        opacity=0.75,
        showlegend=False
    ))

    # Histogram: PnL ≥ 0
    fig.add_trace(go.Histogram(
        x=pos_values,
        xbins=dict(start=start, end=end, size=bin_width),
        marker=dict(color='#A8E6A1', line=dict(color='black', width=1)),
        name='PnL ≥ 0',
        opacity=0.75,
        showlegend=False
    )),

    # KDE curve
    kde = gaussian_kde(trans_hist['Real.PnL(bps)'])
    x_vals = np.linspace(start, end, 500)
    y_vals = kde(x_vals)

    # Scale KDE to match histogram height (rough approximation)
    hist_counts, _ = np.histogram(trans_hist['Real.PnL(bps)'], bins=np.arange(start, end + bin_width, bin_width))
    scale_factor = hist_counts.max() / y_vals.max()
    y_scaled = y_vals * scale_factor

    # Add KDE curve
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_scaled,
        mode='lines',
        line=dict(color='black', width=0.5),
        name='Estimated Density',
        showlegend=False
    ))

    # Vertical line at zero
    # ymax = np.histogram(trans_hist['Real.PnL(EUR)'], bins=np.arange(start, end + bin_width, bin_width))[0].max()

    # fig.add_shape(
    #     type="line",
    #     x0=0, x1=0,
    #     y0=0, y1=ymax,
    #     line=dict(color="red", width=2, dash="dash")
    # )

    # Layout settings
    fig.update_layout(
        title=dict(
            text="PnL Distribution (bps) - " + year_string,
            x=0.0,
            font=dict(
                size=15,  # You can adjust the font size
                color="black",  # You can adjust the color
                family="Arial, sans-serif",  # Optional: specify the font family
                weight="bold"  # Makes the title bold
            )
        ),
        barmode='overlay',
        xaxis_title="PnL (bps)",
        yaxis_title="Number of Trades",
        template="plotly_white",
        margin=dict(l=20, r=20, t=25, b=20),
        # height=500,
        paper_bgcolor="#f8f9fa",  # Background color of the entire figure
        plot_bgcolor="#f8f9fa"  # Background color of the plot area
    )

    return fig

def butterfly_PnL_plot(trans_hist):

    # Step 1: Aggregate and sort
    grouped = trans_hist.groupby('Asset Class')['Real.PnL(EUR)'].sum().sort_values(ascending=False)

    # Step 2: Create butterfly chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=grouped.values,
        y=grouped.index,
        orientation='h',
        marker_color=['#A8E6A1' if x >= 0 else '#F7B7B7' for x in grouped.values],
    ))

    fig.update_layout(
        xaxis_title='PnL EUR',
        yaxis=dict(autorange="reversed"),  # Highest PnL at the top
        bargap=0.3,
        showlegend=False
    )

    fig.update_layout(
        title=dict(
            text="Total PnL (EUR) per Asset Class",
            x=0.0,
            font=dict(
                size=15,  # You can adjust the font size
                color="black",  # You can adjust the color
                family="Arial, sans-serif",  # Optional: specify the font family
                weight="bold"  # Makes the title bold
            )
        ),
        barmode='overlay',
        xaxis_title="PnL (EUR)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=25, b=20),
        # height=500,
        paper_bgcolor="#f8f9fa",  # Background color of the entire figure
        plot_bgcolor="#f8f9fa"  # Background color of the plot area
    )

    return fig

    ###

def butterfly_contr_plot(file_path):

    df = get_sample_data(file_path)

    df.dropna(subset=['Daily Contribution (bps)'], inplace=True)

    grouped = df.groupby('Name')['Daily Contribution (bps)'].sum().sort_values(ascending=False)

    # Step 2: Create butterfly chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=grouped.values,
        y=grouped.index,
        orientation='h',
        marker_color=['#A8E6A1' if x >= 0 else '#F7B7B7' for x in grouped.values],
        text=[str(int(x)) for x in grouped.values],  # Rounded integer labels
        textposition='outside',  # Show labels outside the bar ends
        insidetextanchor='start',  # Ensures alignment
        cliponaxis=False  # Allows text to overflow beyond axis
    ))

    fig.update_layout(
        xaxis_title='PnL EUR',
        yaxis=dict(autorange="reversed"),  # Highest PnL at the top
        bargap=0.3,
        showlegend=False
    )

    fig.update_layout(
        barmode='overlay',
        xaxis_title="Contribution (bps)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=25, b=20),
        # height=500,
        paper_bgcolor="#f8f9fa",  # Background color of the entire figure
        plot_bgcolor="#f8f9fa"  # Background color of the plot area
    )

    return fig

def get_statistics():
    # transaction history
    trans_hist = get_journal_data(file_path_transaction_history)

    # Ensure date column is datetime
    trans_hist['Date'] = pd.to_datetime(trans_hist['First Entry Date'])  # Adjust column name if needed

    # ADDING 2022 data
    trans_hist_22 = pd.read_csv(file_path_2022_stats)
    trans_hist_22['Seconds'] = (pd.to_datetime(trans_hist_22['Date Closed'], format='%m/%d/%Y') - \
                                pd.to_datetime(trans_hist_22['Execution Date'], format='%m/%d/%Y'))

    trans_hist_22['Seconds'] = trans_hist_22['Seconds'].dt.total_seconds()

    trans_hist_22['Real.PnL(bps)'] = trans_hist_22['Realised EUR PnL']/trans_hist_22['AUM']*10000
    trans_hist_22['Real.PnL(EUR)'] = trans_hist_22['Realised EUR PnL']

    trans_hist_22['Date'] = pd.to_datetime(trans_hist_22['Date Closed'], format='%m/%d/%Y')

    trans_hist = pd.concat([trans_hist[['Date', 'Real.PnL(bps)', 'Real.PnL(EUR)','Seconds']],
                            trans_hist_22[['Date', 'Real.PnL(bps)', 'Real.PnL(EUR)', 'Seconds']]])


    def get_stats(df):
        if df.empty:
            return {
                "Number of Trades": "0",
                "Win Rate": "0.00%",
                "Avg Profit (EUR)": "0",
                "Avg Profit (bps)": "0",
                "Avg Loss (EUR)": "0",
                "Avg Loss (bps)": "0",
                "Ratio win/loss size": "N/A",
                "% Profits Top 15%": "0.00%",
                "Pareto Ratio": "0.00",
                "Avg Trade Duration": "0d0h",
                "Avg Trade Duration (Profits)": "0d0h",
                "Avg Trade Duration (Losses)": "0d0h"
            }

        win_rate = (df['Real.PnL(EUR)'] > 0).sum() / (df['Real.PnL(EUR)'] != 0).sum()
        avg_profit = df[df['Real.PnL(EUR)'] > 0]['Real.PnL(EUR)'].mean()
        avg_profit_bps = round(df[df['Real.PnL(bps)'] > 0]['Real.PnL(bps)'].mean(), 2)
        avg_loss = np.abs(df[df['Real.PnL(EUR)'] <= 0]['Real.PnL(EUR)'].mean())
        avg_loss_bps = round(np.abs(df[df['Real.PnL(bps)'] <= 0]['Real.PnL(bps)'].mean()), 2)

        avg_duration_days = df['Seconds'].mean() / (60 * 60 * 24)
        avg_duration_days_str = f"{floor(avg_duration_days)}d{int((avg_duration_days - floor(avg_duration_days)) * 24)}h"

        avg_duration_days_P = df[df['Real.PnL(EUR)'] > 0]['Seconds'].mean() / (60 * 60 * 24)
        avg_duration_days_str_P = f"{floor(avg_duration_days_P)}d{int((avg_duration_days_P - floor(avg_duration_days_P)) * 24)}h"

        avg_duration_days_L = df[df['Real.PnL(EUR)'] <= 0]['Seconds'].mean() / (60 * 60 * 24)
        avg_duration_days_str_L = f"{floor(avg_duration_days_L)}d{int((avg_duration_days_L - floor(avg_duration_days_L)) * 24)}h"

        n_top = int(len(df['Real.PnL(EUR)']) * 0.15)
        top_15_percent = df['Real.PnL(EUR)'].nlargest(n_top).sum() / df[df['Real.PnL(EUR)'] > 0]['Real.PnL(EUR)'].sum()

        return {
            "Number of Trades": str((df['Real.PnL(EUR)'] != 0).count()),
            "Win Rate": f"{round(win_rate * 100, 2)}%",
            "Avg Profit (EUR)": f"{int(avg_profit):,}".replace(",", "'") if not pd.isna(avg_profit) else "0",
            "Avg Profit (bps)": str(avg_profit_bps),
            "Avg Loss (EUR)": f"{int(avg_loss):,}".replace(",", "'") if not pd.isna(avg_loss) else "0",
            "Avg Loss (bps)": str(avg_loss_bps),
            "Ratio win/loss size": str(round(avg_profit / abs(avg_loss), 2)) if avg_loss != 0 else "N/A",
            "% Profits Top 15%": f"{round(top_15_percent * 100, 2)}%",
            "Pareto Ratio": str(round(top_15_percent / 0.85, 2)) if top_15_percent != 0 else "0",
            "Avg Trade Duration": avg_duration_days_str,
            "Avg Trade Duration (Profits)": avg_duration_days_str_P,
            "Avg Trade Duration (Losses)": avg_duration_days_str_L
        }

    # Get stats for each period
    stats = {
        "2022": get_stats(trans_hist[trans_hist['Date'].dt.year == 2022]),
        "2023": get_stats(trans_hist[trans_hist['Date'].dt.year == 2023]),
        "2024": get_stats(trans_hist[trans_hist['Date'].dt.year == 2024]),
        "2025": get_stats(trans_hist[trans_hist['Date'].dt.year == 2025]),
        "SI": get_stats(trans_hist),
    }

    return stats


# Layout
app.layout = html.Div([

    # add QCORe logo
    html.Img(
        src="../assets/QCORE_Logo.jpg",
        style={
            "width": "100px",
            "display": "block",  # optional here
            "margin-left": "0",  # no left margin
            "margin-right": "auto"  # push everything else away
        }
    ),
    html.Br(),
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
        df = sample_data

        # sorting, first open orders
        df = df.sort_values(by='maxL/minP (bps)', ascending=False)

        # Dynamically generate columns, but modify the Date column to include format
        columns = [{"name": i, "id": i} for i in df.columns]

        # Format only the 'Date' column
        for col in columns:
            if (col["id"] == 'FX') | (col["id"] == 'Last Price'):
                col["type"] = "numeric"
                col["format"] = dict(specifier='.4~f')
            elif ((col["id"] == 'Risk (EUR)') | (col["id"] == 'Exposure (EUR)') |
                  (col["id"] == 'UnRlzdPnL(EUR)') | (col["id"] == 'Rlzd PnL (EUR)') |
                  (col["id"] == 'Tot PnL (EUR)') | (col["id"] == 'maxL/minP (EUR)')):
                col["type"] = "numeric"
                col["format"] = Format(precision=2, scheme=Scheme.decimal_integer,
                                       group_delimiter="'", group=Group.yes, groups=[3])
            elif ((col["id"] == 'Risk (bps)') | (col["id"] == 'Rlzd PnL (bps)') |
                  (col["id"] == 'UnRlzdPnL(bps)') | (col["id"] == 'Tot PnL (bps)')| (col["id"] == 'maxL/minP (bps)')):
                col["type"] = "numeric"
                col["format"] = dict(specifier='.0~f')

            elif (col["id"] == 'Expos. (%)'):
                col["type"] = "numeric"
                col["format"] = dict(specifier='.2~f')

            elif (col["id"] == 'ATR 30D'):
                col["type"] = "numeric"
                col["format"] = dict(specifier='.3~f')

            elif (col["id"] == 'ATR 30D (%)'):
                col["type"] = "numeric"
                col["format"] = dash_table.FormatTemplate.percentage(2)


        return html.Div([

            # Report time outside the table, at the top
            html.P(f"Report generated on: {report_time}", style={
                "fontSize": "12px",
                "fontStyle": "italic",
                "color": "#7f8c8d",
                "marginBottom": "8px"
            }),

            html.Div([
                html.H3("Fund Exposure", style={
                    "marginBottom": "10px",
                    "color": "#12365a",
                    "fontSize": "20px"
                }),

                html.Table([
                    html.Tr([
                        html.Td("Open Positions:", style={"fontWeight": "bold", "paddingRight": "10px"}),
                        html.Td(str(get_number_positions(sample_data)['open']))
                    ]),
                    html.Tr([
                        html.Td("Working Positions:", style={"fontWeight": "bold", "paddingRight": "10px"}),
                        html.Td(str(get_number_positions(sample_data)['working']))
                    ]),
                    html.Tr([
                        html.Td("Net Liquidation Value (NLV):", style={"fontWeight": "bold", "paddingRight": "10px"}),
                        html.Td("{:,.0f}".format(NLV).replace(",", "'") + " EUR")
                    ]),
                    html.Tr([
                        html.Td("Total Risk (all stops hit):", style={"fontWeight": "bold", "paddingRight": "10px"}),
                        html.Td("{:,.0f}".format(
                            sample_data.loc[df["Status"] == "open", "Risk (EUR)"].sum()).replace(",", "'") +
                                " EUR | " + str(
                            round(sample_data.loc[df["Status"] == "open", "Risk (EUR)"].sum() / NLV * 10000,
                                  2)) + " bps")
                    ]),
                    html.Tr([
                        html.Td("Gross Exposure (Trading):", style={"fontWeight": "bold", "paddingRight": "10px"}),
                        html.Td("{:,.0f}".format(
                            sample_data.loc[df["Status"] == "open", "Exposure (EUR)"].sum()).replace(",", "'") +
                                " EUR | " + str(
                            round(sample_data.loc[df["Status"] == "open", "Exposure (EUR)"].sum() / NLV * 100,
                                  2)) + "%")
                    ]),
                ], style={
                    "fontSize": "12px",
                    "color": "#2c3e50",
                    "backgroundColor": "#ecf0f1",
                    "borderCollapse": "collapse",
                    "width": "320px",  # fixed width
                    "textAlign": "left"
                }),
            ], style={
                "padding": "0px",
                "marginBottom": "20px"
            }),

            html.H3("Position Details",

                    style={
                            "marginBottom": "10px",
                            "color": "#12365a",
                            "fontSize": "20px"
                        }),
            dash_table.DataTable(
                data=df.to_dict('records'),
                columns=columns,
                sort_action="native",
                style_table={'overflowX': 'auto', "border": "1px solid #ddd"},
                style_header={'backgroundColor': "rgb(18,54,90)", 'color': 'white', 'fontWeight': 'bold'},
                style_data={'backgroundColor': '#ecf0f1', 'color': '#2c3e50'},
                # Conditionally style the 'pnl' column
                style_data_conditional=[
                {
                    'if': {'filter_query': '{Status} = "working"', 'column_id': 'Status'},
                    'backgroundColor': 'lightblue',
                },
                {
                    'if': {'filter_query': '{Status} = "open"', 'column_id': 'Status'},
                    'backgroundColor': 'lightgreen',
                },
                {
                    'if': {'filter_query': '{Status} = "closed"', 'column_id': 'Status'},
                    'backgroundColor': '#F7B7B7',
                },
                {
                    'if': {'filter_query': '{Tot PnL (bps)} < 0', 'column_id': 'Tot PnL (bps)'},
                    'backgroundColor': '#F7B7B7',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{Tot PnL (bps)} > 0', 'column_id': 'Tot PnL (bps)'},
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
                {
                    "if": {"column_id": "Risk (bps)"},
                    "fontWeight": "bold"
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
                    'if': {'filter_query': '{Asset Class} = "Crypto"', 'column_id': 'Asset Class'},
                    'backgroundColor': 'rgb(200, 180, 255)',  # Lavender
                }
                ]
            ),
            html.H4("Jan-F // Feb-G // Mar-H  // Apr-J  // May-K  // Jun-M  // Jul-N  // Aug-Q  // Sep-U  // Oct-V  // Nov-X  // Dec-Z",
                    style={"color": "darkgray"}),

            # html.H3("Correlation Matrix Heatmap",  style={
            #         "marginBottom": "10px",
            #         "color": "#12365a",
            #         "fontSize": "20px"
            #     }),
            #
            # html.P(get_string_corr_ass(), style={
            #     "fontSize": "12px",
            #     "fontStyle": "italic",
            #     "color": "#7f8c8d",
            #     "marginBottom": "8px"
            # }),

            html.H3("Daily Contribution per Position",  style={
                    "marginBottom": "10px",
                    "color": "#12365a",
                    "fontSize": "20px"
                }),

            html.P("CASH and money market trades not included",style={
                "fontSize": "12px",
                "fontStyle": "italic",
                "color": "#7f8c8d",
                "marginBottom": "8px"
            }),

            dcc.Graph(figure=butterfly_contr_plot(file_path=file_path_open_risks),  style={'width': '50%'})  # Render heatmap
        ])
    elif tab == 'journal':
        df = get_journal_data(file_path_transaction_history)
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
            elif col["id"] == 'EntryPrice(wAvg)':
                col["type"] = "numeric"
                col["format"] = dict(specifier='.4~f')

            elif col["id"] == 'ExitPrice(wAvg)':
                col["type"] = "numeric"
                col["format"] = dict(specifier='.4~f')

            elif col["id"] == 'FX':
                col["type"] = "numeric"
                col["format"] = dict(specifier='.4~f')

            elif col["id"] == 'Notional(EUR)':
                col["type"] = "numeric"
                col["format"] = Format(precision=2, scheme=Scheme.decimal_integer,
                                       group_delimiter="'", group=Group.yes, groups=[3])
            elif col["id"] == 'NAV at Entry':
                col["type"] = "numeric"
                col["format"] = Format(precision=2, scheme=Scheme.decimal_integer,
                                       group_delimiter="'", group=Group.yes, groups=[3])
            elif col["id"] == 'Real.PnL(bps)':
                col["type"] = "numeric"
                col["format"] = dict(specifier='.0~f')

            elif (col["id"] == 'Change(%)'):
                col["type"] = "numeric"
                col["format"] = dict(specifier='.2~f')


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
                    'if': {'filter_query': '{Real.PnL(bps)} < 0', 'column_id': 'Real.PnL(bps)'},
                    'backgroundColor': '#F7B7B7',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{Real.PnL(bps)} >= 0', 'column_id': 'Real.PnL(bps)'},
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
                {
                    'if': {'filter_query': '{Change(%)} >= 0', 'column_id': 'Change(%)'},
                    'color': 'forestgreen',
                },
                {
                    'if': {'filter_query': '{Change(%)} < 0', 'column_id': 'Change(%)'},
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
                    'if': {'filter_query': '{Asset Class} = "Crypto"', 'column_id': 'Asset Class'},
                    'backgroundColor': 'rgb(200, 180, 255)',  # Lavender
                }
                ]
            )
        ])
    elif tab == 'stats':
        stats = pd.DataFrame(get_statistics())
        df_stats = stats.reset_index().rename(columns={"index": "Metric"})

        return html.Div([
            dash_table.DataTable(
                data=df_stats.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df_stats.columns],
                style_table={
                    'overflowX': 'auto',
                    'overflowY': 'auto',
                    'width': '600px',  # Fixed width
                    'border': "1px solid #ddd"
                },
                style_header={'backgroundColor': "rgb(18,54,90)", 'color': 'white', 'fontWeight': 'bold'},
                style_data={'backgroundColor': '#ecf0f1', 'color': '#2c3e50'}
            ),
            html.Br(),
            html.Br(),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': str(year), 'value': year} for year in available_years],
                value=available_years[0]
            ),
            html.Br(),
            dcc.Graph(figure=plot_return_distribution(get_journal_data(file_path_transaction_history), "Since Inception"),
                      style={'width': '900px', 'height': '450px'},
                      id='pnl_distribution'),

            html.Br(),
            dcc.Graph(figure=butterfly_PnL_plot(get_journal_data(file_path_transaction_history)),
                      style={'width': '500px', 'height': '300px'})
        ])

@app.callback(
    Output('pnl_distribution', 'figure'),
    Input('year-dropdown', 'value')
)
def update_chart(selected_year):

    if selected_year == 'Since Inception':
        filtered_df = get_journal_data(file_path_transaction_history)
    else:
        start = f"{selected_year}-01-01"
        end = f"{int(selected_year) + 1}-01-01"  # Start of next year

        filtered_df = get_journal_data(file_path_transaction_history).query("`Last Exit Date` >= @start and `Last Exit Date` < @end")

    return plot_return_distribution(filtered_df, str(selected_year))

# Run server
if __name__ == '__main__':
    # app.run(debug=False, port=8050)
    app.run(host='0.0.0.0', port=8072, debug=False)
