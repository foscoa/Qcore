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
import isodate
from ib_insync import *
from pandas.core.groupby.base import transform_kernel_allowlist


# Define the file path
file_path = "Q_Pareto_Transaction_History_PROD/Data/U15721173_TradeHistory_04162025.csv"


asst_path = os.path.dirname(os.path.abspath(__name__)) + '/Q_Pareto_Transaction_History_DEV/assets/'
# asst_path = "/Users/foscoantognini/Documents/QCore/Qcore/Q_Pareto_Transaction_History_DEV/assets/"
# Initialize app
app = dash.Dash(__name__, suppress_callback_exceptions=True, assets_folder=asst_path)
app.title = "Q - PT Trading Overview"


# Sample DataFrames
def get_sample_data():
    return pd.read_csv("Q_Pareto_Transaction_History_PROD/Data/open_risks.csv", index_col=0)

sample_data = get_sample_data()
# summary table:
def get_summary_table():
    df = get_sample_data()

    summary_df = pd.DataFrame(columns=[
        'Name',
        ''
    ])

def get_number_postions(df):
    nr_positions = df["Status"].value_counts().to_dict()
    if 'working' not in nr_positions.keys():
        nr_positions['working'] = 0

    if 'open' not in nr_positions.keys():
        nr_positions['open'] = 0
    return nr_positions

def get_corr_matrix():
    return pd.read_csv("Q_Pareto_Transaction_History_PROD/Data/corr_matrix.csv", index_col=0)


def get_journal_data(file_path):

    aggregated_positions_df = pd.read_csv("Q_Pareto_Transaction_History_PROD/Data/aggregated_transaction_history.csv",
                                             index_col=0)

    manual_entries = pd.read_excel(
        'Q_Pareto_Transaction_History_PROD/Data/aggregated_transaction_history_manual_entries.xls', engine='xlrd')

    # get rid of mone market trades
    order_IDs = list(manual_entries[manual_entries.Scope != 'money market'].IBOrderID)
    aggregated_positions_df = aggregated_positions_df.query('IBOrderID in @order_IDs')

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
        'AvgClosePrice': 'Close Price (wAvg)',
        'AvgOpenPrice': 'Open Price (wAvg)',
        'TradeDuration': 'Trade Duration',
        'AssetClass': 'Asset Class',


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
        "Open Price (wAvg)",
        "Close Price (wAvg)",
        # "Open_CloseIndicator",
        "Multiplier",
        "Instr.",
        "Symbol",
        "Exchange",
        # "IBOrderID",
        "Seconds",
        "Conid"
    ]]

    # Return the cleaned and aggregated DataFrame
    return aggregated_positions_df.sort_values(by="Last Exit Date", ascending=False)

def plot_return_distribution(trans_hist):
    # Plot
    # Set fixed bin width to 1.5k EUR
    bin_width = 1500

    # Calculate bin range
    x_min = trans_hist['Real.PnL(EUR)'].min()
    x_max = trans_hist['Real.PnL(EUR)'].max()

    start = bin_width * np.floor(x_min / bin_width)
    end = bin_width * np.ceil(x_max / bin_width)

    # Split data by sign
    neg_values = trans_hist[trans_hist['Real.PnL(EUR)'] < 0]['Real.PnL(EUR)']
    pos_values = trans_hist[trans_hist['Real.PnL(EUR)'] >= 0]['Real.PnL(EUR)']

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
    kde = gaussian_kde(trans_hist['Real.PnL(EUR)'])
    x_vals = np.linspace(start, end, 500)
    y_vals = kde(x_vals)

    # Scale KDE to match histogram height (rough approximation)
    hist_counts, _ = np.histogram(trans_hist['Real.PnL(EUR)'], bins=np.arange(start, end + bin_width, bin_width))
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
            text="PnL Distribution (EUR)",
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


def get_statistics():

    # transaction history
    trans_hist = get_journal_data(file_path)


    win_rate = (trans_hist['Real.PnL(EUR)'] > 0).sum()/(trans_hist['Real.PnL(EUR)'] > 0).count()

    avg_profit = trans_hist[(trans_hist['Real.PnL(EUR)'] > 0)]['Real.PnL(EUR)'].mean()

    avg_loss = trans_hist[(trans_hist['Real.PnL(EUR)'] <= 0)]['Real.PnL(EUR)'].mean()

    # all trades
    avg_duration_days = trans_hist.Seconds.mean()/(60*60*24)
    avg_duration_days_str = str(floor(avg_duration_days)) + "d" + \
                            str(int((avg_duration_days - floor(avg_duration_days))*24)) + "h"

    # avg duration profits
    avg_duration_days_P = trans_hist[(trans_hist['Real.PnL(EUR)'] > 0)].Seconds.mean() / (60 * 60 * 24)
    avg_duration_days_str_P = str(floor(avg_duration_days_P)) + "d" + \
                            str(int((avg_duration_days_P - floor(avg_duration_days_P)) * 24)) + "h"

    # avg duration losses
    avg_duration_days_L = trans_hist[(trans_hist['Real.PnL(EUR)'] <= 0)].Seconds.mean() / (60 * 60 * 24)
    avg_duration_days_str_L = str(floor(avg_duration_days_L)) + "d" + \
                            str(int((avg_duration_days_L - floor(avg_duration_days_L)) * 24)) + "h"

    # % Profits Top 15%
    n_top = int(len(trans_hist['Real.PnL(EUR)']) * 0.15)
    top_15_percent = trans_hist['Real.PnL(EUR)'].nlargest(n_top).sum()/trans_hist['Real.PnL(EUR)'][(trans_hist['Real.PnL(EUR)'] > 0)].sum()

    return {"Number of Trades": str((trans_hist['Real.PnL(EUR)'] != 0).count()),
            "Win Rate": str(round(win_rate*100,2)) + "%",
            "Avg Profit (EUR)": f"{int(avg_profit):,}".replace(",", "'"),
            "Avg Loss (EUR)": f"{int(avg_loss):,}".replace(",", "'"),
            "Ratio win/loss size": str(round(avg_profit/abs(avg_loss),2)),
            "% Profits Top 15%": str(round(top_15_percent*100,2)) + "%",
            "Pareto Ratio": str(round(top_15_percent/0.85,2)),
            "Avg Trade Duration": avg_duration_days_str,
            "Avg Trade Duration (Profits)": avg_duration_days_str_P,
            "Avg Trade Duration (Losses)": avg_duration_days_str_L}

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
        df = df.sort_values(by='Status', key=lambda x: x.map({'open':1, 'working':2}))

        # Dynamically generate columns, but modify the Date column to include format
        columns = [{"name": i, "id": i} for i in df.columns]

        # Format only the 'Date' column
        for col in columns:
            if (col["id"] == 'FX') | (col["id"] == 'Last Price'):
                col["type"] = "numeric"
                col["format"] = dict(specifier='.4~f')
            elif ((col["id"] == 'Risk (EUR)') | (col["id"] == 'Exposure (EUR)') |
                  (col["id"] == 'UnRlzdPnL(EUR)') | (col["id"] == 'Rlzd PnL (EUR)') |
                  (col["id"] == 'Tot PnL (EUR)')):
                col["type"] = "numeric"
                col["format"] = Format(precision=2, scheme=Scheme.decimal_integer,
                                       group_delimiter="'", group=Group.yes, groups=[3])
            elif ((col["id"] == 'Risk (bps)') | (col["id"] == 'Rlzd PnL (bps)') |
                  (col["id"] == 'UnRlzdPnL(bps)') | (col["id"] == 'Tot PnL (bps)')):
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



        # Compute Correlation Matrix
        corr_matrix = get_corr_matrix().round(2)

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
            font_colors = ["black"]  # Set annotation text color to black
        )

        # Improve Layout
        fig.update_layout(
            xaxis=dict(side="bottom"),
            width=480,
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#f8f9fa",  # Background color of the entire figure
            plot_bgcolor="#f8f9fa"  # Background color of the plot area
        )

        return html.Div([

            html.Div([
                html.H3("Fund Exposure", style={
                    "marginBottom": "10px",
                    "color": "#12365a",
                    "fontSize": "20px"
                }),

                html.Table([
                    html.Tr([
                        html.Td("Open Positions:", style={"fontWeight": "bold", "paddingRight": "10px"}),
                        html.Td(str(get_number_postions(sample_data)['open']))
                    ]),
                    html.Tr([
                        html.Td("Working Positions:", style={"fontWeight": "bold", "paddingRight": "10px"}),
                        html.Td(str(get_number_postions(sample_data)['working']))
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

            html.Br(),
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
                    'if': {'filter_query': '{Tot PnL (EUR)} < 0', 'column_id': 'Tot PnL (EUR)'},
                    'backgroundColor': '#F7B7B7',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{Tot PnL (EUR)} > 0', 'column_id': 'Tot PnL (EUR)'},
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
            ),
            html.H4("Jan-F // Feb-G // Mar-H  // Apr-J  // May-K  // Jun-M  // Jul-N  // Aug-Q  // Sep-U  // Oct-V  // Nov-X  // Dec-Z",
                    style={"color": "darkgray"}),
            html.Br(),
            # html.H3("Correlation Matrix Heatmap"),
            #dcc.Graph(figure=fig)  # Render heatmap
        ])
    elif tab == 'journal':
        df = get_journal_data(file_path)
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
            elif col["id"] == 'Open Price (wAvg)':
                col["type"] = "numeric"
                col["format"] = dict(specifier='.4~f')

            elif col["id"] == 'Close Price (wAvg)':
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
        df_stats = pd.DataFrame(list(stats.items()), columns=["Metric", "2025"])
        return html.Div([
            dash_table.DataTable(
                data=df_stats.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df_stats.columns],
                style_table={
                    'overflowX': 'auto',
                    'overflowY': 'auto',
                    'width': '450px',  # Fixed width
                    'border': "1px solid #ddd"
                },
                style_header={'backgroundColor': "rgb(18,54,90)", 'color': 'white', 'fontWeight': 'bold'},
                style_data={'backgroundColor': '#ecf0f1', 'color': '#2c3e50'}
            ),
            html.Br(),
            html.Br(),
            dcc.Graph(figure=plot_return_distribution(get_journal_data(file_path)),
                      style={'width': '900px', 'height': '450px'}),

            html.Br(),
            dcc.Graph(figure=butterfly_PnL_plot(get_journal_data(file_path)),
                      style={'width': '500px', 'height': '300px'})
        ])

# Run server
if __name__ == '__main__':
    app.run(debug=False, port=8051)
