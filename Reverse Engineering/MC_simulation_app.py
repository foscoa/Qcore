import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Initialize the Dash app with Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Monte Carlo Trading Simulation"

# Layout of the app
app.layout = dbc.Container([
    html.H1("Monte Carlo Simulation of Trading Strategy", className="my-4 text-center"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Number of Simulations"),
            dbc.Input(id='num_simulations', type='number', value=100000, min=1000, step=1000),

            dbc.Label("Trades per Simulation", className="mt-3"),
            dbc.Input(id='num_trades', type='number', value=50, min=1, step=1),

            dbc.Label("Win Rate (%)", className="mt-3"),
            dbc.Input(id='win_rate', type='number', value=50, min=0, max=100, step=0.1),

            dbc.Label("Average Win (bps)", className="mt-3"),
            dbc.Input(id='avg_win', type='number', value=80, min=0, step=0.1),

            dbc.Label("Average Loss (bps)", className="mt-3"),
            dbc.Input(id='avg_loss', type='number', value=15, min=0, step=0.1),

            dbc.Label("Target Return (%)", className="mt-3"),
            dbc.Input(id='target_return', type='number', value=10, min=0, step=0.1),
        ], md=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Simulation Summary"),
                dbc.CardBody(dcc.Loading(id="loading-summary", type="default", children=html.Div(id='summary'))),
            ], className="mb-4"),

            dcc.Loading(id="loading-graph", type="default", children=dcc.Graph(id='histogram'))
        ], md=8),
    ])
], fluid=True)

# Callback to update the graph and summary
@app.callback(
    Output('histogram', 'figure'),
    Output('summary', 'children'),
    Input('num_simulations', 'value'),
    Input('num_trades', 'value'),
    Input('win_rate', 'value'),
    Input('avg_win', 'value'),
    Input('avg_loss', 'value'),
    Input('target_return', 'value'),
)
def update_graph(num_simulations, num_trades, win_rate, avg_win, avg_loss, target_return):
    # Handle None inputs with default fallbacks
    num_simulations = int(num_simulations) if num_simulations is not None else 100000
    num_trades = int(num_trades) if num_trades is not None else 50
    win_rate = float(win_rate) if win_rate is not None else 50
    avg_win = float(avg_win) if avg_win is not None else 80
    avg_loss = float(avg_loss) if avg_loss is not None else 15
    target_return = float(target_return) if target_return is not None else 10

    # Scale inputs
    win_rate /= 100
    target_return_pct = target_return / 100
    avg_win /= 10000
    avg_loss /= 10000

    # Run simulation
    final_returns = []
    for _ in range(num_simulations):
        outcomes = np.random.rand(num_trades) < win_rate
        profits = outcomes * avg_win - (~outcomes) * avg_loss
        total_return = profits.sum()
        final_returns.append(total_return)

    final_returns = np.array(final_returns)
    prob_meeting_target = np.mean(final_returns >= target_return_pct)
    mean_return = final_returns.mean()
    std_return = final_returns.std()

    # Summary
    summary = [
        html.P(f"\ud83d\udd1d Probability of achieving ≥ {target_return}%: {prob_meeting_target * 100:.2f}%"),
        html.P(f"\ud83d\udcc8 Mean return: {mean_return * 100:.2f}%"),
        html.P(f"\ud83d\udd22 Standard deviation of returns: {std_return * 100:.2f}%")
    ]

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=final_returns * 100,
        nbinsx=50,
        marker=dict(color='lightblue', line=dict(color='black', width=1)),
        name='Simulated Returns'
    ))

    fig.add_shape(
        type="line",
        x0=target_return, x1=target_return,
        y0=0, y1=max(np.histogram(final_returns * 100, bins=50)[0]),
        line=dict(color="red", width=2, dash="dash")
    )

    fig.update_layout(
        title=dict(text="Distribution of Simulated Returns", x=0.5),
        xaxis_title="Total Return (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        height=500
    )

    return fig, summary

if __name__ == '__main__':
    app.run(debug=False, port=8052)
