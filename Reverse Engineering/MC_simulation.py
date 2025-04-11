import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Use browser for rendering
pio.renderers.default = 'browser'

# PARAMETERS
num_simulations = 100000
num_trades_per_sim = 50
win_rate = 50               # %
avg_win = 80                # bps
avg_loss = 15               # bps
target_return = 10          # %

# SIMULATION
final_returns = []

# scaling
win_rate /= 100
target_return /= 100
avg_win /= 10000
avg_loss /= 10000

for _ in range(num_simulations):
    outcomes = np.random.rand(num_trades_per_sim) < win_rate
    profits = outcomes * avg_win - (~outcomes) * avg_loss
    total_return = profits.sum()
    final_returns.append(total_return)

final_returns = np.array(final_returns)
prob_meeting_target = np.mean(final_returns >= target_return)
mean_return = final_returns.mean()
std_return = final_returns.std()

# CREATE CUSTOM TITLE
title_text = (
    f"Monte Carlo Simulation of Trading Strategy<br><br>"
    f" Probability of achieving ≥ ${target_return}: {prob_meeting_target * 100:.2f}%<br>"
    f" Mean return: ${mean_return:.2f}<br>"
    f" Std dev of returns: ${std_return:.2f}"
)

# PLOT
fig = go.Figure()

fig.add_trace(go.Histogram(
    x=final_returns,
    nbinsx=50,
    marker=dict(color='skyblue', line=dict(color='black', width=1)),
    name='Simulated Returns'
))

fig.add_shape(
    type="line",
    x0=target_return, x1=target_return,
    y0=0, y1=max(np.histogram(final_returns, bins=50)[0]),
    line=dict(color="red", width=2, dash="dash")
)

fig.update_layout(
    title=dict(text=title_text, x=0.5),
    xaxis_title="Total Return ($)",
    yaxis_title="Frequency",
    showlegend=False
)

fig.show()