import pandas as pd
import plotly.graph_objects as go
import utils.utils
from utils import *
import plotly.io as pio
from ta.volatility import BollingerBands
import numpy as np

# Force Plotly to open plots in your web browser
pio.renderers.default = "browser"


def add_atr(df, period=30):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f'ATR_{period}'] = tr.rolling(window=period).mean()

    return df



CONTRACT_ID = 597391689  # Example: SPY

# === Step 1: Load OHLC Data ===
df = utils.IBKR_download_OHLC(CONTRACT_ID=CONTRACT_ID)

# === Step 2: Calculate Support and Resistance ===
window = 8*5
df['resistance'] = df['high'].rolling(window).max()
df['support'] = df['low'].rolling(window).min()

# === Step 3: Detect Breakouts ===
df['breakout_up'] = (df['high'] > df['resistance'].shift(1))
df['breakout_down'] = (df['low'] < df['support'].shift(1))

# BB bands
indicator_bb = BollingerBands(close=df["close"], window=20, window_dev=2)
df['bb_bbm'] = indicator_bb.bollinger_mavg()
df['bb_bbh'] = indicator_bb.bollinger_hband()
df['bb_bbl'] = indicator_bb.bollinger_lband()

# Add 30-day ATR
df = add_atr(df, period=30)

# === Step 4: Plot with Plotly ===
fig = go.Figure()

# Candlestick chart
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name='OHLC'
))

# Resistance and support
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['resistance'],
    mode='lines',
    line=dict(color='orange', dash='dot'),
    name='Resistance'
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df['support'],
    mode='lines',
    line=dict(color='blue', dash='dot'),
    name='Support'
))

# Breakout Up Markers
fig.add_trace(go.Scatter(
    x=df.index[df['breakout_up']],
    y=df['close'][df['breakout_up']],
    mode='markers',
    marker=dict(color='green', size=10, symbol='triangle-up'),
    name='Breakout Up'
))

# Breakout Down Markers
fig.add_trace(go.Scatter(
    x=df.index[df['breakout_down']],
    y=df['close'][df['breakout_down']],
    mode='markers',
    marker=dict(color='red', size=10, symbol='triangle-down'),
    name='Breakout Down'
))

# Layout
fig.update_layout(
    title='Breakout Pattern Detection on OHLC Chart',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False,
    template='plotly_dark',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)

fig.show()