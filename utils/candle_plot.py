import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_candles(df,
                 title,
                 interval,
                 my_entry,
                 my_exit):

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be datetime")
    if 'volume' not in df.columns:
        raise ValueError("DataFrame must contain a 'volume' column")


    x_labels = df.index.strftime('%Y-%m-%d %H:%M')  # Full timestamp for plotting
    step = max(1, len(x_labels) // 20)

    raw_tickvals = df.index[::step]  # Still a DatetimeIndex

    if 'min' not in interval:
        ticktext = raw_tickvals.strftime('%d %b %y')  # Simplified tick labels (e.g., "May 27")
    else:
        ticktext = x_labels

    tickvals = x_labels[::step]  # Match x_labels for positions

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0,         # no space between plots
        row_heights=[0.85, 0.15],   # top plot takes 85% height
        # Removed subplot_titles to drop titles
    )

    fig.add_trace(go.Candlestick(
        x=x_labels,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='limegreen',
        decreasing_line_color='firebrick',
        name='OHLC'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=x_labels,
        y=df['volume'],
        marker_color='lightblue',
        name='Volume'
    ), row=2, col=1)

    # Add horizontal lines for my_entry and my_exit
    if my_entry is not None:
        fig.add_shape(
            type="line",
            x0=0, x1=1,
            y0=my_entry, y1=my_entry,
            xref='paper', yref='y1',
            line=dict(color="yellow", width=2, dash="dash")
        )

    if my_exit is not None:
        fig.add_shape(
            type="line",
            x0=0, x1=1,
            y0=my_exit, y1=my_exit,
            xref='paper', yref='y1',
            line=dict(color="cyan", width=2, dash="dash")
        )

    # Fill the area between my_entry and my_exit if both are provided
    if my_entry is not None and my_exit is not None:
        fig.add_shape(
            type="rect",
            x0=0, x1=1,
            y0=min(my_entry, my_exit), y1=max(my_entry, my_exit),
            xref='paper', yref='y1',
            fillcolor="rgba(255, 255, 0, 0.2)",  # semi-transparent yellow
            line=dict(width=0),
            layer="below"
        )

    fig.update_layout(
        title=title,
        xaxis=dict(
            type='category',
            rangeslider_visible=False,
            tickvals=tickvals,
            ticktext=ticktext,  # ✅ Show simplified labels
            tickangle=-45,
        ),
        xaxis2=dict(
            type='category',
            tickvals=tickvals,
            ticktext=ticktext,  # ✅ Also for volume subplot
            tickangle=-45,
        ),
        template='plotly_dark',
        hovermode='x unified',
        showlegend=False,
        margin=dict(t=50, b=50, l=70, r=40)  # reduce top margin a bit
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig

