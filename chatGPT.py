from openai import OpenAI
import yfinance as yf

client = OpenAI(api_key="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")

ticker = "NET"

# Fetch the stock information
stock = yf.Ticker(ticker)

# Get the latest news
news = stock.news

count = 0
# Display the news
while count < min(len(news), 20):
    article = news[count]
    print(f"Title: {article['title']} ")
    print(f"Link: {article['link']}")

    url = article['title']

    promt =  (
        f"Forget all your previous instructions. Pretend you are a financial expert. You are "
        f"a financial expert with stock recommendation experience. Your task is to judge {ticker},"
        f" Answer “YES” if good news, “NO” if bad news, 'UNKNOWN' if it is not clear based on the headline. Then"
        f"elaborate with one short and concise sentence on the next line. Is this headline"
        f"good or bad for the stock price of company name in the short term? Headline: {url}"
    )

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": promt}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")

    print("----------------- \n")

    count += 1

###

import dash
from dash import dcc, html, Input, Output, State

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("String Input App", style={"textAlign": "center", "color": "#ffffff", "backgroundColor": "#4CAF50", "padding": "20px", "borderRadius": "10px"}),
    ], style={"marginBottom": "20px"}),

    html.Div([
        html.Label("Enter a string:", style={"fontSize": "18px", "marginRight": "10px"}),
        dcc.Input(id='string-input', type='text', placeholder='Type something...', style={"width": "60%", "padding": "10px", "border": "1px solid #ddd", "borderRadius": "5px"}),
        html.Button('Submit', id='submit-button', n_clicks=0, style={"marginLeft": "10px", "padding": "10px 20px", "backgroundColor": "#4CAF50", "color": "#ffffff", "border": "none", "borderRadius": "5px", "cursor": "pointer"}),
    ], style={"padding": "20px", "textAlign": "center", "backgroundColor": "#f9f9f9", "border": "1px solid #ddd", "borderRadius": "10px", "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)"}),

    html.Div(id='output-text', style={
        "marginTop": "20px",
        "padding": "20px",
        "textAlign": "center",
        "fontSize": "24px",
        "color": "#4CAF50",
        "border": "1px solid #ddd",
        "borderRadius": "10px",
        "backgroundColor": "#ffffff",
        "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)"
    }),
])

@app.callback(
    Output('output-text', 'children'),
    Input('submit-button', 'n_clicks'),
    State('string-input', 'value')
)
def update_text(n_clicks, input_value):
    if n_clicks > 0:
        return f"You entered: {input_value}"
    return ""

if __name__ == '__main__':
    app.run_server(debug=False, port=5002)





