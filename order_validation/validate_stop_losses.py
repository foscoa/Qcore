from ib_insync import IB, util
import pandas as pd
import numpy as np
from datetime import datetime
import win32com.client as win32

# Connect to IBKR Gateway or TWS
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=1)  # Use 4002 for IB Gateway paper trading

# === Fetch Open Positions ===
positions = ib.positions()
positions_data = []

for pos in positions:
    contract = pos.contract
    details = ib.reqContractDetails(contract)
    detail = details[0] if details else None

    positions_data.append({
        'ConID': contract.conId,
        'Symbol': contract.symbol,
        'Position': pos.position,
        'Local Symbol': contract.localSymbol,
        'SecType': contract.secType,
        'Name': detail.longName if detail else np.nan,
        'SecID': detail.secIdList[0].value if detail and detail.secIdList else np.nan,
        'Currency': contract.currency
    })

positions_df = pd.DataFrame(positions_data)
positions_df['Direction'] = positions_df['Position'].apply(lambda x: 'LONG' if x > 0 else 'SHORT')

# === Fetch Open Orders ===
ib.reqAllOpenOrders()
trades = ib.trades()

orders_data = []
for trade in trades:
    order = trade.order
    contract = trade.contract
    if not order:
        continue

    orders_data.append({
        'PermID': order.permId,
        'ConID': contract.conId,
        'Symbol': contract.symbol,
        'Local Symbol': contract.localSymbol,
        'SecType': contract.secType,
        'Exchange': contract.exchange,
        'Currency': contract.currency,
        'Multiplier': getattr(contract, 'multiplier', 1),
        'Order Type': order.orderType,
        'Quantity': order.totalQuantity,
        'Action': order.action,
        'Limit Price': getattr(order, 'lmtPrice', None),
        'Stop Price': getattr(order, 'auxPrice', None),
        'Status': trade.orderStatus.status,
        'TIF': order.tif,
        'Valid During RTH': not getattr(order, 'outsideRth', False)
    })

orders_df = pd.DataFrame(orders_data)
orders_df = orders_df[~orders_df['Status'].isin(['Filled', 'Cancelled'])]

# === Market Prices for Current Positions ===
portfolio = ib.portfolio()
market_price_df = pd.DataFrame([{
    'ConID': pos.contract.conId,
    'Market Price': pos.marketPrice
} for pos in portfolio])

ib.disconnect()

# === Merge Positions with Market Price ===
positions_df = positions_df.merge(market_price_df, on='ConID', how='inner')

# === Identify Positions Without Matching Stop Losses ===
contracts_without_stop = []
contracts_with_stop = []

for _, pos_row in positions_df.iterrows():
    conid = pos_row['ConID']
    curr_pos = pos_row['Position']
    mkt_price = pos_row['Market Price']

    sub_orders = orders_df[orders_df['ConID'] == conid]

    # contracts is not in orders
    if sub_orders.empty:
        contracts_without_stop.append(conid)
        continue

    if curr_pos > 0:
        stops = sub_orders[~((sub_orders['Action'] == 'SELL') & (sub_orders['Limit Price'] > mkt_price))]
    else:
        stops = sub_orders[~((sub_orders['Action'] == 'BUY') & (0 < sub_orders['Limit Price']) & (sub_orders['Limit Price'] < mkt_price))]

    net_pos = curr_pos + (stops['Action'].map({'SELL': -1, 'BUY': 1}) * stops['Quantity']).sum()

    if net_pos != 0:
        contracts_without_stop.append(conid)
    else:
        contracts_with_stop.append(conid)

# === Final Output ===
contracts_without_stop_df = positions_df[positions_df['ConID'].isin(contracts_without_stop)].sort_values(by='SecType')
contracts_without_stop_df.drop(['ConID'], axis=1, inplace=True)

contracts_without_stop_df = contracts_without_stop_df[['Symbol', 'Local Symbol','Name', 'Currency', 'SecID', 'SecType',
        'Direction', 'Position', 'Market Price']]

contracts_with_stop_df = pd.concat([positions_df, orders_df])
contracts_with_stop_df = contracts_with_stop_df[contracts_with_stop_df['ConID'].isin(contracts_with_stop)].sort_values(by='ConID')

contracts_with_stop_df = contracts_with_stop_df[['Symbol',
                     'Local Symbol',
                     'Name',
                     'SecType',
                     'Currency',
                     'Market Price',
                     'Direction',
                     'Position',
                     'Quantity',
                     'Order Type',
                     'Action',
                     'Limit Price',
                     'Stop Price',
                     'Status',
                     'TIF',
                     'Valid During RTH']]

contracts_with_stop_df.rename(columns = {'Valid During RTH': 'Validity',
                                         'Quantity': 'Order Quantity',
                                         'Position': 'Open Position'}, inplace=True)

contracts_with_stop_df['Validity'] = contracts_with_stop_df['Validity'].map({1:'RTH', 0:'ORTH'})


#### EMAIL #### --------------------------------------------------------------------------------------------------------

# Round all numeric columns to 4 decimal places
contracts_without_stop_df = contracts_without_stop_df.round(4)
contracts_with_stop_df = contracts_with_stop_df.round(4)

# Replace NaNs with empty strings
contracts_without_stop_df = contracts_without_stop_df.fillna("")
contracts_with_stop_df = contracts_with_stop_df.fillna("")


# Function to generate the table HTML with inline styles
def generate_html_table(df):
    html = '<table style="width: 100%; border-collapse: collapse; margin: 10px 0; border: 0px solid #444;">'

    # Table headers with inline styles
    html += '<thead>'
    html += '<tr>'
    for column in df.columns:
        html += f'<th style="background-color: rgb(18,54,90); color: white; font-weight: bold; padding: 0px; text-align: center; border: 1px solid #444;">{column}</th>'
    html += '</tr>'
    html += '</thead>'

    # Table body with inline styles
    html += '<tbody>'
    for _, row in df.iterrows():
        html += '<tr>'
        for value in row:
            html += f'<td style="background-color: #ecf0f1; color: #2c3e50; padding: 0px; text-align: center; border: 1px solid #444;">{value}</td>'
        html += '</tr>'
    html += '</tbody>'

    html += '</table>'
    return html


# Generate HTML table for positions without stop-loss
contracts_without_stop_html = generate_html_table(contracts_without_stop_df)

# Build grouped HTML tables by Symbol with inline styles
grouped_tables_html = ""
for symbol, group_df in contracts_with_stop_df.groupby("Symbol"):
    group_df.sort_values(by='Name', ascending=False, inplace=True)
    group_html = generate_html_table(group_df)
    grouped_tables_html += f"<h4>📈 Symbol: {symbol}</h4>{group_html}<br>"

# Construct the HTML body
html_body = f"""
<html>
<head>
</head>
<body>
<p style="font-size:15px; color:gray; margin-top:20px;">
    <em>This email was generated and sent automatically by a monitoring system.
    <span style="color:red;"> Please review the content carefully and take appropriate action if needed.</span></em>
</p>

<h3>⚠️ Positions without matching Stop Loss Orders</h3>
<p>The following positions do not have matching stop-loss orders in place:</p>
{contracts_without_stop_html}

<br><h3>✅ Positions with matching Stop Loss Orders</h3>
{grouped_tables_html}


<p>Best regards,<br>Q-PT Team</p>
</body>
</html>
"""


# Create and send the Outlook email

outlook = win32.Dispatch('Outlook.Application')
namespace = outlook.GetNamespace("MAPI")

current_hour = datetime.now().hour

if current_hour < 12:
    title = "Stop Loss Validation (Q-PT) – Morning Run"
else:
    title = "Stop Loss Validation (Q-PT) – Afternoon Run"



mail = outlook.CreateItem(0)
mail.To = 'fosco.antognini@qcore.ch; pc@qcore.group; nh@qcore.fund; sven.schmidt@qcore.group'
mail.Subject = title
mail.HTMLBody = html_body

#mail.Display()
mail.Send()
