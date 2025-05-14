import numpy as np
from ib_insync import *
import pandas as pd

# Connect to IBKR Gateway or TWS
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=1)  # Use 4002 for IB Gateway paper trading

# Fetch open positions
positions = ib.positions()
positions_data = []

for pos in positions:
    contract = pos.contract
    details = ib.reqContractDetails(contract)

    if details:
        detail = details[0]
        long_name = detail.longName
        sec_id = detail.secIdList[0].value if detail.secIdList else "N/A"
    else:
        long_name = np.nan
        sec_id = np.nan

    positions_data.append({
        'ConID': contract.conId,
        'Symbol': contract.symbol,
        'Local Symbol': contract.localSymbol,
        'SecType': contract.secType,
        'Exchange': contract.exchange,
        'Name': long_name,
        'SecID': sec_id,
        'Currency': contract.currency,
        # 'Multiplier': contract.multiplier if hasattr(contract, 'multiplier') else 1,
        'Position': pos.position,
        # 'Avg Cost': pos.avgCost
    })

positions_df = pd.DataFrame(positions_data)

# Request all open orders (manual + API)
ib.reqAllOpenOrders()
trades = ib.trades()  # Fetch updated order list with contracts
# Convert trades to DataFrame
orders_df = pd.DataFrame([
    {
        'PermID': trade.order.permId,
        'ConID': trade.contract.conId,
        'Symbol': trade.contract.symbol,
        'Local Symbol': trade.contract.localSymbol,
        'SecType': trade.contract.secType,
        'Exchange': trade.contract.exchange,
        'Currency': trade.contract.currency,
        'Multiplier': trade.contract.multiplier if hasattr(trade.contract, 'multiplier') else 1,
        'Order Type': trade.order.orderType,
        'Quantity': trade.order.totalQuantity,
        'Action': trade.order.action,
        'Limit Price': trade.order.lmtPrice if hasattr(trade.order, 'lmtPrice') else None,
        'Stop Price': trade.order.auxPrice if hasattr(trade.order, 'auxPrice') else None,
        'Status': trade.orderStatus.status,
        'TIF': trade.order.tif,
        'Fills': trade.fills
    } for trade in trades if trade.order  # Ensure order exists
])

# delete filled and cancelled orders
orders_df = orders_df[~orders_df['Status'].isin(['Filled', 'Cancelled'])]

# remove CASH and WAR
positions_df = positions_df[~positions_df['SecType'].isin(['CASH', 'WAR'])]


