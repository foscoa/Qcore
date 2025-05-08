from ib_insync import *
import pandas as pd

# Connect to IBKR Gateway or TWS
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=1)  # Use 4002 for IB Gateway paper trading

# Fetch open positions
positions = ib.positions()
positions_df = pd.DataFrame([
    {
        'ConID': pos.contract.conId,
        'Symbol': pos.contract.symbol,
        'Local Symbol': pos.contract.localSymbol,
        'SecType': pos.contract.secType,
        'Exchange': pos.contract.exchange,
        'Currency': pos.contract.currency,
        'Multiplier': pos.contract.multiplier if hasattr(pos.contract, 'multiplier') else 1,
        'Position': pos.position,
        'Avg Cost': pos.avgCost
    } for pos in positions
])

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
