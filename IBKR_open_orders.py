from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.execution import ExecutionFilter
from ibapi.order import Order
import pandas as pd
import time
import threading

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.orders_list = []  # List to store order data
        self.trades_list = []  # List to store executed trades
        self.positions_list = []  # List to store position data

    def nextValidId(self, orderId):
        """ IBKR assigns a valid order ID when connected. """
        print(f"Connected! Next Order ID: {orderId}")
        self.reqAllOpenOrders()  # Request open orders for ALL subaccounts
        self.reqExecutions(1001, ExecutionFilter())  # Request all executions for today
        self.reqPositions()  # Request all open positions

    def openOrder(self, orderId, contract, order, orderState):
        """ Callback method that receives open orders. """
        order_data = {
            "Order ID": orderId,
            "Account": order.account,  # Important for subaccounts!
            "Symbol": contract.symbol,
            "Sec Type": contract.secType,
            "Exchange": contract.exchange,
            "Action": order.action,
            "Quantity": order.totalQuantity,
            "Order Type": order.orderType,
            "Price": order.lmtPrice ,# if order.orderType in ["LMT", "STP"] else "Market",
            "Status": orderState.status
        }
        self.orders_list.append(order_data)
        print(f"Received Order: {order_data}")  # Debugging output

    def execDetails(self, reqId, contract, execution):
        """ Called when trade execution details are received. """
        trade_data = {
            "Execution ID": execution.execId,
            "Order ID": execution.orderId,
            "Account": execution.acctNumber,
            "Symbol": contract.symbol,
            "Sec Type": contract.secType,
            "Exchange": execution.exchange,
            "Action": execution.side,  # BUY/SELL
            "Quantity": execution.shares,
            "Price": execution.price,
            "Execution Time": execution.time,
            "Permanent ID": execution.permId
        }
        self.trades_list.append(trade_data)
        print(f"Received Execution: {trade_data}")  # Debugging output

    def position(self, account, contract, position, avgCost):
        """ Called when an open position is received. """
        position_data = {
            "Account": account,  # Important for subaccounts!
            "Symbol": contract.symbol,
            "Sec Type": contract.secType,
            "Exchange": contract.exchange,
            "Position": position,  # Positive for long, negative for short
            "Avg Cost": avgCost
        }
        self.positions_list.append(position_data)

    def openOrderEnd(self):
        """ Triggered when all open orders are received. """
        print("All open orders received.")

# Start IBAPI
app = IBapi()
app.connect('127.0.0.1', 7496, clientId=1)  # Use 7496 for live trading

# Run the client in a separate thread to avoid blocking
api_thread = threading.Thread(target=app.run, daemon=True)
api_thread.start()

# Allow time for data collection
time.sleep(3)

# Convert collected orders to a DataFrame
df_orders = pd.DataFrame(app.orders_list)
trade_orders = pd.DataFrame(app.trades_list)
df_positions = pd.DataFrame(app.positions_list)

# Disconnect after fetching data
app.disconnect()


