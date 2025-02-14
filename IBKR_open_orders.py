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
        self.contract_details_map = {}  # Map permId -> longName

    def nextValidId(self, orderId):
        """ IBKR assigns a valid order ID when connected. """
        print(f"Connected! Next Order ID: {orderId}")
        self.reqAllOpenOrders()  # Request open orders for ALL subaccounts
        self.reqExecutions(1001, ExecutionFilter())  # Request all executions for today
        self.reqPositions()  # Request all open positions

    def openOrder(self, orderId, contract, order, orderState):

        self.reqContractDetails(order.permId, contract)

        """ Callback method that receives open orders. """
        order_data = {
            "Perm ID": order.permId,
            "Account": order.account,  # Important for subaccounts!
            "Symbol": contract.symbol,
            "Sec Type": contract.secType,
            "Exchange": contract.exchange,
            "Action": order.action,
            "Quantity": order.totalQuantity,
            "Order Type": order.orderType,
            "Price": order.lmtPrice ,# if order.orderType in ["LMT", "STP"] else "Market",
            "Status": orderState.status,
            "CondId": contract.conId,
            "Long Name": "Fetching..."  # Placeholder
        }
        self.orders_list.append(order_data)
        print(f"Received Order: {order_data}")  # Debugging output

        # Request contract details to get longName, linked by permId
        if contract.conId not in self.contract_details_map:  # Avoid duplicate requests
            self.reqContractDetails(contract.conId, contract)

    def execDetails(self, reqId, contract, execution):
        """ Called when trade execution details are received. """
        trade_data = {
            "Execution_ID": execution.execId,
            "Account": execution.acctNumber,
            "Execution_Time": execution.time,
            "Permanent_ID": execution.permId,
            "CondId": contract.conId,
            "Symbol": contract.symbol,
            "Sec Type": contract.secType,
            "Exchange": execution.exchange,
            "Action": execution.side,  # BUY/SELL
            "Quantity": execution.shares,
            "Price": execution.price,
            "CumQty_filled": execution.cumQty,
            "Avg_Price":execution.avgPrice,
            "OrderRef":execution.orderRef,
            "Market_rule_ID":execution.evRule,
            "Multiplier":execution.evMultiplier,

        }
        self.trades_list.append(trade_data)
        print(f"Received Execution: {trade_data}")  # Debugging output

    def position(self, account, contract, position, avgCost):
        """ Called when an open position is received. """
        base_currency = "USD"
        fx_rate = self.get_fx_rate_ibkr(base_currency, contract.currency)  # Fetch dynamically
        multiplier = float(contract.multiplier) if contract.multiplier else 1.0

        # Calculate notional exposure
        notional_exposure = position * multiplier * avgCost * fx_rate

        position_data = {
            "Account": account,
            "Symbol": contract.symbol,
            "Sec Type": contract.secType,
            "Exchange": contract.exchange,
            "CondId": contract.conId,
            "Multiplier": multiplier,
            "Currency": contract.currency,
            "Position": position,
            "Avg Cost": avgCost,
            "Notional Exposure": notional_exposure
        }
        self.positions_list.append(position_data)
        print(f"Processed Position: {position_data}")

    def contractDetails(self, reqId, contractDetails):
        """ Called when contract details are received. """
        longName = contractDetails.longName
        permId = reqId  # We requested it using permId, so reqId == permId

        # Store contract longName in a dictionary using permId as key
        self.contract_details_map[permId] = longName

        # Update existing orders with the correct name
        for order in self.orders_list:
            if order["Perm ID"] == permId:
                order["Long Name"] = longName

        print(f"Received Contract Details: {contractDetails.contract.symbol} - {longName}")


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
open_orders = pd.DataFrame(app.orders_list)
trades = pd.DataFrame(app.trades_list)
open_positions = pd.DataFrame(app.positions_list)

# Disconnect after fetching data
app.disconnect()

#