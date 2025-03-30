from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import *
import time

class IBApp(EWrapper, EClient):

    def __init__(self):
        EClient.__init__(self, self)
        self.positions = []
        self.market_data = {}

    def nextValidId(self, orderId: int):
        """ Called when the connection is made, orderId is ready """
        self.start()

    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """ Called for each position """
        self.positions.append({
            "account": account,
            "symbol": contract.symbol,
            "position": position,
            "avgCost": avgCost
        })

    def positionEnd(self):
        """ Called when the positions are fully received """
        print("Positions received")
        self.request_last_prices()

    def tickPrice(self, tickerId: int, field: int, price: float, attrib: TickAttrib):
        """ Called for each price update """
        if tickerId in self.market_data:
            self.market_data[tickerId] = price

    def request_last_prices(self):
        """ Request the market data for all positions """
        for idx, position in enumerate(self.positions):
            contract = Contract()
            contract.symbol = position["symbol"]
            contract.secType = "STK"
            contract.currency = "USD"
            contract.exchange = "SMART"
            self.reqMktData(idx, contract, "", False, False, [])

    def start(self):
        """ Request positions """
        self.reqPositions()

    def run(self):
        """ Run the client """
        self.runLoop()

def main():
    app = IBApp()
    app.connect("127.0.0.1", 7496, 0)  # Connect to TWS or IB Gateway
    app.run()

    # Wait for positions and prices to be populated
    time.sleep(5)

    # Print out the positions and last prices
    for position in app.positions:
        symbol = position['symbol']
        quantity = position['position']
        avg_cost = position['avgCost']
        last_price = app.market_data.get(symbol, "N/A")
        print(f"Symbol: {symbol}, Quantity: {quantity}, Avg Cost: {avg_cost}, Last Price: {last_price}")

    app.disconnect()

if __name__ == "__main__":
    main()
