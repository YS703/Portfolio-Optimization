class account:
    def __init__(self, initial_balance: float):
        self.balance = initial_balance
        self.total_invested = initial_balance
        self.holdings = {} # Stocks the assets' quantities : { 'XWD.TO': 15.4 , ... }

    def deposit(self, amount: float):
        self.balance += amount
        self.total_invested += amount

    def invest(self, amount: float, stock: str, stock_price: float):
        self.balance -= amount
        qty = amount / stock_price if stock_price > 0 else 0
        if stock in self.holdings:
            self.holdings[stock] += qty
        else:
            self.holdings[stock] = qty

