import numpy as np


def _log_trade(agent, step_index, price, side, trade_value, confidence=1.0):
    if step_index is None or trade_value <= 0:
        return
    agent.trade_history.append((int(step_index), float(price), str(side), float(trade_value), float(confidence)))


class Mark:
    def __init__(self, cash=1000.0, stocks=0.0):
        self.name = "Cautious Mark"
        self.cash = cash
        self.stocks = stocks
        self.trade_history = []

    def capital(self, price):
        return self.cash + self.stocks * price

    def act(self, obs, step_index=None):
        if len(obs) >= 3:
            price = obs[-1]
            if obs[-3] < obs[-2] < obs[-1]:
                amt = self.stocks * 0.1
                self.stocks -= amt
                self.cash += amt * price
                _log_trade(self, step_index, price, "sell", amt * price, 1.0)
            elif obs[-3] > obs[-2] > obs[-1]:
                if price > 0:
                    cost = self.cash * 0.1
                    self.stocks += cost / price
                    self.cash -= cost
                    _log_trade(self, step_index, price, "buy", cost, 1.0)


class Bill:
    def __init__(self, cash=1000.0, stocks=0.0):
        self.name = "Risky Bill"
        self.cash = cash
        self.stocks = stocks
        self.trade_history = []

    def capital(self, price):
        return self.cash + self.stocks * price

    def act(self, obs, step_index=None):
        if len(obs) > 0:
            price = obs[-1]
            if price == min(obs) and price > 0:
                spend = self.cash
                self.stocks += self.cash / price
                self.cash = 0.0
                _log_trade(self, step_index, price, "buy", spend, 1.0)
            elif price == max(obs):
                sell_value = self.stocks * price
                self.cash += self.stocks * price
                self.stocks = 0.0
                _log_trade(self, step_index, price, "sell", sell_value, 1.0)


class MovingAverageCrossoverAgent:
    def __init__(self, cash=1000.0, stocks=0.0, short_window=5, long_window=20):
        self.name = "MA Crossover"
        self.cash = cash
        self.stocks = stocks
        self.short_window = short_window
        self.long_window = long_window
        self.prev_diff = 0.0
        self.trade_history = []

    def capital(self, price):
        return self.cash + self.stocks * price

    def act(self, obs, step_index=None):
        if len(obs) < self.long_window:
            return
        short = np.mean(obs[-self.short_window:])
        long = np.mean(obs[-self.long_window:])
        diff = short - long
        price = obs[-1]

        if self.prev_diff <= 0.0 and diff > 0.0 and price > 0:
            spend = self.cash * 0.5
            self.stocks += spend / price
            self.cash -= spend
            _log_trade(self, step_index, price, "buy", spend, 1.0)
        elif self.prev_diff >= 0.0 and diff < 0.0:
            sell = self.stocks * 0.5
            self.stocks -= sell
            self.cash += sell * price
            _log_trade(self, step_index, price, "sell", sell * price, 1.0)

        self.prev_diff = diff


class RSIAgent:
    def __init__(self, cash=1000.0, stocks=0.0, period=14, low=30.0, high=70.0):
        self.name = "RSI 14"
        self.cash = cash
        self.stocks = stocks
        self.period = period
        self.low = low
        self.high = high
        self.trade_history = []

    def capital(self, price):
        return self.cash + self.stocks * price

    def _rsi(self, obs):
        if len(obs) < self.period + 1:
            return None
        gains = []
        losses = []
        for i in range(-self.period, 0):
            delta = obs[i] - obs[i - 1]
            if delta >= 0:
                gains.append(delta)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(-delta)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0.0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def act(self, obs, step_index=None):
        rsi = self._rsi(obs)
        if rsi is None:
            return
        price = obs[-1]
        if rsi <= self.low and price > 0:
            spend = self.cash * 0.5
            self.stocks += spend / price
            self.cash -= spend
            _log_trade(self, step_index, price, "buy", spend, 1.0)
        elif rsi >= self.high:
            sell = self.stocks * 0.5
            self.stocks -= sell
            self.cash += sell * price
            _log_trade(self, step_index, price, "sell", sell * price, 1.0)


class BollingerBandsAgent:
    def __init__(self, cash=1000.0, stocks=0.0, window=20, num_std=2.0):
        self.name = "Bollinger Bands"
        self.cash = cash
        self.stocks = stocks
        self.window = window
        self.num_std = num_std
        self.trade_history = []

    def capital(self, price):
        return self.cash + self.stocks * price

    def act(self, obs, step_index=None):
        if len(obs) < self.window:
            return
        window = np.array(obs[-self.window:])
        mid = np.mean(window)
        std = np.std(window)
        upper = mid + self.num_std * std
        lower = mid - self.num_std * std
        price = obs[-1]

        if price <= lower and price > 0:
            spend = self.cash * 0.5
            self.stocks += spend / price
            self.cash -= spend
            _log_trade(self, step_index, price, "buy", spend, 1.0)
        elif price >= upper:
            sell = self.stocks * 0.5
            self.stocks -= sell
            self.cash += sell * price
            _log_trade(self, step_index, price, "sell", sell * price, 1.0)


class RandomAgent:
    def __init__(self, cash=1000.0, stocks=0.0, seed=42):
        self.name = "Random Monkey"
        self.cash = cash
        self.stocks = stocks
        self.rng = np.random.default_rng(seed)
        self.trade_history = []

    def capital(self, price):
        return self.cash + self.stocks * price

    def act(self, obs, step_index=None):
        if len(obs) == 0:
            return
        price = obs[-1]
        action = self.rng.choice(["buy", "sell", "hold"])

        if action == "buy" and price > 0:
            amt = self.cash * 0.2
            self.stocks += amt / price
            self.cash -= amt
            _log_trade(self, step_index, price, "buy", amt, 1.0)
        elif action == "sell" and self.stocks > 0:
            amt = self.stocks * 0.2
            self.stocks -= amt
            self.cash += amt * price
            _log_trade(self, step_index, price, "sell", amt * price, 1.0)


class BuyAndHoldAgent:
    def __init__(self, cash=1000.0, stocks=0.0):
        self.name = "HODL"
        self.cash = cash
        self.stocks = stocks
        self.first_step = True
        self.trade_history = []

    def capital(self, price):
        return self.cash + self.stocks * price

    def act(self, obs, step_index=None):
        if self.first_step and len(obs) > 0:
            price = obs[-1]
            if price > 0:
                spend = self.cash
                self.stocks += self.cash / price
                self.cash = 0.0
                self.first_step = False
                _log_trade(self, step_index, price, "buy", spend, 1.0)

class InsiderAgent:
    def __init__(self, market, cash=1000.0, stocks=0.0):
        self.name = "Insider (Whale)"
        self.market = market  
        self.cash = cash
        self.stocks = stocks

    def capital(self, price):
        return self.cash + self.stocks * price

    def act(self, obs, step_index=None):
        if len(obs) == 0:
            return
        
        current_price = self.market.price
        
        if self.market.mode == "synthetic":
            market_state = self.market.state
            
            if market_state == 0:  
                if self.cash > 0:
                    self.stocks += self.cash / current_price
                    self.cash = 0.0
            elif market_state == 1:  
                if self.stocks > 0:
                    self.cash += self.stocks * current_price
                    self.stocks = 0.0
            else: 
                pass
                
        elif self.market.mode == "historical":
            h_idx = self.market.h_index
            if h_idx + 1 < len(self.market.historical_prices):
                next_price = self.market.historical_prices[h_idx + 1]
                if next_price > current_price:
                    self.stocks += self.cash / current_price
                    self.cash = 0.0
                elif next_price < current_price:
                    self.cash += self.stocks * current_price
                    self.stocks = 0.0