import numpy as np
import pandas as pd


class Market:
    def __init__(
        self,
        start_price=5.0,
        window=5,
        mode="synthetic",          # "synthetic" или "historical"
        csv_path="Tesla.csv",
    ):
        self.start_price = float(start_price)
        self.window = window
        self.mode = mode

        # ===== synthetic параметры =====
        self.trans = np.array(
            [
                [0.8, 0.05, 0.15],
                [0.05, 0.8, 0.15],
                [0.2, 0.2, 0.6],
            ]
        )
        self.dists = [(5, 5), (-5, 5), (0, 5)]

        if self.mode == "synthetic":
            self._init_synthetic()

        elif self.mode == "historical":
            self._init_historical(csv_path)

        else:
            raise ValueError("mode must be 'synthetic' or 'historical'")

    # ============================================================
    # Synthetic market
    # ============================================================
    def _init_synthetic(self):
        self.state = 0
        self.price = max(5.0, self.start_price)
        self.history = [self.price]

    def _step_synthetic(self):
        self.state = np.random.choice(3, p=self.trans[self.state])
        mu, sigma = self.dists[self.state]
        next_price = self.price + np.random.normal(mu, sigma)
        self.price = max(5.0, next_price)
        self.history.append(self.price)

    # ============================================================
    # Historical market
    # ============================================================
    def _init_historical(self, csv_path):
        self.state = 0
        df = pd.read_csv(csv_path)

        if "Close" not in df.columns:
            raise ValueError("CSV must contain column 'Close'")

        prices = df["Close"].values.astype(float)

        if len(prices) < 2:
            raise ValueError("Historical dataset too short")

        # нормировка: первая цена → start_price
        scale = self.start_price / prices[0]
        prices = prices * scale

        self.historical_prices = prices
        self.h_index = 0

        self.price = float(prices[0])
        self.history = [self.price]

    def _step_historical(self):
        self.h_index += 1

        if self.h_index >= len(self.historical_prices):
            # если дошли до конца — можно:
            # 1) остановить
            # 2) зациклить
            # 3) выбросить исключение
            # здесь сделаем цикл
            self.h_index = 0

        self.price = float(self.historical_prices[self.h_index])
        self.history.append(self.price)

    # ============================================================
    # Unified step
    # ============================================================
    def step(self):
        if self.mode == "synthetic":
            self._step_synthetic()
        else:
            self._step_historical()

    # ============================================================
    def get_obs(self):
        return self.history[-self.window:]

