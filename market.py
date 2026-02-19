import numpy as np
import pandas as pd

import numpy as np


class NewMarket:
    """
    More dynamic yet stable synthetic market.
    - Price is always > 0 (log-price model).
    - Movement is driven by order-flow imbalance + noise + rare jumps.
    - Includes memory (pressure accumulation) and volatility clustering.
    Interface compatible with Market:
      - self.price, self.history, self.state
      - step()
      - get_obs()
    """

    def __init__(
        self,
        start_price=100.0,
        window=50,
        seed=None,
        price_floor=1e-6,
    ):
        self.rng = np.random.default_rng(seed)

        self.window = int(window)
        self.price_floor = float(price_floor)

        self.log_price = float(np.log(max(start_price, self.price_floor)))
        self.price = float(np.exp(self.log_price))
        self.history = [self.price]

        # hidden states: 0 growth, 1 depression, 2 volatility
        self.state = 0
        self.trans = np.array(
            [
                [0.90, 0.04, 0.06],
                [0.04, 0.90, 0.06],
                [0.12, 0.12, 0.76],
            ],
            dtype=float,
        )

        # --- Regime parameters (tune for desired dynamics) ---
        # drift of log-return per step (very small)
        self.state_drift = np.array([+0.0006, -0.0006, 0.0000], dtype=float)

        # base volatility (std of noise in log-return)
        self.state_sigma = np.array([0.006, 0.006, 0.014], dtype=float)

        # order flow intensity (expected number of orders per step)
        self.state_intensity = np.array([8.0, 8.0, 16.0], dtype=float)

        # probability of buy orders (order flow imbalance)
        self.state_buy_prob = np.array([0.58, 0.42, 0.50], dtype=float)

        # "market depth" (larger → smaller price impact from imbalance)
        self.state_depth = np.array([1.2, 1.2, 0.7], dtype=float)

        # --- Memory / stability ---
        # AR(1) memory of imbalance: closer to 1 → longer trends
        self.imbalance_rho = 0.92
        # scale of imbalance impact on log-price
        self.impact_k = 0.035

        # volatility clustering (EWMA of volatility level)
        self.vol_ewma = self.state_sigma[self.state]
        self.vol_alpha = 0.93  # closer to 1 → stronger clustering

        # rare jumps (news / thin liquidity events)
        self.jump_prob = 0.02
        self.jump_scale = 0.030  # typical jump size in log space

        # internal state variable
        self.imbalance = 0.0  # accumulated order-flow imbalance

    def step(self):
        # 1) state transition
        self.state = int(self.rng.choice(3, p=self.trans[self.state]))

        # 2) generate order flow and update imbalance
        intensity = self.state_intensity[self.state]
        n = int(self.rng.poisson(intensity))

        buy_p = self.state_buy_prob[self.state]
        # buy_count ~ Binomial(n, buy_p)
        buy_count = int(self.rng.binomial(n, buy_p))
        sell_count = n - buy_count

        # raw imbalance for this step (normalized for stability)
        raw_imb = (buy_count - sell_count) / max(1.0, np.sqrt(n + 1e-6))

        # AR(1) accumulation of imbalance (memory → trends)
        self.imbalance = self.imbalance_rho * self.imbalance + (1.0 - self.imbalance_rho) * raw_imb

        # 3) noise + volatility clustering
        base_sigma = self.state_sigma[self.state]
        # EWMA volatility reacts to recent activity,
        # but is bounded below to avoid collapse
        self.vol_ewma = max(
            0.5 * base_sigma,
            self.vol_alpha * self.vol_ewma + (1.0 - self.vol_alpha) * base_sigma,
        )

        noise = self.rng.normal(0.0, self.vol_ewma)

        # 4) rare jumps (more likely in volatility regime)
        jump = 0.0
        if self.rng.random() < (self.jump_prob * (1.8 if self.state == 2 else 1.0)):
            # jump sign slightly correlated with imbalance
            sign = 1.0 if (self.rng.random() < (0.5 + 0.25 * np.tanh(self.imbalance))) else -1.0
            jump = sign * abs(self.rng.normal(0.0, self.jump_scale))

        # 5) update log-price
        drift = self.state_drift[self.state]
        depth = self.state_depth[self.state]

        # smaller depth → larger price impact (thin liquidity)
        impact = (self.impact_k / max(0.15, depth)) * np.tanh(self.imbalance)

        self.log_price = self.log_price + drift + impact + noise + jump

        # 6) update price (strictly > 0)
        self.price = float(max(self.price_floor, np.exp(self.log_price)))
        self.history.append(self.price)

    def get_obs(self):
        return self.history[-self.window:]


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

        prices = df["Close"].values.astype(float)[2000:]


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

