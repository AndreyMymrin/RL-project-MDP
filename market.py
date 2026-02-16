import numpy as np


class Market:
    def __init__(self, start_price=5.0, window=5):
        self.price = max(5.0, start_price)
        self.history = [self.price]
        self.state = 0
        self.window = window
        self.trans = np.array(
            [
                [0.8, 0.05, 0.15],
                [0.05, 0.8, 0.15],
                [0.2, 0.2, 0.6],
            ]
        )
        self.dists = [(5, 5), (-5, 5), (0, 5)]

    def step(self):
        self.state = np.random.choice(3, p=self.trans[self.state])
        mu, sigma = self.dists[self.state]
        next_price = self.price + np.random.normal(mu, sigma)
        self.price = max(5.0, next_price)
        self.history.append(self.price)

    def get_obs(self):
        return self.history[-self.window:]
