import numpy as np

from agents import Bill, BollingerBandsAgent, Mark, MovingAverageCrossoverAgent, RSIAgent
from market import Market


class ReinforceAgent:
    def __init__(
        self,
        window=20,
        cash=1000.0,
        stocks=0.0,
        lr=0.01,
        gamma=0.99,
        seed=42,
        epsilon_start=0.3,
        epsilon_end=0.02,
        epsilon_decay=0.999,
        baseline_beta=0.05,
    ):
        self.name = "REINFORCE"
        self.window = window
        self.cash = cash
        self.stocks = stocks
        self.lr = lr
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.baseline_beta = baseline_beta

        self.actions = [-1.0, 0.0, 1.0]  # sell, hold, buy (fraction of holdings/cash)
        self.n_actions = len(self.actions)
        self.n_features = 6
        self.w = self.rng.normal(0.0, 0.01, size=(self.n_features, self.n_actions))
        self.b = np.zeros(self.n_actions)

        self.ep_states = []
        self.ep_actions = []
        self.ep_rewards = []
        self.total_steps = 0
        self.epsilon = epsilon_start
        self.running_return = 0.0
        self.baseline = 0.0
        self.trade_history = []

    def reset(self, cash=1000.0, stocks=0.0):
        self.cash = cash
        self.stocks = stocks
        self.ep_states = []
        self.ep_actions = []
        self.ep_rewards = []
        self.running_return = 0.0
        self.baseline = 0.0
        self.total_steps = 0
        self.epsilon = self.epsilon_start
        self.trade_history = []

    def capital(self, price):
        return self.cash + self.stocks * price

    def _features(self, obs):
        if len(obs) < 2:
            return np.zeros(self.n_features)
        window = obs[-self.window:]
        price = window[-1]
        prev = window[-2]
        ret = (price - prev) / max(1.0, prev)
        short = np.mean(window[-min(5, len(window)):])
        long = np.mean(window)
        ma_diff = (short - long) / max(1.0, long)
        vol = np.std(window) / max(1.0, long)
        mom = (price - window[0]) / max(1.0, window[0])
        return np.array([1.0, ret, ma_diff, vol, mom, price / 100.0])

    def _policy(self, feats):
        logits = feats @ self.w + self.b
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def act(self, obs, explore=True, step_index=None):
        feats = self._features(obs)
        probs = self._policy(feats)
        if explore:
            self.total_steps += 1
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            if self.rng.random() < self.epsilon:
                a_idx = int(self.rng.integers(self.n_actions))
            else:
                a_idx = self.rng.choice(self.n_actions, p=probs)
        else:
            a_idx = int(np.argmax(probs))
        action = self.actions[a_idx]

        price = obs[-1]
        if action > 0 and price > 0:
            spend = self.cash * action
            self.stocks += spend / price
            self.cash -= spend
            if step_index is not None:
                self.trade_history.append((step_index, price, "buy"))
        elif action < 0:
            sell = self.stocks * (-action)
            self.stocks -= sell
            self.cash += sell * price
            if step_index is not None:
                self.trade_history.append((step_index, price, "sell"))

        return feats, a_idx

    def record(self, feats, action_idx, reward):
        self.ep_states.append(feats)
        self.ep_actions.append(action_idx)
        self.ep_rewards.append(reward)

    def update(self):
        if not self.ep_rewards:
            return
        returns = []
        g = 0.0
        for r in reversed(self.ep_rewards):
            g = r + self.gamma * g
            returns.append(g)
        returns = np.array(list(reversed(returns)))
        if np.std(returns) > 1e-6:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        for feats, a_idx, g in zip(self.ep_states, self.ep_actions, returns):
            probs = self._policy(feats)
            grad_logits = -probs
            grad_logits[a_idx] += 1.0
            self.w += self.lr * g * np.outer(feats, grad_logits)
            self.b += self.lr * g * grad_logits

    def update_step(self, feats, a_idx, reward):
        self.running_return = self.gamma * self.running_return + reward
        self.baseline = (1.0 - self.baseline_beta) * self.baseline + self.baseline_beta * self.running_return
        advantage = self.running_return - self.baseline
        probs = self._policy(feats)
        grad_logits = -probs
        grad_logits[a_idx] += 1.0
        self.w += self.lr * advantage * np.outer(feats, grad_logits)
        self.b += self.lr * advantage * grad_logits


class ReinforceAgentMLP:
    def __init__(
        self,
        window=20,
        cash=1000.0,
        stocks=0.0,
        lr=0.01,
        gamma=0.99,
        seed=42,
        epsilon_start=0.3,
        epsilon_end=0.02,
        epsilon_decay=0.999,
        baseline_beta=0.05,
        hidden_size=128,
    ):
        self.name = "REINFORCE MLP"
        self.window = window
        self.cash = cash
        self.stocks = stocks
        self.lr = lr
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.baseline_beta = baseline_beta

        self.actions = [-1.0, 0.0, 1.0]
        self.n_actions = len(self.actions)
        self.n_features = 6
        self.hidden_size = hidden_size
        self.w1 = self.rng.normal(0.0, 0.05, size=(self.n_features, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size)
        self.w2 = self.rng.normal(0.0, 0.05, size=(self.hidden_size, self.hidden_size))
        self.b2 = np.zeros(self.hidden_size)
        self.w3 = self.rng.normal(0.0, 0.05, size=(self.hidden_size, self.hidden_size))
        self.b3 = np.zeros(self.hidden_size)
        self.w4 = self.rng.normal(0.0, 0.05, size=(self.hidden_size, self.n_actions))
        self.b4 = np.zeros(self.n_actions)

        self.ep_states = []
        self.ep_actions = []
        self.ep_rewards = []
        self.total_steps = 0
        self.epsilon = epsilon_start
        self.running_return = 0.0
        self.baseline = 0.0
        self.trade_history = []

    def reset(self, cash=1000.0, stocks=0.0):
        self.cash = cash
        self.stocks = stocks
        self.ep_states = []
        self.ep_actions = []
        self.ep_rewards = []
        self.running_return = 0.0
        self.baseline = 0.0
        self.total_steps = 0
        self.epsilon = self.epsilon_start
        self.trade_history = []

    def capital(self, price):
        return self.cash + self.stocks * price

    def _features(self, obs):
        if len(obs) < 2:
            return np.zeros(self.n_features)
        window = obs[-self.window:]
        price = window[-1]
        prev = window[-2]
        ret = (price - prev) / max(1.0, prev)
        short = np.mean(window[-min(5, len(window)):])
        long = np.mean(window)
        ma_diff = (short - long) / max(1.0, long)
        vol = np.std(window) / max(1.0, long)
        mom = (price - window[0]) / max(1.0, window[0])
        return np.array([1.0, ret, ma_diff, vol, mom, price / 100.0])

    def _policy(self, feats):
        h1 = feats @ self.w1 + self.b1
        h1 = np.maximum(0.0, h1)
        h2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0.0, h2)
        h3 = h2 @ self.w3 + self.b3
        h3 = np.maximum(0.0, h3)
        logits = h3 @ self.w4 + self.b4
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def _forward(self, feats):
        h1 = feats @ self.w1 + self.b1
        h1 = np.maximum(0.0, h1)
        h2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0.0, h2)
        h3 = h2 @ self.w3 + self.b3
        h3 = np.maximum(0.0, h3)
        logits = h3 @ self.w4 + self.b4
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        probs = exp / np.sum(exp)
        return (h1, h2, h3), probs

    def _backward(self, feats, hs, probs, a_idx, scale):
        h1, h2, h3 = hs
        grad_logits = -probs
        grad_logits[a_idx] += 1.0
        grad_logits *= scale

        grad_w4 = np.outer(h3, grad_logits)
        grad_b4 = grad_logits

        grad_h3 = self.w4 @ grad_logits
        grad_h3[h3 <= 0.0] = 0.0

        grad_w3 = np.outer(h2, grad_h3)
        grad_b3 = grad_h3

        grad_h2 = self.w3 @ grad_h3
        grad_h2[h2 <= 0.0] = 0.0

        grad_w2 = np.outer(h1, grad_h2)
        grad_b2 = grad_h2

        grad_h1 = self.w2 @ grad_h2
        grad_h1[h1 <= 0.0] = 0.0

        grad_w1 = np.outer(feats, grad_h1)
        grad_b1 = grad_h1

        self.w4 += self.lr * grad_w4
        self.b4 += self.lr * grad_b4
        self.w3 += self.lr * grad_w3
        self.b3 += self.lr * grad_b3
        self.w2 += self.lr * grad_w2
        self.b2 += self.lr * grad_b2
        self.w1 += self.lr * grad_w1
        self.b1 += self.lr * grad_b1

    def act(self, obs, explore=True, step_index=None):
        feats = self._features(obs)
        probs = self._policy(feats)
        if explore:
            self.total_steps += 1
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            if self.rng.random() < self.epsilon:
                a_idx = int(self.rng.integers(self.n_actions))
            else:
                a_idx = self.rng.choice(self.n_actions, p=probs)
        else:
            a_idx = int(np.argmax(probs))
        action = self.actions[a_idx]

        price = obs[-1]
        if action > 0 and price > 0:
            spend = self.cash * action
            self.stocks += spend / price
            self.cash -= spend
            if step_index is not None:
                self.trade_history.append((step_index, price, "buy"))
        elif action < 0:
            sell = self.stocks * (-action)
            self.stocks -= sell
            self.cash += sell * price
            if step_index is not None:
                self.trade_history.append((step_index, price, "sell"))

        return feats, a_idx

    def record(self, feats, action_idx, reward):
        self.ep_states.append(feats)
        self.ep_actions.append(action_idx)
        self.ep_rewards.append(reward)

    def update(self):
        if not self.ep_rewards:
            return
        returns = []
        g = 0.0
        for r in reversed(self.ep_rewards):
            g = r + self.gamma * g
            returns.append(g)
        returns = np.array(list(reversed(returns)))
        if np.std(returns) > 1e-6:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        for feats, a_idx, g in zip(self.ep_states, self.ep_actions, returns):
            hs, probs = self._forward(feats)
            self._backward(feats, hs, probs, a_idx, g)

    def update_step(self, feats, a_idx, reward):
        self.running_return = self.gamma * self.running_return + reward
        self.baseline = (1.0 - self.baseline_beta) * self.baseline + self.baseline_beta * self.running_return
        advantage = self.running_return - self.baseline
        hs, probs = self._forward(feats)
        self._backward(feats, hs, probs, a_idx, advantage)


def simulate_episode(
    agent,
    steps=200,
    start_price=5.0,
    window=20,
    seed=None,
    train=True,
    online_update=True,
):
    rng = np.random.default_rng(seed)
    np_random_state = np.random.get_state()
    np.random.seed(rng.integers(0, 2**32 - 1, dtype=np.uint32))

    market = Market(start_price=start_price, window=window)
    agent.reset(cash=1000.0, stocks=0.0)

    for _ in range(steps):
        prev_cap = agent.capital(market.price)
        obs = market.get_obs()
        feats, a_idx = agent.act(obs, explore=train)
        market.step()
        reward = agent.capital(market.price) - prev_cap
        if train:
            if online_update:
                agent.update_step(feats, a_idx, reward)
            else:
                agent.record(feats, a_idx, reward)

    if train and not online_update:
        agent.update()

    np.random.set_state(np_random_state)
    return agent.capital(market.price)


def evaluate_baselines(steps=200, start_price=5.0, window=20, seed=123):
    rng = np.random.default_rng(seed)
    np_random_state = np.random.get_state()
    np.random.seed(rng.integers(0, 2**32 - 1, dtype=np.uint32))

    market = Market(start_price=start_price, window=window)
    agents = [
        Mark(1000.0, 0.0),
        Bill(1000.0, 0.0),
        MovingAverageCrossoverAgent(1000.0, 0.0),
        RSIAgent(1000.0, 0.0),
        BollingerBandsAgent(1000.0, 0.0),
    ]
    for _ in range(steps):
        obs = market.get_obs()
        for a in agents:
            a.act(obs)
        market.step()

    np.random.set_state(np_random_state)
    return {a.name: a.capital(market.price) for a in agents}
