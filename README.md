# RL-project-MDP — RL Trader in a Hidden-Regime Market (MDP / POMDP)

A toy **algorithmic trading** project where an agent learns to trade a single asset in a simplified market with **hidden internal regimes** (e.g., growth / depression / volatility).  
The agent observes prices and its own portfolio, but **does not directly observe the market regime**.

This repository currently contains a **first working iteration**: a synthetic (and optional historical) market, several baseline strategies, and an RL agent trained with **REINFORCE**.

---

## Contents

- [Project idea](#project-idea)
- [Environment](#environment)
  - [State (observations)](#state-observations)
  - [Hidden regime dynamics](#hidden-regime-dynamics)
  - [Actions](#actions)
  - [Reward](#reward)
- [Agents](#agents)
- [Code structure](#code-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Configuration tips](#configuration-tips)
- [Roadmap](#roadmap)

---

## Project idea

We model a trader interacting with a “market world”.

- The market has an **observable price** and a **hidden regime** (trend up / trend down / volatile).
- Price transitions are **(approximately) Markovian** given the current hidden regime.
- The agent’s objective is to maximize **final capital**:

\[
\text{capital} = \text{cash} + \text{stocks} \times \text{current\_price}
\]

This is essentially a **POMDP**: the agent cannot see the true regime, only price history (which contains information about the regime).

---

## Environment

### State (observations)

In the current implementation, the market provides the agent with the last `window` prices:

- `Market.get_obs()` returns `history[-window:]`

The agent also maintains its own internal portfolio state:

- `cash` — unspent money
- `stocks` — number of shares held

> The hidden regime is **not** included in observations.

### Hidden regime dynamics

`market.py` implements two modes:

#### 1) Synthetic mode (default)
- Hidden regime is sampled from a transition matrix.
- Price changes by adding Gaussian noise with regime-dependent `(mu, sigma)`.

#### 2) Historical mode
- Loads prices from `Tesla.csv` (`Close` column) and replays them in a loop.
- The hidden regime is not meaningful in this mode (it’s essentially a deterministic replay).

### Actions

There are two “action styles”:

#### Current implementation (discrete fractions)
The RL agent uses **three actions**:

- `-1.0` → sell a fraction of current holdings
- `0.0`  → hold
- `+1.0` → buy using a fraction of available cash

This keeps the baseline REINFORCE implementation simple and stable.

#### Planned (continuous control)
A more realistic design is a continuous action `k ∈ [-1, 1]`:

- `k > 0`: buy for `k * cash`
- `k < 0`: sell `(-k) * stocks`

This avoids “buy and sell in the same step” and matches common continuous-action RL formulations.

### Reward

In the current code, the reward is:

- **Δcapital** between steps
- minus small penalties for trading and for switching trade direction

This encourages profitability while discouraging excessive churning.

---

## Agents

Baselines (rule-based), implemented in `agents.py`:

- **Cautious Mark** — trades on short-term monotonic price patterns
- **Risky Bill** — buys at local minima and sells at local maxima (within the window)
- **MA Crossover** — moving average crossover strategy
- **RSI 14** — RSI-based strategy
- **Bollinger Bands** — Bollinger-band strategy
- **Random Monkey** — random buy/sell/hold actions
- **HODL** — buys once and holds
- **Insider (Whale)** — (debug agent) can access hidden info in synthetic mode / looks ahead in historical mode

RL agents (in `train.py`):

- **ReinforceAgent** — linear softmax policy over handcrafted features
- **ReinforceAgentMLP** — simple NumPy MLP policy (still trained with REINFORCE-style gradient updates)

---

## Code structure

```

.
├── README.md
├── Tesla.csv
├── agents.py        # baseline strategies + helper logging
├── market.py        # synthetic/historical market
├── train.py         # REINFORCE agents + simulate_episode()
├── evaluate.py      # metrics + Monte Carlo evaluation + equity curve plot
├── visualize.py     # matplotlib animation visualizer
└── main.py          # entry point: train → evaluate → visualize

````

---

## Installation

### Requirements

This project uses standard scientific Python:

- `numpy`
- `pandas`
- `matplotlib`

Install via pip:

```bash
pip install numpy pandas matplotlib
````

(Optionally use a virtual environment.)

---

## Quick start

Run the full pipeline:

```bash
python main.py
```

What it does:

1. Trains the REINFORCE agent for a number of episodes.
2. Evaluates the RL agent vs baseline strategies.
3. Plots equity curves.
4. Opens an interactive visualizer window.

---

## Training

Training loop is in `main.py` and calls:

* `simulate_episode(agent, steps=..., window=..., train=True, online_update=True)`

Key parameters you can tune in `main.py`:

* `episodes` — number of episodes
* `steps` — steps per episode
* `window` — how many past prices are observed

RL agent hyperparameters (see `ReinforceAgent` in `train.py`):

* `lr` — learning rate
* `gamma` — discount factor
* `epsilon_*` — exploration schedule
* `baseline_beta` — baseline smoothing (variance reduction)

---

## Evaluation

Evaluation utilities live in `evaluate.py`.

### Metrics

For each agent, evaluation computes:

* Total return (%)
* Sharpe ratio (using log returns)
* Max drawdown (%)
* Final capital

### Monte Carlo evaluation

`evaluate_strategies(...)` runs multiple simulations with different seeds and reports averaged metrics (mean + some stability stats like standard deviation of returns).

In `main.py`, the code builds a list of baselines and compares them against the RL agent.

---

## Visualization

`visualize.py` provides a Matplotlib animation:

* **Top right**: price curve + trade markers of one selected agent
* **Bottom right**: equity curves for all agents
* **Left**: table with portfolio states (cash / shares / capital)

Controls:

* `Space` — toggle autoplay / pause
* `Right arrow` — advance one step (when paused)

---

## Configuration tips

* Start with **synthetic** mode to verify learning dynamics quickly.
* Increase `window` (e.g., 20 → 50) if you want the agent to infer regimes better from history.
* If learning is unstable:

  * lower `lr`
  * increase the number of episodes
  * increase trade penalties slightly (reduces churning)

---

## Roadmap

Planned improvements (aligned with the original idea):

1. Use an **MLP policy** with richer inputs:

   * min / max / mean over last `n` steps (e.g., 50)
   * short history window of last `m` steps (e.g., 10)
2. Add an explicit **train/eval split** for historical mode
3. Logging:

   * reproducible seeds everywhere
   * saved runs (CSV/JSON)
   * training curves (reward, return, Sharpe)
4. Optional: implement PPO/SAC via a standard RL library once the environment API is stabilized

---

### Notes

This is a research/learning project, not financial advice and not a production trading system.
