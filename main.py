# import numpy as np

# from agents import Bill, BollingerBandsAgent, Mark, MovingAverageCrossoverAgent, RSIAgent
# from market import Market
# from train import ReinforceAgent, evaluate_baselines, simulate_episode
# from visualize import Visualizer


# def main():
#     episodes = 400
#     steps = 200
#     window = 20

#     agent = ReinforceAgent(window=window, lr=0.02, gamma=0.98, seed=42)
#     for ep in range(episodes):
#         simulate_episode(agent, steps=steps, window=window, seed=ep, train=True, online_update=True)

#     eval_runs = 30
#     rl_caps = []
#     baseline_caps = {}
#     for i in range(eval_runs):
#         rl_caps.append(simulate_episode(agent, steps=steps, window=window, seed=1000 + i, train=False))
#         res = evaluate_baselines(steps=steps, window=window, seed=1000 + i)
#         for name, cap in res.items():
#             baseline_caps.setdefault(name, []).append(cap)

#     print("REINFORCE avg:", float(np.mean(rl_caps)))
#     for name in sorted(baseline_caps.keys()):
#         print(f"{name} avg:", float(np.mean(baseline_caps[name])))

#     max_frames = 100
#     m = Market(start_price=5.0, window=window)
#     agent.reset(cash=1000.0, stocks=0.0)
#     ags = [
#         Mark(1000.0, 0.0),
#         Bill(1000.0, 0.0),
#         MovingAverageCrossoverAgent(1000.0, 0.0),
#         RSIAgent(1000.0, 0.0),
#         BollingerBandsAgent(1000.0, 0.0),
#         agent,
#     ]
#     v = Visualizer(m, ags, max_frames)
#     plt_show = True
#     if plt_show:
#         import matplotlib.pyplot as plt
#         try:
#             plt.show()
#         except KeyboardInterrupt:
#             pass
#         finally:
#             plt.close("all")


# if __name__ == "__main__":
#     main()

import numpy as np
import matplotlib.pyplot as plt
from agents import RandomAgent, BuyAndHoldAgent
from agents import Bill, BollingerBandsAgent, Mark, MovingAverageCrossoverAgent, RSIAgent
from market import Market
from train import ReinforceAgent, ReinforceAgentMLP, simulate_episode
from visualize import Visualizer
from evaluate import evaluate_baselines, evaluate_strategies, plot_equity_curves

def main():
    episodes = 600
    steps = 400
    window = 20
    use_mlp = False

    print(f"Started training for ({episodes} eposodes)...")
    if use_mlp:
        agent = ReinforceAgentMLP(window=window, lr=0.001, gamma=0.98, seed=42)
    else:
        agent = ReinforceAgent(window=window, lr=0.001, gamma=0.98, seed=42)
    
    for ep in range(episodes):
        simulate_episode(agent, steps=steps, window=window, seed=ep, train=True, online_update=True)
        if (ep + 1) % 50 == 0:
            print(f"Episod {ep + 1}/{episodes} is finished")

    print("Final Evaluation started ...")

    baselines = [
        Mark(1000.0, 0.0),
        Bill(1000.0, 0.0),
        MovingAverageCrossoverAgent(1000.0, 0.0),
        RSIAgent(1000.0, 0.0),
        BollingerBandsAgent(1000.0, 0.0),
        RandomAgent(1000.0, 0.0, seed=123),
        BuyAndHoldAgent(1000.0, 0.0),
    ]

    df_results, histories = evaluate_strategies(
        Market, 
        agent, 
        baselines, 
        steps=steps, 
        # seed=42, 
        start_price=5.0
    )

    print("\nResults:")
    print(df_results)

    plot_equity_curves(histories)

    print("\nVisualizer started ...")
    max_frames = 100
    m = Market(start_price=5.0, window=window)
    
    agent.reset(cash=1000.0, stocks=0.0)
    for b in baselines:
        b.cash = 1000.0
        b.stocks = 0.0
        if hasattr(b, 'first_step'): b.first_step = True

    ags = [agent] + baselines
    
    v = Visualizer(m, ags, max_frames)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        plt.close("all")

if __name__ == "__main__":
    main()