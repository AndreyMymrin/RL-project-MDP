import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agents import RandomAgent, BuyAndHoldAgent
from agents import Bill, BollingerBandsAgent, Mark, MovingAverageCrossoverAgent, RSIAgent
from market import Market
from train import ReinforceAgent, evaluate_baselines, simulate_episode


def calculate_metrics(history):
    values = np.array(history)
    # Защита от отрицательных значений или нулей (если агент все слил)
    values = np.clip(values, 1e-7, None)
    
    # 1. Total Return (%) - остается как есть
    total_return = ((values[-1] - values[0]) / values[0]) * 100
    
    # 2. Daily Returns
    # Используем логарифмические доходности для более точного Шарпа
    log_returns = np.diff(np.log(values))
    
    # 3. Sharpe Ratio
    std = np.std(log_returns)
    if std > 1e-9:
        # Среднее лог-доходностей / стандартное отклонение * корень из времени
        # Для лог-доходностей это более устойчивая оценка
        sharpe = (np.mean(log_returns) / std) * np.sqrt(252)
    else:
        sharpe = 0.0
        
    # 4. Max Drawdown (%)
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max
    max_drawdown = np.min(drawdowns) * 100
    
    return total_return, sharpe, max_drawdown

def evaluate_strategies(
    market_class, 
    rl_agent, 
    baselines, 
    steps=500, 
    window=20, 
    seed=42, 
    start_price=10.0
):
    """
    Сравнивает RL агента с базовыми стратегиями на одних и тех же данных.
    """
    print(f"--- Started evaluation on {steps} steps (Seed: {seed}) ---")
    
    # 1. Инициализация рынка
    # Фиксируем seed, чтобы все агенты торговали на ОДИНАКОВОМ рынке
    rng = np.random.default_rng(seed)
    np_random_state = np.random.get_state() # Сохраняем состояние, чтобы не сломать глобальный рандом
    
    market = market_class(start_price=start_price, window=window)
    
    # 2. Подготовка агентов
    all_agents = [rl_agent] + baselines
    histories = {agent.name: [agent.capital(start_price)] for agent in all_agents}
    
    # Сбрасываем кэш агентов перед тестом
    for agent in all_agents:
        if hasattr(agent, 'reset'):
            agent.reset(cash=1000.0, stocks=0.0)
        else:
            agent.cash = 1000.0
            agent.stocks = 0.0

    # 3. Цикл симуляции
    for step in range(steps):
        obs = market.get_obs()
        price = market.price
        
        for agent in all_agents:
            # RL агент обычно требует explore=False на тесте
            if agent.name == rl_agent.name:
                # Проверка сигнатуры метода, чтобы не упало с ошибкой
                try:
                    agent.act(obs, explore=False)
                except TypeError:
                    agent.act(obs) # Если у агента нет параметра explore
            else:
                agent.act(obs)
            
            # Записываем состояние портфеля
            val = agent.capital(price)
            histories[agent.name].append(val)
            
        market.step()
    
    # Возвращаем рандом в исходное состояние
    np.random.set_state(np_random_state)

    # 4. Сбор результатов в таблицу
    results_data = []
    for agent_name, history in histories.items():
        ret, sharpe, mdd = calculate_metrics(history)
        results_data.append({
            "Agent": agent_name,
            "Return (%)": round(ret, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Max Drawdown (%)": round(mdd, 2),
            "Final Capital ($)": round(history[-1], 2)
        })
        
    df = pd.DataFrame(results_data)
    df.set_index("Agent", inplace=True)
    
    # Сортируем по Доходности
    return df.sort_values(by="Return (%)", ascending=False), histories

def plot_equity_curves(histories):
    """Рисует графики роста капитала всех агентов."""
    plt.figure(figsize=(12, 6))
    
    # Цветовая схема
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for (name, history), color in zip(histories.items(), colors):
        # Выделяем RL агента жирной линией
        if "REINFORCE" in name or "DQN" in name or "RL" in name:
            plt.plot(history, label=name, linewidth=3, color='black', alpha=0.9)
        else:
            plt.plot(history, label=name, linewidth=1.5, color=color, alpha=0.7, linestyle='--')
            
    plt.title("Сравнение эффективности стратегий (Equity Curves)")
    plt.xlabel("Шаги (Time)")
    plt.ylabel("Капитал ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()