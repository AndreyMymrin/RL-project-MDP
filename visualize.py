# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.collections import LineCollection
#
#
# class Visualizer:
#     def __init__(self, market, agents, frames=None):
#         self.market = market
#         self.agents = agents
#         self.auto = True
#         self.states = ["Growing", "Depressing", "Volatile"]
#
#         self.fig, (self.ax_info, self.ax_graph) = plt.subplots(
#             1, 2, gridspec_kw={"width_ratios": [1, 3]}, figsize=(12, 6)
#         )
#         self.fig.canvas.mpl_connect("key_press_event", self.on_key)
#         self.anim = FuncAnimation(self.fig, self.draw, interval=500, cache_frame_data=False, frames=frames)
#
#     def on_key(self, event):
#         if event.key == " ":
#             self.auto = not self.auto
#         elif event.key == "right":
#             if not self.auto:
#                 self.sim_step()
#                 self.draw(None)
#                 self.fig.canvas.draw()
#
#     def sim_step(self):
#         self.market.step()
#         obs = self.market.get_obs()
#         step_index = len(self.market.history) - 1
#         for a in self.agents:
#             a.act(obs, step_index=step_index)
#
#     def draw(self, frame):
#         if self.auto and frame is not None:
#             self.sim_step()
#
#         self.ax_graph.clear()
#         self.ax_info.clear()
#         self.ax_info.axis("off")
#
#         hist = self.market.history
#         if len(hist) > 1:
#             x = np.arange(len(hist))
#             pts = np.array([x, hist]).T.reshape(-1, 1, 2)
#             segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
#             cols = ["green" if hist[i] >= hist[i - 1] else "red" for i in range(1, len(hist))]
#             lc = LineCollection(segs, colors=cols, linewidths=2)
#             self.ax_graph.add_collection(lc)
#             self.ax_graph.scatter(x, hist, color="black", s=10, zorder=3)
#             self.ax_graph.set_xlim(0, max(10, len(hist)))
#             self.ax_graph.set_ylim(0, max(hist) * 1.1 + 1)
#         elif len(hist) == 1:
#             self.ax_graph.scatter([0], hist, color="black", s=10)
#
#         self.ax_graph.set_title(f"Market: {self.states[self.market.state]}")
#
#         for a in self.agents:
#             if not hasattr(a, "trade_history"):
#                 continue
#             buys = [(i, p) for (i, p, side) in a.trade_history if side == "buy" and i < len(hist)]
#             sells = [(i, p) for (i, p, side) in a.trade_history if side == "sell" and i < len(hist)]
#             if buys:
#                 bx, by = zip(*buys)
#                 self.ax_graph.scatter(bx, by, marker="^", s=60, color="green", edgecolor="black", zorder=4, label="RL buy")
#             if sells:
#                 sx, sy = zip(*sells)
#                 self.ax_graph.scatter(sx, sy, marker="v", s=60, color="red", edgecolor="black", zorder=4, label="RL sell")
#             if buys or sells:
#                 self.ax_graph.legend(loc="upper left")
#             break
#
#         cells = []
#         for a in self.agents:
#             cells.append([a.name, f"{a.stocks:.2f}", f"{a.cash:.2f}", f"{a.capital(hist[-1]):.2f}"])
#
#         tab = self.ax_info.table(
#             cellText=cells,
#             colLabels=["Name", "Shares", "Money", "Capital"],
#             loc="center",
#             cellLoc="center",
#         )
#         tab.auto_set_font_size(False)
#         tab.set_fontsize(8)
#         tab.scale(1.5, 3)
#
#         control_text = "Controls:\nSpace - Pause / Auto\nRight arrow - One step (during pause)"
#         self.ax_info.text(
#             0.5,
#             0.1,
#             control_text,
#             ha="center",
#             va="center",
#             transform=self.ax_info.transAxes,
#             fontsize=9,
#             color="gray",
#         )

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


class Visualizer:
    """
    Слева:
      - таблица с состоянием агентов (акции/деньги/капитал)

    Справа:
      - верхний график: цена + сделки одного выбранного агента
      - нижний график: кривые капитала всех агентов (сравнение эффективности)

    Дополнительно:
      - бледная раскраска фона по скрытому режиму рынка (Growing/Depressing/Volatile)
      - сделки рисуются как вертикальные стрелки:
           длина стрелки ~ объёму сделки (trade_value)
    """

    def __init__(
        self,
        market,
        agents,
        frames=None,
        interval=400,
        regime_alpha=0.06,
        arrow_min_len=2.0,
        arrow_max_len=15.0,
        trade_agent_name=None,     # если None — берём первого агента с trade_history
        show_price_points=True,
        table_bbox=(-0.30, 0.18, 1.12, 0.78),
        table_fontsize=8
    ):
        self.market = market
        self.agents = agents
        self.auto = True

        self.states = ["Growing", "Depressing", "Volatile"]

        # разные режимы рынка
        self.regime_colors = {
            0: (0.3, 1.0, 0.3, regime_alpha),  # Growing
            1: (1.0, 0.3, 0.3, regime_alpha),  # Depressing
            2: (1.0, 1.0, 0.3, regime_alpha),  # Volatile
        }

        # Длина стрелок в единицах цены: объём -> arrow_min_len..arrow_max_len
        self.arrow_min_len = float(arrow_min_len)
        self.arrow_max_len = float(arrow_max_len)

        # Какого агента рисуем на верхнем графике (сделки)
        self.trade_agent_name = trade_agent_name

        # Показывать ли точки цены
        self.show_price_points = bool(show_price_points)

        # История скрытого режима рынка (для раскраски фона)
        self.state_hist = [int(self.market.state)]

        # История капитала каждого агента (для нижнего графика)
        self.capital_hist = {a: [a.capital(self.market.price)] for a in self.agents}

        # Параметры таблицы
        self.table_bbox = table_bbox
        self.table_fontsize = table_fontsize

        # ---------- Разметка фигуры ----------
        self.fig = plt.figure(figsize=(14, 7))

        gs = self.fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.8, 3.05])

        self.ax_info = self.fig.add_subplot(gs[0, 0])

        gs_right = gs[0, 1].subgridspec(nrows=2, ncols=1, height_ratios=[2.0, 1.0], hspace=0.12)
        self.ax_price = self.fig.add_subplot(gs_right[0, 0])
        self.ax_cap = self.fig.add_subplot(gs_right[1, 0], sharex=self.ax_price)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # Анимация
        self.anim = FuncAnimation(
            self.fig,
            self.draw,
            interval=interval,
            cache_frame_data=False,
            frames=frames
        )

    # -----------------------------
    # Управление: пробел — пауза/авто, стрелка вправо — шаг в паузе
    # -----------------------------
    def on_key(self, event):
        if event.key == " ":
            self.auto = not self.auto
        elif event.key == "right":
            if not self.auto:
                self.sim_step()
                self.draw(None)
                self.fig.canvas.draw()

    # -----------------------------
    # Один шаг симуляции: рынок -> агенты -> обновить истории
    # -----------------------------
    def sim_step(self):
        self.market.step()
        obs = self.market.get_obs()
        step_index = len(self.market.history) - 1

        # Агенты делают действия (и могут писать trade_history)
        for a in self.agents:
            a.act(obs, step_index=step_index)

        # Запоминаем режим рынка (для фона)
        self.state_hist.append(int(self.market.state))

        # Запоминаем капитал каждого агента
        price = float(self.market.price)
        for a in self.agents:
            self.capital_hist[a].append(a.capital(price))

    # -----------------------------
    # Покраска фона по режимам рынка
    # -----------------------------
    def _shade_regimes(self, ax, x_max):
        if len(self.state_hist) < 2:
            return

        start = 0
        for i in range(1, len(self.state_hist)):
            if self.state_hist[i] != self.state_hist[start]:
                s = self.state_hist[start]
                ax.axvspan(start, i, color=self.regime_colors[s])
                start = i

        s = self.state_hist[start]
        ax.axvspan(start, x_max, color=self.regime_colors[s])

    # -----------------------------
    # Нормализация массива в [0,1]
    # -----------------------------
    def _normalize01(self, arr):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            return arr
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        if (mx - mn) < 1e-12:
            return np.zeros_like(arr)
        return np.clip((arr - mn) / (mx - mn), 0.0, 1.0)

    # -----------------------------
    # Объём сделки -> длина стрелки (в единицах цены)
    # -----------------------------
    def _len_from_trade(self, trade_values):
        z = self._normalize01(trade_values)
        return self.arrow_min_len + z * (self.arrow_max_len - self.arrow_min_len)

    # -----------------------------
    # Выбор агента, чьи сделки рисуем на верхнем графике
    # -----------------------------
    def _pick_trade_agent(self):
        if self.trade_agent_name is not None:
            for a in self.agents:
                if getattr(a, "name", None) == self.trade_agent_name and hasattr(a, "trade_history"):
                    return a
            return None

        # Если имя не задано — берём первого агента, у которого есть trade_history
        for a in self.agents:
            if hasattr(a, "trade_history"):
                return a
        return None

    # -----------------------------
    # Основной рендер одного кадра
    # -----------------------------
    def draw(self, frame):
        if self.auto and frame is not None:
            self.sim_step()

        # Очистка осей
        self.ax_price.clear()
        self.ax_cap.clear()
        self.ax_info.clear()
        self.ax_info.axis("off")

        hist = self.market.history
        n = len(hist)
        x = np.arange(n)

        # -----------------------------
        # Верхний правый график: цена + сделки
        # -----------------------------
        self._shade_regimes(self.ax_price, x_max=max(1, n - 1))

        if n > 1:
            # Линия цены сегментами: зелёный если шаг вверх, красный если вниз
            pts = np.array([x, hist]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            cols = ["green" if hist[i] >= hist[i - 1] else "red" for i in range(1, n)]
            lc = LineCollection(segs, colors=cols, linewidths=2, zorder=2)
            self.ax_price.add_collection(lc)

            # Чёрные точки цены (если включено)
            if self.show_price_points:
                self.ax_price.scatter(x, hist, color="black", s=10, zorder=3)

            self.ax_price.set_xlim(0, max(10, n))
            self.ax_price.set_ylim(0, max(hist) * 1.12 + 1)
        else:
            if self.show_price_points:
                self.ax_price.scatter([0], hist, color="black", s=10)

        self.ax_price.set_title(f"Market: {self.states[int(self.market.state)]}")

        # -----------------------------
        # Сделки выбранного агента: стрелки вверх/вниз, длина ~ объём
        # -----------------------------
        trade_agent = self._pick_trade_agent()

        if trade_agent is not None and hasattr(trade_agent, "trade_history"):
            buys = []
            sells = []
            buy_tv = []
            sell_tv = []

            # trade_history поддерживает форматы:
            #   (i, price, side) или (i, price, side, trade_value, confidence)
            # confidence здесь игнорируем
            for item in trade_agent.trade_history:
                if len(item) == 3:
                    i, p, side = item
                    tv = 1.0
                else:
                    i, p, side, tv, _conf = item

                if i >= n:
                    continue

                if side == "buy":
                    buys.append((int(i), float(p)))
                    buy_tv.append(float(tv))
                elif side == "sell":
                    sells.append((int(i), float(p)))
                    sell_tv.append(float(tv))

            buy_len = self._len_from_trade(buy_tv) if len(buy_tv) else np.array([])
            sell_len = self._len_from_trade(sell_tv) if len(sell_tv) else np.array([])

            green = (0.0, 0.75, 0.0)
            red = (0.85, 0.0, 0.0)

            # BUY: стрелка вверх
            for (xi, yi), L in zip(buys, buy_len):
                self.ax_price.annotate(
                    "",
                    xy=(xi, yi + float(L)),
                    xytext=(xi, yi),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=green,
                        lw=2.2
                    ),
                    zorder=6
                )

            # SELL: стрелка вниз
            for (xi, yi), L in zip(sells, sell_len):
                self.ax_price.annotate(
                    "",
                    xy=(xi, yi - float(L)),
                    xytext=(xi, yi),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=red,
                        lw=2.2
                    ),
                    zorder=6
                )

            handles = [
                Line2D([0], [0], color=green, lw=2.2, marker=">", markersize=8, label=f"{trade_agent.name}: BUY"),
                Line2D([0], [0], color=red, lw=2.2, marker=">", markersize=8, label=f"{trade_agent.name}: SELL"),
            ]
            self.ax_price.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.9)

        # -----------------------------
        # Нижний правый график: капитал всех агентов
        # -----------------------------
        self._shade_regimes(self.ax_cap, x_max=max(1, n - 1))

        for a in self.agents:
            series = self.capital_hist.get(a, None)
            if series is None:
                continue
            t = np.arange(len(series))
            self.ax_cap.plot(t, series, linewidth=2, label=a.name)

        self.ax_cap.set_ylabel("Capital")
        self.ax_cap.set_xlabel("Step")
        self.ax_cap.grid(True, alpha=0.25)
        self.ax_cap.legend(loc="upper left", fontsize=9, framealpha=0.9)

        # -----------------------------
        # Левая часть: таблица состояний агентов
        # -----------------------------
        price_now = float(hist[-1])
        cells = []
        for a in self.agents:
            cells.append([
                a.name,
                f"{a.stocks:.2f}",
                f"{a.cash:.2f}",
                f"{a.capital(price_now):.2f}"
            ])

        tab = self.ax_info.table(
            cellText=cells,
            colLabels=["Name", "Shares", "Money", "Capital"],
            loc="center",
            cellLoc="center",
            colLoc="center",
            bbox=list(self.table_bbox),
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(self.table_fontsize)
        tab.scale(1.15, 1.35)

        for (row, col), cell in tab.get_celld().items():
            if row == 0:
                cell.set_facecolor("#f0f0f0")
                cell.set_text_props(weight="bold")
            else:
                cell.set_facecolor("#fbfbfb" if (row % 2 == 0) else "white")
            cell.set_edgecolor("#333333")
            cell.set_linewidth(1.0)

        control_text = "Controls:\nSpace — Pause/Auto\nRight arrow — One step (during pause)"
        self.ax_info.text(
            0.5, 0.06,
            control_text,
            ha="center",
            va="center",
            transform=self.ax_info.transAxes,
            fontsize=9,
            color="gray"
        )
