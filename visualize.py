import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection


class Visualizer:
    def __init__(self, market, agents, frames=None):
        self.market = market
        self.agents = agents
        self.auto = True
        self.states = ["Growing", "Depressing", "Volatile"]

        self.fig, (self.ax_info, self.ax_graph) = plt.subplots(
            1, 2, gridspec_kw={"width_ratios": [1, 3]}, figsize=(12, 6)
        )
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.anim = FuncAnimation(self.fig, self.draw, interval=500, cache_frame_data=False, frames=frames)

    def on_key(self, event):
        if event.key == " ":
            self.auto = not self.auto
        elif event.key == "right":
            if not self.auto:
                self.sim_step()
                self.draw(None)
                self.fig.canvas.draw()

    def sim_step(self):
        self.market.step()
        obs = self.market.get_obs()
        step_index = len(self.market.history) - 1
        for a in self.agents:
            a.act(obs, step_index=step_index)

    def draw(self, frame):
        if self.auto and frame is not None:
            self.sim_step()

        self.ax_graph.clear()
        self.ax_info.clear()
        self.ax_info.axis("off")

        hist = self.market.history
        if len(hist) > 1:
            x = np.arange(len(hist))
            pts = np.array([x, hist]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            cols = ["green" if hist[i] >= hist[i - 1] else "red" for i in range(1, len(hist))]
            lc = LineCollection(segs, colors=cols, linewidths=2)
            self.ax_graph.add_collection(lc)
            self.ax_graph.scatter(x, hist, color="black", s=10, zorder=3)
            self.ax_graph.set_xlim(0, max(10, len(hist)))
            self.ax_graph.set_ylim(0, max(hist) * 1.1 + 1)
        elif len(hist) == 1:
            self.ax_graph.scatter([0], hist, color="black", s=10)

        self.ax_graph.set_title(f"Market: {self.states[self.market.state]}")

        for a in self.agents:
            if not hasattr(a, "trade_history"):
                continue
            buys = [(i, p) for (i, p, side) in a.trade_history if side == "buy" and i < len(hist)]
            sells = [(i, p) for (i, p, side) in a.trade_history if side == "sell" and i < len(hist)]
            if buys:
                bx, by = zip(*buys)
                self.ax_graph.scatter(bx, by, marker="^", s=60, color="green", edgecolor="black", zorder=4, label="RL buy")
            if sells:
                sx, sy = zip(*sells)
                self.ax_graph.scatter(sx, sy, marker="v", s=60, color="red", edgecolor="black", zorder=4, label="RL sell")
            if buys or sells:
                self.ax_graph.legend(loc="upper left")
            break

        cells = []
        for a in self.agents:
            cells.append([a.name, f"{a.stocks:.2f}", f"{a.cash:.2f}", f"{a.capital(hist[-1]):.2f}"])

        tab = self.ax_info.table(
            cellText=cells,
            colLabels=["Name", "Shares", "Money", "Capital"],
            loc="center",
            cellLoc="center",
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(8)
        tab.scale(1.5, 3)

        control_text = "Controls:\nSpace - Pause / Auto\nRight arrow - One step (during pause)"
        self.ax_info.text(
            0.5,
            0.1,
            control_text,
            ha="center",
            va="center",
            transform=self.ax_info.transAxes,
            fontsize=9,
            color="gray",
        )
