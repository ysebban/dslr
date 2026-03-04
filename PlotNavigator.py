"""
PlotNavigator.py

Small class helpers to navigate and render various plots

left/right arrows -> move across plots

r -> redraw current plot

q or [esc] -> quit
"""

from __future__ import annotations

import matplotlib.pyplot as plt


class PlotNavigator:

    def __init__(self, items, *, render, make_figure, title="Plot"):
        if not items:
            raise ValueError("PlotNavigator requires at least one item.")
        self.items = items
        self.render = render
        self.make_figure = make_figure
        self.title = title
        self.index = 0
        self.figure = None
        self.axes = None

    def show(self, start_index=0):
        self.index = max(0, min(start_index, len(self.items) - 1))

        self.figure, self.axes = self.make_figure()
        self.figure.canvas.mpl_connect("key_release_event", self._on_key)

        self._draw()
        plt.show()

    def _draw(self):
        item = self.items[self.index]

        for ax in self.axes:
            ax.clear()

        # This is acutally a SUB title
        title = self.render(item, self.axes, self.index, len(self.items))

        if title:
            self.figure.suptitle(
                f"{self.title} — {self.index + 1}/{len(self.items)}: {title}"
            )
        else:
            self.figure.suptitle(
                f"{self.title} — {self.index + 1}/{len(self.items)}"
            )

        plt.draw()

    def _on_key(self, event):
        if event.key == "right" and self.index < len(self.items) - 1:
            self.index += 1
            self._draw()
        elif event.key == "left" and self.index > 0:
            self.index -= 1
            self._draw()
        elif event.key in ("r",):
            self._draw()
        elif event.key in ("q", "escape"):
            plt.close(self.figure)
