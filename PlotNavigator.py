"""
PlotNavigator.py

Interactive plot navigation utilities.

The main abstraction of the project is a "plot item":
- A plot item is any object that can be rendered as a plot.
- A render function is responsible for drawing the item.
- A figure factory creates the matplotlib figure.
- An axes factory creates the axes inside that figure.
- Keyboard shortcuts allow navigating between items.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


class PlotNavigator:
    """
    Plot navigation utilities for this project.

    This class is used to display a sequence of plots interactively.

    The recommended usage is:
        navigator = PlotNavigator(
            items,
            render=render_function,
            make_figure=figure_factory,
            make_axes=axes_factory,
            title="My plots"
        )

        navigator.show()
    """

    def __init__(self, items, *, render, make_figure, make_axes, title="Plot"):
        """
        Initialize a plot navigator.

        Args:
            items: Sequence of items that can be plotted.
            render: Function responsible for drawing an item.
            make_figure: Function creating the matplotlib figure.
            make_axes: Function creating the axes inside a figure.
            title: Base title displayed at the top of the figure.

        Returns:
            A PlotNavigator instance.

        Notes:
            Raises ValueError if `items` is empty.
        """
        if not items:
            raise ValueError("PlotNavigator requires at least one item.")

        self.items = items
        self.render = render
        self.make_figure = make_figure
        self.make_axes = make_axes
        self.title = title
        self.index = 0
        self.figure = None
        self.axes = None
        self.is_drawing = False
        self.redraw_scheduled = False
        self._timer = None

    def show(self, start_index=0):
        """
        Start the interactive plot navigator.

        Args:
            start_index: Index of the first item to display.

        Notes:
            The starting index is clamped within valid bounds.
            This method opens a matplotlib window and waits for keyboard input.
        """
        self.index = max(0, min(start_index, len(self.items) - 1))

        if self.figure is None:
            self.figure = self.make_figure()
            self.figure.canvas.mpl_connect("key_press_event", self._on_key)

        self._draw()
        # plt.show()

    def _draw(self):
        """
        Render the current plot item.

        Behavior:
            - Clears the figure
            - Recreates fresh axes in the same figure
            - Calls the render function
            - Updates the figure title
        """
        if self.is_drawing:
            return

        self.is_drawing = True
        try:
            item = self.items[self.index]

            self.figure.clf()
            self.axes = self.make_axes(self.figure)

            title = self.render(item, self.axes, self.index, len(self.items))

            if title:
                self.figure.suptitle(
                    f"{self.title} — {self.index + 1}/{len(self.items)}: {title}"
                )
            else:
                self.figure.suptitle(
                    f"{self.title} — {self.index + 1}/{len(self.items)}"
                )

            self.figure.canvas.draw()
            
        finally:
            self.is_drawing = False
            plt.show()

    def _schedule_draw(self):
        """
        Schedule a redraw after the current GUI callback finishes.

        Notes:
            This avoids redrawing directly inside the key event callback.
        """
        if self.redraw_scheduled:
            return
        if self.figure is None:
            return

        self.redraw_scheduled = True

        timer = self.figure.canvas.new_timer(interval=1)
        timer.single_shot = True
        timer.add_callback(self._run_scheduled_draw)
        timer.start()
        self._timer = timer

    def _run_scheduled_draw(self):
        """
        Execute a scheduled redraw.
        """
        self.redraw_scheduled = False
        self._draw()

    def _on_key(self, event):
        """
        Handle keyboard navigation.

        Supported keys:
            - right: next plot
            - left: previous plot
            - r: redraw current plot
            - q or escape: quit
        """
        if self.is_drawing:
            return

        if event.key == "right" and self.index < len(self.items) - 1:
            self.index += 1
            self._schedule_draw()
        elif event.key == "left" and self.index > 0:
            self.index -= 1
            self._schedule_draw()
        elif event.key == "r":
            self._schedule_draw()
        elif event.key in ("q", "escape"):
            plt.close("all")
