"""
PlotNavigator.py

Interactive plot navigation utilities.

The main abstraction of the project is a "plot item":
- A plot item is any object that can be rendered as a plot.
- A render function is responsible for drawing the item.
- A figure factory creates the matplotlib figure.
- An axes factory creates the axes inside that figure.
- Keyboard shortcuts allow navigating between items.

Navigation model:
- `index` is the currently displayed item.
- `pending_index` is the latest requested item.
- Key events update navigation state.
- Rendering is coalesced through Tk `after()`.
- A redraw is never performed directly inside a raw key callback.

Rendering model:
- The figure is created once.
- The axes are created once and kept inside the navigator.
- On navigation, we reuse the same figure/axes.
- We clear only the existing axes content before rendering again.
"""

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

    Expected collaborators:
    - `make_figure()` returns a matplotlib Figure
    - `make_axes(figure)` creates and returns the axes for that figure
    - `render(item, axes, index, total)` draws the current item
    """

    def __init__(self, items, *, render, make_figure, make_axes, title="Plot"):
        """
        Initialize a plot navigator.
        Args:
            items: Sequence of items that can be plotted.
            render: Function responsible for drawing one item.
            make_figure: Function creating the matplotlib figure.
            make_axes: Function creating the axes inside that figure.
            title: Base title displayed at the top of the figure.
        Returns:
            A PlotNavigator instance.
        Raises:
            ValueError: If `items` is empty.
        """
        if not items:
            raise ValueError("PlotNavigator requires at least one item.")

        self.items = items
        self.render = render
        self.make_figure = make_figure
        self.make_axes = make_axes
        self.title = title

        # Current displayed item.
        self.index = 0
        # Latest requested item.
        self.pending_index = 0
        # Matplotlib objects kept alive during the whole session.
        self.figure = None
        self.axes = None
        # Drawing / scheduling state.
        self.is_drawing = False
        self._after_id = None
        # Key state.
        self.key_held = False
        self.held_key = None

    def show(self, start_index=0):
        """
        Start the interactive plot navigator.
        Args:
            start_index: Index of the first item to display.
        """
        start = max(0, min(start_index, len(self.items) - 1))
        self.index = start
        self.pending_index = start

        if self.figure is None:
            self.figure = self.make_figure()
            self.axes = self.make_axes(self.figure)

            canvas = self.figure.canvas
            canvas.mpl_connect("key_press_event", self._on_key_press)
            canvas.mpl_connect("key_release_event", self._on_key_release)
            canvas.mpl_connect("close_event", self._on_close)

        self._draw()
        plt.show()

    def _get_tk_widget(self):
        """
        Return the Tk widget backing the canvas.

        Returns:
            The Tk widget associated with the current figure canvas.

        Raises:
            RuntimeError: If the figure is missing.
        """
        if self.figure is None:
            raise RuntimeError("Figure is not initialized.")

        canvas = self.figure.canvas
        if not hasattr(canvas, "get_tk_widget"):
            raise RuntimeError(
                "PlotNavigator requires a Tk-based Matplotlib backend "
                "to use Tk.after()."
            )

        return canvas.get_tk_widget()

    def _request_index(self, delta):
        """
        Update the pending index by `delta`, clamped to valid bounds.

        Args:
            delta: Signed navigation step.

        Returns:
            True if the pending index changed, False otherwise.
        """
        new_index = self.pending_index + delta
        new_index = max(0, min(new_index, len(self.items) - 1))

        if new_index == self.pending_index:
            return False

        self.pending_index = new_index
        return True

    def _schedule_draw(self, delay_ms=10):
        """
        Schedule a coalesced redraw with Tk `after()`.
        Args:
            delay_ms: Delay in milliseconds before running the draw callback.
        """
        if self.figure is None:
            return
        if self._after_id is not None:
            return

        widget = self._get_tk_widget()
        self._after_id = widget.after(delay_ms, self._run_scheduled_draw)

    def _cancel_scheduled_draw(self):
        """
        Cancel a pending Tk `after()` callback if one exists.
        """
        if self.figure is None or self._after_id is None:
            return

        try:
            widget = self._get_tk_widget()
            widget.after_cancel(self._after_id)
        except Exception:
            pass
        finally:
            self._after_id = None

    def _run_scheduled_draw(self):
        """
        Execute a scheduled redraw.
        Behavior:
            - Clear the current scheduled callback id
            - If a draw is already running, reschedule once later
            - Otherwise, commit the latest pending index and render it
        """
        self._after_id = None

        if self.figure is None:
            return

        if self.is_drawing:
            self._schedule_draw()
            return

        if self.pending_index != self.index:
            self.index = self.pending_index

        self._draw()

    def _clear_axes(self):
        """
        Clear the current axes content without rebuilding the whole figure.
        """
        if self.axes is None:
            return

        if isinstance(self.axes, (list, tuple)):
            for ax in self.axes:
                if hasattr(ax, "flat"):
                    for sub_ax in ax.flat:
                        sub_ax.clear()
                else:
                    ax.clear()
            return
        # Single axes object.
        self.axes.clear()

    def _draw(self):
        """
        Render the current plot item.
        """
        if self.figure is None or self.axes is None:
            return
        if self.is_drawing:
            return

        try:
            self.is_drawing = True
            item = self.items[self.index]
            items_len = len(self.items)

            # Reuse axes instead of rebuilding the whole figure each time.
            self._clear_axes()

            title = self.render(item, self.axes, self.index, items_len)

            if title:
                self.figure.suptitle(
                    f"{self.title} — {self.index + 1}/{items_len}: {title}"
                )
            else:
                self.figure.suptitle(
                    f"{self.title} — {self.index + 1}/{items_len}"
                )

            self.figure.canvas.draw()

        finally:
            self.is_drawing = False

    def _on_key_press(self, event):
        """
        Handle key press events.
        Supported keys:
            - right: move to next plot
            - left: move to previous plot
            - r: redraw current plot
            - q or escape: quit
        """
        key = event.key
        if key is None or self.key_held is True:
            return

        if key in ("q", "escape"):
            self._cancel_scheduled_draw()
            plt.close(self.figure)
            return

        if key == "r":
            self._schedule_draw()
            return

        if key == "right":
            self.key_held = True
            self.held_key = "right"
            if self._request_index(+1):
                self._schedule_draw()
            return

        if key == "left":
            self.key_held = True
            self.held_key = "left"
            if self._request_index(-1):
                self._schedule_draw()
            return

    def _on_key_release(self, event):
        """
        Handle key release events.

        Notes:
            Release events clear held-key state and ensure the latest pending
            item is rendered if it differs from the currently displayed one.
        """
        key = event.key
        if key is None or self.key_held is False:
            return

        if key in ("left", "right"):
            if self.held_key == key:
                self.key_held = False
                self.held_key = None

            if self.pending_index != self.index:
                self._schedule_draw()

    def _on_close(self, event):
        """
        Cleanup scheduled callbacks when the figure is closed.
        """
        self._cancel_scheduled_draw()
        self.figure = None
        self.axes = None
