import tkinter as tk
from functools import partial
from tkinter import ttk

import numpy as np
from matplotlib.backend_bases import MouseEvent, key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

from core_adversarial_dbm.compute.dbm_manager import DBMManager
from core_adversarial_dbm.compute.neighbors import Neighbors

from ..main import DataHolder
from .painters import (confidence_painter, dbm_painter, grad_map_painter,
                       neighbors_painter, train_set_painter, wormhole_painter)


class DBMPlot(tk.Frame):
    def __init__(
        self,
        master,
        dbm_manager: DBMManager,
        data: DataHolder,
        neighbors: Neighbors,
        *args,
        **kwargs,
    ):
        super().__init__(master, *args, **kwargs)
        self.dbm_manager = dbm_manager
        self.data = data
        self.neighbors_db = neighbors

        self.fig = Figure(figsize=(8, 8), dpi=100, frameon=False, tight_layout=True)
        self.ax = self.fig.add_subplot(frameon=False)
        self.ax.set_autoscale_on(False)
        self.ax.set_ylim(0.0 - 0.05, 1.0 + 0.05)
        self.ax.set_xlim(0.0 - 0.05, 1.0 + 0.05)
        self.dist_map = None

        self.options_frame = tk.Frame(self.master)
        self.options_frame.grid(column=1, row=0, sticky=("N", "E", "W"))

        self.tooltip_frame = tk.Frame(self.master)
        self.tooltip_frame.grid(column=1, row=1, sticky=tk.NSEW)

        self.tooltip_label = ttk.Label(self.tooltip_frame, text="")
        self.tooltip_label.grid(column=0, row=0, sticky=tk.NSEW)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        # Disable text that shows value under cursor because in some
        # cases it makes the window become absurdly large and then
        # collapse again.
        self.toolbar.set_message = lambda *args, **kwargs: None
        self.toolbar.update()

        self.dbm_painter = dbm_painter.DBMPainter(
            self.ax, self.options_frame, self.dbm_manager
        )
        self.dbm_painter.attach_for_redraw(self)
        self.dbm_painter.grid(column=0, row=0, sticky=tk.NSEW, padx=5, pady=5)
        self.dbm_painter.draw()

        self.train_set_painter = train_set_painter.TrainSetPainter(
            self.ax, self.options_frame, self.dbm_manager, self.data
        )
        self.train_set_painter.attach_for_redraw(self)
        self.train_set_painter.grid(column=0, row=1, sticky=tk.NSEW, padx=5, pady=5)

        self.wormhole_painter = wormhole_painter.WormholePainter(
            self.ax, self.options_frame, self.dbm_manager
        )
        self.wormhole_painter.attach_for_redraw(self)
        self.wormhole_painter.grid(column=0, row=2, sticky=tk.NSEW, padx=5, pady=5)

        self.neighbors_painter = neighbors_painter.NeighborsPainter(
            self.ax, self.options_frame, self.neighbors_db
        )
        self.neighbors_painter.attach_for_redraw(self)
        self.neighbors_painter.grid(column=0, row=3, sticky=tk.NSEW, padx=5, pady=5)

        self.confidence_painter = confidence_painter.ConfidencePainter(
            self.ax,
            self.options_frame,
            self.dbm_manager.grid,
            self.dbm_manager.inverter,
            self.dbm_manager.classifier,
        )
        self.confidence_painter.attach_for_redraw(self)
        self.confidence_painter.grid(column=0, row=4, sticky=tk.NSEW, padx=5, pady=5)

        self.grad_map_painter = grad_map_painter.GradMapPainter(
            self.ax, self.options_frame, self.dbm_manager
        )
        self.grad_map_painter.attach_for_redraw(self)
        self.grad_map_painter.grid(column=0, row=5, sticky=tk.NSEW, padx=5, pady=5)

        self.canvas.mpl_connect("button_press_event", self.invert_on_click)
        self.canvas.mpl_connect("motion_notify_event", self.invert_if_drag)
        self.canvas.mpl_connect(
            "motion_notify_event", self.update_tooltip_with_distance
        )
        self.canvas.mpl_connect("button_release_event", self.stop_inverting)

        self.canvas.get_tk_widget().grid(column=0, row=0, sticky="WNES")
        self.toolbar.grid(column=0, row=1, sticky=tk.EW)

        self.options_frame.grid_columnconfigure(0, weight=1)
        self.options_frame.grid_rowconfigure(0, weight=1)
        self.options_frame.grid_rowconfigure(1, weight=1)
        self.options_frame.grid_rowconfigure(2, weight=1)
        self.options_frame.grid_rowconfigure(3, weight=1)
        self.options_frame.grid_rowconfigure(4, weight=1)
        self.options_frame.grid_rowconfigure(5, weight=1)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=5)
        self.grid_rowconfigure(1, weight=1, minsize=30)

        self._invert_on = False
        self._scheduled_tooltip_update = None

    def redraw(self, *args):
        self.canvas.draw()

    def update_tooltip_with_distance(self, event: MouseEvent):
        # We need to figure out row and col from xdata and ydata
        if (
            not self.dbm_painter.options.encode_distance
            or event.xdata is None
            or event.ydata is None
        ):
            return
        if self.dbm_manager.get_distance_map() is None:
            if self._scheduled_tooltip_update is not None:
                # If we're moving the mouse and data is not available, we'll
                # schedule so many after() tasks that it might be a problem.
                # This prevents it.
                self.after_cancel(self._scheduled_tooltip_update)
            self._scheduled_tooltip_update = self.after(
                200, partial(self.update_tooltip_with_distance, event)
            )
        width = height = self.dbm_manager.dbm_resolution
        left, right, bottom, top = (0.0, 1.0, 0.0, 1.0)
        if not (left <= event.xdata <= right and bottom <= event.ydata <= top):
            return

        px_width, px_height = (right - left) / width, (top - bottom) / width
        x_cell = int(np.floor(event.xdata / px_width))
        y_cell = int(np.floor(event.ydata / px_height))
        # Row and Col are in internal coordinates. The grid starts from (x=0, y=0) and goes
        # row-wise to (x=1, y=1).
        dist = self.dbm_manager.distance_to_adv_at(row=y_cell, col=x_cell)
        self.tooltip_label["text"] = f"Distance to closest Adv. = {dist:.4f}"

    def display_inverted_img(self, x, y):
        if x is None or y is None:
            return
        inverted = self.dbm_manager.invert_point(x, y)
        self.master.inverted_vis.vis.set_data(inverted)
        self.master.inverted_vis.canvas.draw()
        # self.master.inverted_vis.update()

    def invert_if_drag(self, event):
        if not self._invert_on:
            return
        self.display_inverted_img(event.xdata, event.ydata)

    def invert_on_click(self, event: MouseEvent):
        self._invert_on = True
        self.display_inverted_img(event.xdata, event.ydata)

    def stop_inverting(self, *args):
        self._invert_on = False

    def destroy(self) -> None:
        self.dbm_manager.destroy()
        return super().destroy()


def main():
    root = tk.Tk()
    root.wm_title("Embedding in Tk")

    content = tk.Frame(root)
    content.grid(column=0, row=0, sticky="NESW")
    plot = DBMPlot(content, np.random.randint(10, size=(28, 28)))

    plot.grid(column=0, row=0, sticky="NSEW", padx=5, pady=5)

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    content.grid_columnconfigure(0, weight=4)
    content.grid_columnconfigure(1, weight=1, minsize=10)
    content.grid_rowconfigure(0, weight=1)

    tk.mainloop()


if __name__ == "__main__":
    main()
