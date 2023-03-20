import tkinter as tk

import numpy as np
from matplotlib.backend_bases import MouseEvent, key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..compute.dbm_manager import DBMManager
from ..main import DataHolder
from .painters import dbm_painter, train_set_painter, wormhole_painter


class DBMPlot(tk.Frame):
    def __init__(
        self, master, dbm_manager: DBMManager, data: DataHolder, *args, **kwargs
    ):
        super().__init__(master, *args, **kwargs)
        self.dbm_manager = dbm_manager
        self.data = data

        self.fig = Figure(figsize=(8, 8), dpi=100, frameon=False)
        self.ax = self.fig.add_subplot(frameon=False)
        self.ax.set_autoscale_on(False)
        self.ax.set_ylim(0.0-0.05, 1.0+0.05)
        self.ax.set_xlim(0.0-0.05, 1.0+0.05)
        self.dist_map = None

        self.options_frame = tk.Frame(self.master)
        self.options_frame.grid(column=1, row=0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()

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

        self.wormhole_painter = wormhole_painter.WormholePainter(self.ax, self.options_frame, self.dbm_manager)
        self.wormhole_painter.attach_for_redraw(self)
        self.wormhole_painter.grid(column=0, row=2, sticky=tk.NSEW, padx=5, pady=5)

        # TODO: Add WormholePainter

        self.canvas.mpl_connect("button_press_event", self.invert_on_click)
        self.canvas.mpl_connect("motion_notify_event", self.invert_if_drag)
        self.canvas.mpl_connect("button_release_event", self.stop_inverting)

        self.canvas.get_tk_widget().grid(column=0, row=0, sticky="WNES")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1, minsize=30)

        self._invert_on = False

    def redraw(self, *args):
        self.canvas.draw()

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
