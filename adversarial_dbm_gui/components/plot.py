import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler, MouseEvent, MouseButton
from matplotlib.figure import Figure

import numpy as np

from .plotcontrols import Controls
from ..compute.dbm_manager import DBMManager


class DBMPlot(tk.Frame):
    def __init__(self, master, dbm_manager: DBMManager, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.dbm_manager = dbm_manager

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot()
        self.base_dbm = self.ax.imshow(
            self.dbm_manager.get_dbm_data(),
            extent=(0.0, 1.0, 0.0, 1.0),
            origin="lower",
            interpolation="none",
            cmap="tab10",
        )
        self.dist_map = None

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()

        self.toolbar: tk.Frame = NavigationToolbar2Tk(
            self.canvas, self, pack_toolbar=False
        )
        self.toolbar.update()

        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.canvas.mpl_connect("button_press_event", self.invert_on_click)
        self.canvas.mpl_connect("motion_notify_event", self.invert_if_drag)
        self.canvas.mpl_connect("button_release_event", self.stop_inverting)

        self.canvas.get_tk_widget().grid(column=0, row=0, sticky="WNES")
        self.toolbar.grid(column=0, row=1, sticky="N")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1, minsize=30)

        self._invert_on = False

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

    def on_key_press(self, event):
        print(f"you pressed {event.key}")
        key_press_handler(event, self.canvas, self.toolbar)

    def update_params(self, event):
        if event["control_id"] == "dbm":
            self.update_dbm(event)
        elif event["control_id"] == "dist_map":
            self.update_dist_map(event)
        self.canvas.draw()

    def update_dbm(self, event: dict):
        enabled = event["enabled"]
        self.base_dbm.set_visible(enabled)
        self.base_dbm.set_alpha(event["alpha"])


    def update_dist_map(self, event):
        if self.dist_map is None:
            self.dist_map = self.ax.imshow(
                self.dbm_manager.get_distance_map(),
                extent=(0.0, 1.0, 0.0, 1.0),
                interpolation="none",
                origin="lower",
                cmap="viridis",
            )
        enabled = event["enabled"]
        self.dist_map.set_visible(enabled)
        self.dist_map.set_alpha(event["alpha"])


def main():
    root = tk.Tk()
    root.wm_title("Embedding in Tk")

    content = tk.Frame(root)
    content.grid(column=0, row=0, sticky="NESW")
    plot = DBMPlot(content, np.random.randint(10, size=(28, 28)))

    plot.grid(column=0, row=0, sticky="NSEW", padx=5, pady=5)
    controls = Controls(content)
    controls.grid(column=1, row=0, sticky="NW", padx=5, pady=5)

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    content.grid_columnconfigure(0, weight=4)
    content.grid_columnconfigure(1, weight=1, minsize=10)
    content.grid_rowconfigure(0, weight=1)

    tk.mainloop()


if __name__ == "__main__":
    main()
