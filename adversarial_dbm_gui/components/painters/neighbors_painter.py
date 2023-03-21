import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

from matplotlib import pyplot as plt

from ...compute.neighbors import Neighbors
from .painter import Painter


@dataclass
class Options:
    enabled: bool = False
    alpha: float = 1.0


class NeighborsPainter(Painter):
    def __init__(self, ax: plt.Axes, master: tk.Frame, neighbors: Neighbors) -> None:
        super().__init__(ax, master)

        self.neighbors_db = neighbors
        self.options = Options()

        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        self.enabled = tk.BooleanVar(self.frame, value=self.options.enabled)
        self.enabled_btn = ttk.Checkbutton(
            self.frame,
            text="Distance to Closest Neighbor",
            variable=self.enabled,
            onvalue=True,
            command=self.set_enabled,
        )
        self.alpha_val = tk.DoubleVar(self.frame, value=self.options.alpha)
        self.alpha_slider = ttk.Scale(
            self.frame,
            variable=self.alpha_val,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            command=self.set_alpha,
        )

        self.enabled_btn.grid(column=0, row=0, sticky=tk.EW)
        self.alpha_slider.grid(column=0, row=1, sticky=tk.EW)

        self._redraw_observers = []

    def attach_for_redraw(self, observer):
        self._redraw_observers.append(observer)

    def detach_for_redraw(self, observer):
        self._redraw_observers.remove(observer)

    def update_params(self, *args):
        self.draw()

    def set_enabled(self, *args):
        self.options.enabled = self.enabled.get()
        self.update_params()

    def set_alpha(self, *args):
        self.options.alpha = self.alpha_val.get()
        self.update_params()

    def draw(self):
        if self.drawing is None:
            self.drawing = self.ax.imshow(
                self.neighbors_db.get_distance_to_nearest_same_class_neighbor(),
                extent=(0.0, 1.0, 0.0, 1.0),
                interpolation="none",
                origin="lower",
                cmap="viridis",
            )
        # TODO Update to switch between any-class and same-class.
        # TODO add option for closest different-class neighbor?
        self.drawing.set_data(self.neighbors_db.get_distance_to_nearest_same_class_neighbor())
        self.drawing.set_visible(self.options.enabled)
        self.drawing.set_alpha(self.options.alpha)

        return super().draw()
