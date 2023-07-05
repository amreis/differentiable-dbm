import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

import matplotlib.pyplot as plt

from core_adversarial_dbm.compute.neighbors import Neighbors

from . import painter


@dataclass
class Options:
    enabled: bool = False
    alpha: float = 1.0
    dist_mode: str = "any_class"
    z_order: int = 0


class NeighborsPainter(painter.Painter):
    def __init__(self, ax: plt.Axes, master: tk.Frame, neighbors: Neighbors) -> None:
        super().__init__(ax, master)

        self.neighbors_db = neighbors
        self.options = Options()

        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=2)
        self.frame.grid_columnconfigure(1, weight=2)
        self.frame.grid_columnconfigure(2, weight=1)

        self.enabled = tk.BooleanVar(self.frame, value=self.options.enabled)
        self.enabled_btn = ttk.Checkbutton(
            self.frame,
            text="Distance to Closest Neighbor",
            variable=self.enabled,
            onvalue=True,
            command=self.set_enabled,
        )
        self.dist_mode = tk.StringVar(value=self.options.dist_mode)
        self.mode_listbox = ttk.Combobox(self.frame, textvariable=self.dist_mode)
        self.mode_listbox["values"] = ["any_class", "same_class", "diff_class"]
        self.mode_listbox.state(["readonly"])
        self.mode_listbox.bind("<<ComboboxSelected>>", self.set_dist_mode)

        self.alpha_val = tk.DoubleVar(self.frame, value=self.options.alpha)
        self.alpha_slider = ttk.Scale(
            self.frame,
            variable=self.alpha_val,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            command=self.set_alpha,
        )

        self.z_order_val = tk.IntVar()
        self.z_order_spinbox = ttk.Spinbox(
            self.frame,
            command=self.set_z_order,
            from_=0,
            to=10,
            textvariable=self.z_order_val,
            width=3,
        )
        self.z_order_spinbox.state(["readonly"])

        self.enabled_btn.grid(column=0, row=0, sticky=tk.EW)
        self.mode_listbox.grid(column=1, row=0, sticky=tk.EW)
        self.alpha_slider.grid(column=0, row=1, columnspan=2, sticky=tk.EW)
        self.z_order_spinbox.grid(column=2, row=0, rowspan=2, sticky=tk.E, padx=5)

    def update_params(self, *args):
        self.draw()

    def set_enabled(self, *args):
        self.options.enabled = self.enabled.get()
        self.update_params()

    def set_alpha(self, *args):
        self.options.alpha = self.alpha_val.get()
        self.update_params()

    def set_dist_mode(self, *args):
        self.mode_listbox.selection_clear()
        self.options.dist_mode = self.dist_mode.get()
        self.update_params()

    def set_z_order(self, *args):
        self.z_order_spinbox.selection_clear()
        self.options.z_order = self.z_order_val.get()
        self.update_params()

    def get_distance_fn(self):
        MODE_TO_FN = {
            "any_class": self.neighbors_db.get_distance_to_nearest_neighbor,
            "same_class": self.neighbors_db.get_distance_to_nearest_same_class_neighbor,
            "diff_class": self.neighbors_db.get_distance_to_nearest_diff_class_neighbor,
        }
        return MODE_TO_FN[self.options.dist_mode]

    def draw(self):
        if self.drawing is None:
            self.drawing = self.ax.imshow(
                self.get_distance_fn()(),
                extent=(0.0, 1.0, 0.0, 1.0),
                interpolation="none",
                origin="lower",
                cmap="viridis",
            )
            self.options.z_order = self.drawing.get_zorder()
            self.z_order_val.set(self.drawing.get_zorder())
        self.drawing.set_data(self.get_distance_fn()())
        self.drawing.set_visible(self.options.enabled)
        self.drawing.set_alpha(self.options.alpha)
        self.drawing.set_zorder(self.options.z_order)

        return super().draw()

    def cache_clear(self):
        self.neighbors_db.cache_clear()
