import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

import matplotlib.pyplot as plt

from ...compute.dbm_manager import DBMManager
from ...main import DataHolder
from . import painter


@dataclass
class Options:
    enabled: bool = False
    show_misclassifications: bool = False
    alpha: float = 1.0


class TrainSetPainter(painter.Painter):
    def __init__(
        self, ax: plt.Axes, master: tk.Frame, dbm_manager: DBMManager, data: DataHolder
    ) -> None:
        super().__init__(ax, master)

        self.dbm_manager = dbm_manager
        self.data = data
        self.options = Options()

        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)

        self.enabled = tk.BooleanVar(self.frame, value=self.options.enabled)
        self.enabled_btn = ttk.Checkbutton(
            self.frame,
            text="Training Set",
            variable=self.enabled,
            onvalue=True,
            command=self.set_enabled,
        )

        self.show_misclassifications = tk.BooleanVar(
            self.frame, value=self.options.show_misclassifications
        )
        self.show_misclassifications_btn = ttk.Checkbutton(
            self.frame,
            text="Show misclassified",
            variable=self.show_misclassifications,
            onvalue=True,
            command=self.set_show_misclassifications,
        )

        self.alpha_val = tk.DoubleVar(value=self.options.alpha)
        self.alpha_slider = ttk.Scale(
            self.frame,
            command=self.set_alpha,
            variable=self.alpha_val,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
        )

        self.enabled_btn.grid(column=0, row=0, sticky=tk.EW)
        self.show_misclassifications_btn.grid(column=1, row=0, sticky=tk.EW)
        self.alpha_slider.grid(column=0, row=1, columnspan=2, sticky=tk.NSEW)

    def update_params(self, *args):
        self.draw()

    def set_enabled(self, *args):
        self.options.enabled = self.enabled.get()
        self.update_params()

    def set_show_misclassifications(self, *args):
        self.options.show_misclassifications = self.show_misclassifications.get()
        self.update_params()

    def set_alpha(self, *args):
        self.options.alpha = self.alpha_val.get()
        self.update_params()

    def draw(self):
        if self.drawing is None:
            self.drawing = self.ax.scatter(
                *self.data.X_tsne.T,
                c=self.data.y_classif_train,
                cmap="tab10",
                edgecolors="#FFFFFF",
                linewidths=0.3,
            )
        self.drawing.set_visible(self.enabled.get())
        self.drawing.set_alpha(self.options.alpha)

        return super().draw()