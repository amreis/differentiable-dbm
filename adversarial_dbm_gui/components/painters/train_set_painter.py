import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

import matplotlib.pyplot as plt
import torch as T

from core_adversarial_dbm.compute.dbm_manager import DBMManager

from ...main import DataHolder
from . import painter


@dataclass
class Options:
    enabled: bool = False
    show_misclassifications: bool = False
    alpha: float = 1.0
    z_order: int = 0


class TrainSetPainter(painter.Painter):
    def __init__(
        self, ax: plt.Axes, master: tk.Frame, dbm_manager: DBMManager, data: DataHolder
    ) -> None:
        super().__init__(ax, master)

        self.drawing_incorrect = None

        self.dbm_manager = dbm_manager
        self.data = data
        self.options = Options()

        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=2)
        self.frame.grid_columnconfigure(1, weight=2)
        self.frame.grid_columnconfigure(2, weight=1)

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
        self.show_misclassifications_btn.grid(column=1, row=0, sticky=tk.EW)
        self.alpha_slider.grid(column=0, row=1, columnspan=2, sticky=tk.EW)
        self.z_order_spinbox.grid(column=2, row=0, rowspan=2, sticky=tk.E, padx=5)

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

    def set_z_order(self, *args):
        self.z_order_spinbox.selection_clear()
        self.options.z_order = self.z_order_val.get()
        self.update_params()

    @T.no_grad()
    def draw(self):
        if self.drawing is not None:
            self.drawing.remove()
            self.drawing = None
        if self.drawing_incorrect is not None:
            self.drawing_incorrect.remove()
            self.drawing_incorrect = None

        if self.options.show_misclassifications:
            correct = (
                self.data.y_classif_train
                == self.dbm_manager.classifier.classify(
                    self.dbm_manager.inverter(
                        T.tensor(self.data.X_proj, device=self.dbm_manager.grid.device)
                    )
                )
                .cpu()
                .numpy()
            )
            incorrect = ~correct
            self.drawing = self.ax.scatter(
                *self.data.X_proj[correct].T,
                c=self.data.y_classif_train[correct],
                cmap="tab10",
                edgecolors="#FFFFFF",
                linewidths=0.3,
                vmax=self.dbm_manager.n_classes - 1,
                vmin=0,
            )
            self.drawing_incorrect = self.ax.scatter(
                *self.data.X_proj[incorrect].T,
                c=self.data.y_classif_train[incorrect],
                cmap="tab10",
                marker="x",
            )
        else:
            self.drawing = self.ax.scatter(
                *self.data.X_proj.T,
                c=self.data.y_classif_train,
                cmap="tab10",
                edgecolors="#FFFFFF",
                linewidths=0.3,
            )
            self.options.z_order = self.drawing.get_zorder()
            self.z_order_val.set(self.drawing.get_zorder())

        self.drawing.set_visible(self.enabled.get())
        self.drawing.set_alpha(self.options.alpha)
        self.drawing.set_zorder(self.options.z_order)

        if self.drawing_incorrect is not None:
            self.drawing_incorrect.set_visible(self.options.enabled)
            self.drawing.set_alpha(self.options.alpha)
            self.drawing.set_zorder(self.options.z_order)

        return super().draw()
