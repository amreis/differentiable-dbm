import tkinter as tk
from dataclasses import dataclass
from functools import cache
from tkinter import ttk

import numpy as np
import torch as T
import torch.nn as nn
from matplotlib import pyplot as plt

from . import painter


@dataclass
class Options:
    enabled: bool = False
    conf_mode: str = "prob_best_class"
    z_order: int = 0


class ConfidencePainter(painter.Painter):
    def __init__(
        self,
        ax: plt.Axes,
        master: tk.Frame,
        grid: T.Tensor,
        inverter: nn.Module,
        classifier: nn.Module,
    ) -> None:
        super().__init__(ax, master)

        self._grid = grid.clone()
        self._grid_width = int(np.sqrt(self._grid.shape[0]))
        self._inverter = inverter
        self._classifier = classifier

        self.options = Options()

        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=4)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, weight=1)

        self.enabled = tk.BooleanVar(self.frame, value=self.options.enabled)
        self.enabled_btn = ttk.Checkbutton(
            self.frame,
            text="Classifier Confidence",
            variable=self.enabled,
            onvalue=True,
            command=self.set_enabled,
        )

        self.conf_mode = tk.StringVar(value=self.options.conf_mode)
        self.mode_listbox = ttk.Combobox(self.frame, textvariable=self.conf_mode)
        self.mode_listbox["values"] = ["prob_best_class", "entropy"]
        self.mode_listbox.state(["readonly"])
        self.mode_listbox.bind("<<ComboboxSelected>>", self.set_conf_mode)

        self.z_order_val = tk.IntVar(self.frame, value=self.options.z_order)
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
        self.z_order_spinbox.grid(column=2, row=0, sticky=tk.E, padx=5)

    def update_params(self, *args):
        self.draw()

    def set_enabled(self, *args):
        self.options.enabled = self.enabled.get()
        self.update_params()

    def set_conf_mode(self, *args):
        self.mode_listbox.selection_clear()
        self.options.conf_mode = self.conf_mode.get()
        self.update_params()

    def set_z_order(self, *args):
        self.z_order_spinbox.selection_clear()
        self.options.z_order = self.z_order_val.get()
        self.update_params()

    def draw(self):
        if self.drawing is None:
            self.drawing = self.ax.imshow(
                self._confidence_over_grid(self.options.conf_mode),
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                interpolation="none",
                cmap="viridis",
                # vmin=0.0,
                # vmax=1.0,
            )
            self.options.z_order = self.drawing.get_zorder()
            self.z_order_val.set(self.drawing.get_zorder())

        self.drawing.set_data(self._confidence_over_grid(self.options.conf_mode))
        self.drawing.set_visible(self.options.enabled)
        self.drawing.set_zorder(self.options.z_order)

        return super().draw()

    @cache
    def _confidence_over_grid(self, mode: str):
        with T.no_grad():
            if mode == "prob_best_class":
                confidence = self._classifier.prob_best_class(
                    self._inverter(self._grid)
                ) # TODO apply power transform here too and see if comparable to "entropy" vis.
            else:  # if mode == "entropy"
                confidence = self._classifier.classification_entropy(
                    self._inverter(self._grid)
                ) ** 0.1

        return confidence.cpu().numpy().reshape((self._grid_width, self._grid_width))
