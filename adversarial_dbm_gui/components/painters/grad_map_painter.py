import tkinter as tk
from dataclasses import dataclass
from functools import cache, partial
from tkinter import ttk
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.func import grad, jacfwd, jacrev, vmap

from ...compute.dbm_manager import DBMManager
from . import painter

GradModeType = Literal[
    "d(class)/d(inverter.grid)",
    "d(class.inverter)/d(grid)",
    "d(inverter)/d(grid)",
    "d(H(class))/d(inverter.grid)",
    "d(H(class.inverter)/d(grid))",
    "d(||inverter||)/d(grid)",
]


@dataclass
class Options:
    enabled: bool = False
    alpha: float = 1.0
    grad_mode: GradModeType = "d(class)/d(inverter.grid)"
    z_order: int = 0


class GradMapPainter(painter.Painter):
    # TODO maybe we don't need the DBMManager itself?
    def __init__(self, ax: plt.Axes, master: tk.Frame, dbm_manager: DBMManager) -> None:
        super().__init__(ax, master)

        self.dbm_manager = dbm_manager
        self.options = Options()

        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)

        self.enabled = tk.BooleanVar(self.frame, value=self.options.enabled)
        self.enabled_btn = ttk.Checkbutton(
            self.frame,
            text="Gradient map",
            variable=self.enabled,
            onvalue=True,
            command=self.set_enabled,
        )

        self.grad_mode = tk.StringVar(value=self.options.grad_mode)
        self.mode_listbox = ttk.Combobox(self.frame, textvariable=self.grad_mode)
        self.mode_listbox["values"] = GradModeType.__args__
        self.mode_listbox.state(["readonly"])
        self.mode_listbox.bind("<<ComboboxSelected>>", self.set_grad_mode)

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

    def set_grad_mode(self, *args):
        self.mode_listbox.selection_clear()
        self.options.grad_mode = self.grad_mode.get()
        self.update_params()

    def set_z_order(self, *args):
        self.z_order_spinbox.selection_clear()
        self.options.z_order = self.z_order_val.get()
        self.update_params()

    def draw(self):
        with T.no_grad():
            if self.options.grad_mode == "d(class)/d(inverter.grid)":
                mat_norms = self._norm_jac_classif_wrt_inverted_grid()
            elif self.options.grad_mode == "d(class.inverter)/d(grid)":
                mat_norms = self._norm_jac_classif_and_inversion_wrt_grid()
            elif self.options.grad_mode == "d(inverter)/d(grid)":
                mat_norms = self._norm_jac_inversion_wrt_grid()
            elif self.options.grad_mode == "d(H(class))/d(inverter.grid)":
                mat_norms = self._norm_jac_entropy_of_classif_wrt_inverted_grid()
            elif self.options.grad_mode == "d(H(class.inverter)/d(grid))":
                mat_norms = self._norm_jac_entropy_of_classif_and_inversion_wrt_grid()
            elif self.options.grad_mode == "d(||inverter||)/d(grid)":
                mat_norms = self._norm_jac_norm_of_inversion_wrt_grid()
        mat_norms = mat_norms.cpu().reshape(
            (self.dbm_manager.dbm_resolution, self.dbm_manager.dbm_resolution)
        )
        mat_norms = (mat_norms - mat_norms.min()) / (mat_norms.max() - mat_norms.min())
        if self.drawing is None:
            self.drawing = self.ax.imshow(
                mat_norms,
                extent=(0.0, 1.0, 0.0, 1.0),
                interpolation="none",
                cmap="viridis",
                origin="lower",
            )
            self.options.z_order = self.drawing.get_zorder()
            self.z_order_val.set(self.drawing.get_zorder())
        self.drawing.set_visible(self.options.enabled)
        self.drawing.set_data(mat_norms)
        self.drawing.set_zorder(self.options.z_order)
        self.drawing.set_alpha(self.options.alpha)

        return super().draw()

    @cache
    def _norm_jac_classif_wrt_inverted_grid(self):
        return vmap(
            lambda *a, **kw: T.linalg.matrix_norm(
                jacrev(self.dbm_manager.classifier.activations)(*a, **kw)
            ),
            chunk_size=10000,
        )(self.dbm_manager.inverter(self.dbm_manager.grid))

    @cache
    def _norm_jac_classif_and_inversion_wrt_grid(self):
        return vmap(
            lambda *a, **kw: T.linalg.matrix_norm(
                jacrev(
                    partial(
                        _invert_and_activate,
                        self.dbm_manager.inverter,
                        self.dbm_manager.classifier,
                    )
                )(*a, **kw)
            ),
            chunk_size=5000,
        )(self.dbm_manager.grid)

    @cache
    def _norm_jac_inversion_wrt_grid(self):
        return vmap(
            lambda *a, **kw: T.linalg.matrix_norm(
                jacfwd(self.dbm_manager.inverter)(*a, **kw)
            ),
            chunk_size=10000,
        )(self.dbm_manager.grid)

    @cache
    def _norm_jac_norm_of_inversion_wrt_grid(self):
        def inversion_norm(inputs):
            return T.linalg.vector_norm(self.dbm_manager.inverter(inputs), dim=-1) ** 2

        out = vmap(grad(inversion_norm), chunk_size=10000)(self.dbm_manager.grid)
        return T.linalg.vector_norm(out, dim=1)

    @cache
    def _norm_jac_entropy_of_classif_wrt_inverted_grid(self):
        def entropy_classif(inputs: T.Tensor):
            logits = self.dbm_manager.classifier.activations(inputs)
            min_real = T.finfo(logits.dtype).min
            logits = T.clamp(logits, min=min_real)
            probs = F.softmax(logits, dim=-1)
            logits = logits - T.logsumexp(logits, -1)
            p_log_p = logits * probs
            entropy = -p_log_p.sum(-1) / np.log(self.dbm_manager.n_classes)
            return entropy

        out = T.linalg.vector_norm(
            vmap(grad(entropy_classif), chunk_size=10000)(
                self.dbm_manager.inverter(self.dbm_manager.grid)
            ),
            dim=1,
        )
        return out

    @cache
    def _norm_jac_entropy_of_classif_and_inversion_wrt_grid(self):
        def entropy_classif_inversion(inputs: T.Tensor) -> T.Tensor:
            inverted = self.dbm_manager.inverter(inputs)
            logits = self.dbm_manager.classifier.activations(inverted)
            min_real = T.finfo(logits.dtype).min
            logits = T.clamp(logits, min=min_real)
            probs = F.softmax(logits, dim=-1)
            logits = logits - T.logsumexp(logits, -1)
            p_log_p = logits * probs
            entropy = -p_log_p.sum(-1) / np.log(self.dbm_manager.n_classes)
            return entropy

        out = T.linalg.vector_norm(
            vmap(grad(entropy_classif_inversion), chunk_size=10000)(
                self.dbm_manager.grid
            ),
            dim=1,
        )
        return out


def _invert_and_activate(inverter, classifier, points):
    return classifier.activations(inverter(points))
