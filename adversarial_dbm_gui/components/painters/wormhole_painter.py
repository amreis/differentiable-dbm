import tkinter as tk
from dataclasses import dataclass
from functools import partial
from tkinter import ttk

import matplotlib.pyplot as plt

from core_adversarial_dbm.compute.dbm_manager import DBMManager

from . import painter


@dataclass
class Options:
    enabled: bool = False
    use_naive_wormholes: bool = False
    alpha: float = 1.0
    z_order: int = 0


class WormholePainter(painter.Painter):
    def __init__(self, ax: plt.Axes, master: tk.Frame, dbm_manager: DBMManager) -> None:
        super().__init__(ax, master)

        self.dbm_manager = dbm_manager
        self.options = Options()

        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=2)
        self.frame.grid_columnconfigure(1, weight=2)
        self.frame.grid_columnconfigure(2, weight=1)

        self.enabled = tk.BooleanVar(self.frame, value=self.options.enabled)
        self.enabled_btn = ttk.Checkbutton(
            self.frame,
            text="Wormholes",
            variable=self.enabled,
            onvalue=True,
            command=self.set_enabled,
        )

        self.options_btn = ttk.Button(
            self.frame, command=self.spawn_options, text="Options"
        )

        self.z_order_val = tk.IntVar(value=self.options.z_order)
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
        self.options_btn.grid(column=1, row=0, sticky=tk.E)
        self.z_order_spinbox.grid(column=2, row=0, sticky=tk.E, padx=5)

    def update_params(self, *args):
        self.draw()

    def enable_options(self, window):
        self.options_btn["state"] = tk.NORMAL
        window.destroy()

    def spawn_options(self):
        options_window = tk.Toplevel(self.master)
        self.options_btn["state"] = tk.DISABLED
        options_window.protocol(
            "WM_DELETE_WINDOW", partial(self.enable_options, options_window)
        )
        options_window.attributes("-topmost", 1)

        self.use_naive_wormholes = tk.BooleanVar(
            options_window, value=self.options.use_naive_wormholes
        )
        use_naive_wormholes_btn = ttk.Checkbutton(
            options_window,
            text="Use naive wormholes",
            variable=self.use_naive_wormholes,
            onvalue=True,
            command=self.set_use_naive_wormholes,
        )

        self.alpha_val = tk.DoubleVar(options_window, value=self.options.alpha)
        alpha_slider = ttk.Scale(
            options_window,
            command=self.set_alpha,
            variable=self.alpha_val,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
        )
        self.alpha_slider_label = ttk.Label(
            options_window, text=f"Alpha: {self.alpha_val.get():.4f}"
        )

        use_naive_wormholes_btn.grid(column=0, row=0, columnspan=2, sticky=tk.NW)
        self.alpha_slider_label.grid(column=0, row=1, sticky=tk.NW)
        alpha_slider.grid(column=1, row=1, sticky=tk.NW)

        options_window.grid_rowconfigure(0, weight=1)
        options_window.grid_rowconfigure(1, weight=1)

        options_window.grid_columnconfigure(0, weight=1)
        options_window.grid_columnconfigure(1, weight=2)

    def set_enabled(self, *args):
        self.options.enabled = self.enabled.get()
        self.update_params()

    def set_use_naive_wormholes(self, *args):
        self.options.use_naive_wormholes = self.use_naive_wormholes.get()
        self.update_params()

    def set_alpha(self, *args):
        self.options.alpha = self.alpha_val.get()
        self.alpha_slider_label["text"] = f"Alpha: {self.alpha_val.get():.4f}"
        self.update_params()

    def set_z_order(self, *args):
        self.z_order_spinbox.selection_clear()
        self.options.z_order = self.z_order_val.get()
        self.update_params()

    def draw(self):
        if self.drawing is None:
            self.drawing = self.ax.imshow(
                self.dbm_manager.get_wormhole_data(),
                extent=(0.0, 1.0, 0.0, 1.0),
                interpolation="none",
                origin="lower",
                cmap="tab10",
                vmax=self.dbm_manager.n_classes - 1,
                vmin=0,
            )
            self.options.z_order = self.drawing.get_zorder()
            self.z_order_val.set(self.drawing.get_zorder())
        if self.options.use_naive_wormholes:
            self.drawing.set_data(self.dbm_manager.get_naive_wormhole_data())
        else:
            self.drawing.set_data(self.dbm_manager.get_wormhole_data())
        self.drawing.set_visible(self.options.enabled)
        self.drawing.set_alpha(self.options.alpha)
        self.drawing.set_zorder(self.options.z_order)

        return super().draw()
