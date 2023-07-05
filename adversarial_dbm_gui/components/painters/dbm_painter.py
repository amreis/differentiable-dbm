import tkinter as tk
from dataclasses import dataclass
from functools import partial
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np

from core_adversarial_dbm.compute.dbm_manager import DBMManager

from . import painter


@dataclass
class Options:
    enabled: bool = True
    encode_distance: bool = False
    invert_distance: bool = False
    show_borders: bool = False
    alpha: float = 1.0
    power: float = 1.0
    blend_mode: str = "multiply"
    z_order: int = 0


class DBMPainter(painter.Painter):
    def __init__(self, ax: plt.Axes, master: tk.Frame, dbm_manager: DBMManager) -> None:
        super().__init__(ax, master)

        self._borders_drawing = None

        self.dbm_manager = dbm_manager
        self.options = Options()

        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=2)
        self.frame.grid_columnconfigure(1, weight=2)
        self.frame.grid_columnconfigure(2, weight=1)

        self.enabled = tk.BooleanVar(self.frame, value=self.options.enabled)
        self.enabled_btn = ttk.Checkbutton(
            self.frame,
            text="DBM",
            variable=self.enabled,
            onvalue=True,
            command=self.set_enabled,
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

        self.options_btn = ttk.Button(
            self.frame, command=self.spawn_options, text="Options"
        )

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
        self.distance_enabled = tk.BooleanVar(value=self.options.encode_distance)
        distance_btn = ttk.Checkbutton(
            options_window,
            text="Encode distance in",
            variable=self.distance_enabled,
            onvalue=True,
            command=self.set_encode_distance,
        )
        self.invert_distance = tk.BooleanVar(value=self.options.invert_distance)
        invert_distance_btn = ttk.Checkbutton(
            options_window,
            text="Invert distance",
            variable=self.invert_distance,
            onvalue=True,
            command=self.set_invert_distance,
        )

        self.borders_enabled = tk.BooleanVar(value=self.options.show_borders)
        borders_enabled_btn = ttk.Checkbutton(
            options_window,
            text="Show borders",
            onvalue=True,
            variable=self.borders_enabled,
            command=self.set_show_borders,
        )

        self.alpha_val = tk.DoubleVar(value=self.options.alpha)
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

        self.power = tk.DoubleVar(value=np.log(self.options.power))
        power_transf_slider = ttk.Scale(
            options_window,
            command=self.set_power,
            variable=self.power,
            from_=-2.0,
            to=3.0,
            orient=tk.HORIZONTAL,
        )
        self.power_transf_slider_label = ttk.Label(
            options_window, text=f"Power: {np.exp(self.power.get()):.4f}"
        )

        self.blend_mode = tk.StringVar(value=self.options.blend_mode)
        self.blend_mode_listbox = ttk.Combobox(
            options_window, textvariable=self.blend_mode
        )
        self.blend_mode_listbox["values"] = ["multiply", "hsv", "soft_light", "overlay"]
        self.blend_mode_listbox.state(["readonly"])
        self.blend_mode_listbox.bind("<<ComboboxSelected>>", self.set_blend_mode)

        distance_btn.grid(column=0, row=0, sticky=tk.NW)
        self.blend_mode_listbox.grid(column=1, row=0, sticky=tk.EW)
        invert_distance_btn.grid(column=0, row=1, columnspan=2, sticky=tk.NW)
        borders_enabled_btn.grid(column=0, row=2, columnspan=2, sticky=tk.NW)
        self.alpha_slider_label.grid(column=0, row=3, sticky=tk.NW)
        alpha_slider.grid(column=1, row=3, sticky=tk.EW)
        self.power_transf_slider_label.grid(column=0, row=4, sticky=tk.NW)
        power_transf_slider.grid(column=1, row=4, sticky=tk.EW)

        options_window.grid_rowconfigure(0, weight=1)
        options_window.grid_rowconfigure(1, weight=1)
        options_window.grid_rowconfigure(2, weight=1)
        options_window.grid_rowconfigure(3, weight=1)
        options_window.grid_rowconfigure(4, weight=1)
        options_window.grid_rowconfigure(5, weight=1)

        options_window.grid_columnconfigure(0, weight=1)
        options_window.grid_columnconfigure(1, weight=2)

    def set_enabled(self, *args):
        self.options.enabled = self.enabled.get()
        self.update_params()

    def set_encode_distance(self, *args):
        self.options.encode_distance = self.distance_enabled.get()
        self.update_params()

    def set_invert_distance(self, *args):
        self.options.invert_distance = self.invert_distance.get()
        self.update_params()

    def set_show_borders(self, *args):
        self.options.show_borders = self.borders_enabled.get()
        self.update_params()

    def set_alpha(self, *args):
        self.options.alpha = self.alpha_val.get()
        self.alpha_slider_label["text"] = f"Alpha: {self.alpha_val.get():.4f}"
        self.update_params()

    def set_power(self, *args):
        self.options.power = np.exp(self.power.get())
        self.power_transf_slider_label["text"] = f"Power: {self.options.power:.4f}"
        self.update_params()

    def set_blend_mode(self, *args):
        self.blend_mode_listbox.selection_clear()
        self.options.blend_mode = self.blend_mode.get()
        self.update_params()

    def set_z_order(self, *args):
        self.z_order_spinbox.selection_clear()
        self.options.z_order = self.z_order_val.get()
        self.update_params()

    def draw(self):
        if self.drawing is None:
            self.drawing = self.ax.imshow(
                self.dbm_manager.get_dbm_data(),
                extent=(0.0, 1.0, 0.0, 1.0),
                interpolation="none",
                origin="lower",
                cmap="tab10",
                vmax=self.dbm_manager.n_classes - 1,
                vmin=0,
            )
            self.options.z_order = self.drawing.get_zorder()
            self.z_order_val.set(self.drawing.get_zorder())
        self.drawing.set_data(self.dbm_manager.get_dbm_data())
        self.drawing.set_visible(self.options.enabled)
        self.drawing.set_alpha(self.options.alpha)
        self.drawing.set_zorder(self.options.z_order)

        if self.options.show_borders:
            if self._borders_drawing is None:
                self._borders_drawing = self.ax.imshow(
                    self.dbm_manager.get_dbm_borders(),
                    cmap="gray",
                    interpolation="none",
                    extent=(0.0, 1.0, 0.0, 1.0),
                    origin="lower",
                )
        if self._borders_drawing is not None:
            self._borders_drawing.set_visible(
                self.options.enabled and self.options.show_borders
            )
            self._borders_drawing.set_zorder(self.options.z_order)

        if self.options.encode_distance:
            self.show_distance()

        return super().draw()

    def show_distance(self):
        from matplotlib.colors import LightSource, hsv_to_rgb, rgb_to_hsv

        ls = LightSource()
        rgba, *ignore = self.drawing.make_image(None, unsampled=True)
        rgb = rgba[..., :3] / 255.0

        distances = self.dbm_manager.get_distance_map()
        if distances is None:
            self.frame.after(200, self.show_distance)
            return

        if self.options.invert_distance:
            distances = -distances
        smallest, largest = distances.min(), distances.max()
        distances = (distances - smallest) / (largest - smallest)
        distances **= self.options.power

        blend = self.options.blend_mode
        if blend == "multiply":
            hsv = rgb_to_hsv(rgb)
            hsv[..., 1] *= distances

            self.drawing.set_data(hsv_to_rgb(hsv))
        elif blend == "hsv":
            self.drawing.set_data(
                ls.blend_hsv(
                    rgb,
                    distances[..., None],
                    hsv_min_val=0.3,
                    hsv_max_val=1.0,
                    hsv_min_sat=0.5,
                    hsv_max_sat=1.0,
                )
            )
        elif blend == "soft_light":
            blended = ls.blend_soft_light(rgb, distances[..., None])
            blended = (blended - blended.min()) / (blended.max() - blended.min())
            self.drawing.set_data(blended)
        elif blend == "overlay":
            blended = ls.blend_overlay(rgb, distances[..., None])
            blended = (blended - blended.min()) / (blended.max() - blended.min())
            self.drawing.set_data(blended)

        return super().draw()
