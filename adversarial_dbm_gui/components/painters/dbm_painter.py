import tkinter as tk
from dataclasses import dataclass
from functools import partial
from tkinter import ttk

import matplotlib.pyplot as plt

from ...compute.dbm_manager import DBMManager
from . import painter


@dataclass
class Options:
    enabled: bool = True
    encode_distance: bool = False
    show_borders: bool = False
    alpha: float = 1.0
    power: float = 1.0


class DBMPainter(painter.Painter):
    def __init__(self, ax: plt.Axes, master: tk.Frame, dbm_manager: DBMManager) -> None:
        super().__init__(ax, master)

        self._borders_drawing = None

        self.dbm_manager = dbm_manager
        self.options = Options()

        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)

        self.enabled = tk.BooleanVar(self.frame, value=self.options.enabled)
        self.enabled_btn = ttk.Checkbutton(
            self.frame,
            text="DBM",
            variable=self.enabled,
            onvalue=True,
            command=self.set_enabled,
        )

        self.options_btn = ttk.Button(
            self.frame, command=self.spawn_options, text="Options"
        )

        self.enabled_btn.grid(column=0, row=0, sticky=tk.EW)
        self.options_btn.grid(column=1, row=0, sticky=tk.NE)

        self._redraw_observers = []

    def attach_for_redraw(self, observer):
        self._redraw_observers.append(observer)

    def detach_for_redraw(self, observer):
        self._redraw_observers.remove(observer)

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
            text="Encode distance in Saturation",
            variable=self.distance_enabled,
            onvalue=True,
            command=self.set_encode_distance,
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

        self.power = tk.DoubleVar(value=self.options.power)
        power_transf_slider = ttk.Scale(
            options_window,
            command=self.set_power,
            variable=self.power,
            from_=0.0,
            to=2.0,
            orient=tk.HORIZONTAL,
        )
        self.power_transf_slider_label = ttk.Label(
            options_window, text=f"Power: {self.power.get():.4f}"
        )

        distance_btn.grid(column=0, row=0, columnspan=2, sticky=tk.NW)
        borders_enabled_btn.grid(column=0, row=1, columnspan=2, sticky=tk.NW)
        self.alpha_slider_label.grid(column=0, row=2, sticky=tk.NW)
        alpha_slider.grid(column=1, row=2, sticky=tk.NSEW)
        self.power_transf_slider_label.grid(column=0, row=3, sticky=tk.NW)
        power_transf_slider.grid(column=1, row=3, sticky=tk.NSEW)

        options_window.grid_rowconfigure(0, weight=1)
        options_window.grid_rowconfigure(1, weight=1)
        options_window.grid_rowconfigure(2, weight=1)
        options_window.grid_rowconfigure(3, weight=1)

        options_window.grid_columnconfigure(0, weight=1)
        options_window.grid_columnconfigure(1, weight=2)

    def set_enabled(self, *args):
        self.options.enabled = self.enabled.get()
        self.update_params()

    def set_encode_distance(self, *args):
        self.options.encode_distance = self.distance_enabled.get()
        self.update_params()

    def set_show_borders(self, *args):
        self.options.show_borders = self.borders_enabled.get()
        self.update_params()

    def set_alpha(self, *args):
        self.options.alpha = self.alpha_val.get()
        self.alpha_slider_label["text"] = f"Alpha: {self.alpha_val.get():.4f}"
        self.update_params()

    def set_power(self, *args):
        self.options.power = self.power.get()
        self.power_transf_slider_label["text"] = f"Power: {self.power.get():.4f}"
        self.update_params()

    def draw(self):
        if self.drawing is None:
            self.drawing = self.ax.imshow(
                self.dbm_manager.get_dbm_data(),
                extent=(0.0, 1.0, 0.0, 1.0),
                interpolation="none",
                origin="lower",
                cmap="tab10",
            )
        self.drawing.set_data(self.dbm_manager.get_dbm_data())
        self.drawing.set_visible(self.options.enabled)
        self.drawing.set_alpha(self.options.alpha)

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
            self._borders_drawing.set_visible(self.options.enabled and self.options.show_borders)

        if self.options.encode_distance:
            self.show_distance()

        for obs in self._redraw_observers:
            obs.redraw()

    def show_distance(self):
        from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

        rgba, *ignore = self.drawing.make_image(None, unsampled=True)
        rgb = rgba[..., :3] / 255.0
        hsv = rgb_to_hsv(rgb)
        distances = self.dbm_manager.get_distance_map()
        if distances is None:
            self.frame.after(200, self.show_distance)
            return
        smallest, largest = distances.min(), distances.max()
        distances = (distances - smallest) / (largest - smallest)
        distances **= self.options.power

        hsv[:, :, 1] *= distances

        self.drawing.set_data(hsv_to_rgb(hsv))
        for obs in self._redraw_observers:
            obs.redraw()
