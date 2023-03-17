import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure


class DatapointFrame(tk.Frame):
    def __init__(self, master, image_data, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.image_data = image_data

        self.fig = Figure(dpi=100)
        self.ax = self.fig.add_subplot()
        self.vis = self.ax.imshow(
            self.image_data,
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()

        self.canvas.get_tk_widget().grid(column=0, row=0, padx=5, pady=5, sticky="NSEW")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
