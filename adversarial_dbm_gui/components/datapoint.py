import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class DatapointFrame(tk.Frame):
    def __init__(self, master, image_data, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.image_data = image_data

        self.fig = Figure(dpi=100, tight_layout=True)
        self.ax = self.fig.add_subplot(xmargin=0.0, ymargin=0.0, aspect=1.0)
        self.vis = self.ax.imshow(
            self.image_data,
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        self.ax.axis("off")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()

        self.canvas.get_tk_widget().grid(column=0, row=0, padx=5, pady=5, sticky="NSEW")
        self.canvas.get_tk_widget()["width"] = 100
        self.canvas.get_tk_widget()["height"] = 100

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
