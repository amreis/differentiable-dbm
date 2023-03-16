import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np

from .plotcontrols import Controls


class DBMPlot(tk.Frame):
    def __init__(self, master, dbm_data, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.dbm_data = dbm_data

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot()
        self.base_dbm = self.ax.imshow(
            self.dbm_data, extent=(0.0, 1.0, 0.0, 1.0), origin="lower", cmap="tab10"
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()

        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.update()

        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.canvas.mpl_connect("button_press_event", lambda e: print(e))

        self.canvas.get_tk_widget().grid(column=0, row=0, sticky="WNES")
        self.toolbar.grid(column=0, row=1, sticky="N")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

    def on_key_press(self, event):
        print(f"you pressed {event.key}")
        key_press_handler(event, self.canvas, self.toolbar)


def main():
    root = tk.Tk()
    root.wm_title("Embedding in Tk")

    content = tk.Frame(root)
    content.grid(column=0, row=0, sticky="NESW")
    plot = DBMPlot(content, np.random.randint(10, size=(28, 28)))

    plot.grid(column=0, row=0, sticky="NSEW", padx=5, pady=5)
    controls = Controls(content)
    controls.grid(column=1, row=0, sticky="NW", padx=5, pady=5)

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    content.grid_columnconfigure(0, weight=4)
    content.grid_columnconfigure(1, weight=1, minsize=10)
    content.grid_rowconfigure(0, weight=1)

    tk.mainloop()


if __name__ == "__main__":
    main()
