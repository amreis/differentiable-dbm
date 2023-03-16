import tkinter as tk
from tkinter import ttk


class DBMControls:
    def __init__(self, master, label: str, row, *args, **kwargs):
        # self.grid(column=0, row=0)

        self.enabled = tk.BooleanVar(master, value=False)
        self.enabled_btn = ttk.Checkbutton(
            master,
            text=label,
            variable=self.enabled,
            onvalue=True,
            command=self.enabled_changed,
        )

        self.alpha = tk.DoubleVar(master)
        self.alpha_slider = ttk.Scale(
            master, orient=tk.HORIZONTAL, from_=0.0, to=1.0, variable=self.alpha
        )
        self.grid(row, 0)

    def grid(self, row: int, firstcolumn: int):
        self.enabled_btn.grid(row=row, column=firstcolumn, sticky="W", padx=5, pady=5)
        self.alpha_slider.grid(
            row=row, column=firstcolumn + 1, sticky="W", padx=5, pady=5
        )

    def enabled_changed(self):
        print("Enabled Changed")


class ScalarMapControls:
    def __init__(self, master, row, label: str):
        self.frame: tk.Frame = master
        # self.grid(column=0, row=0)

        self.enabled = tk.BooleanVar(self.frame, value=False)
        self.enabled_btn = ttk.Checkbutton(
            self.frame,
            text=label,
            variable=self.enabled,
            onvalue=True,
            command=self.enabled_changed,
        )

        self.alpha = tk.DoubleVar(self.frame)
        self.alpha_slider = ttk.Scale(
            self.frame, orient=tk.HORIZONTAL, from_=0.0, to=1.0, variable=self.alpha
        )

        self.power = tk.DoubleVar(self.frame)
        self.power_entry = ttk.Entry(self.frame, textvariable=self.power, width=5)
        self.power_entry.bind("<Return>", self.update_power)

        self.grid(row, 0)

    def grid(self, row: int, firstcolumn: int):
        self.enabled_btn.grid(row=row, column=firstcolumn, sticky="W", padx=5, pady=5)
        self.alpha_slider.grid(
            row=row, column=firstcolumn + 1, sticky="W", padx=5, pady=5
        )
        self.power_entry.grid(
            row=row, column=firstcolumn + 2, sticky="W", padx=5, pady=5
        )

    def enabled_changed(self):
        print("Enabled changed")

    def update_power(self):
        print("updating power...")


class Controls(tk.Frame):
    def __init__(self, master, *args, **kwargs) -> None:
        super().__init__(master, *args, **kwargs)
        self.master = master

        self.dbm_control = DBMControls(self, label="Show DBM", row=0)
        self.wormhole_control = DBMControls(self, label="Show Wormholes", row=1)
        self.distance_map_control = ScalarMapControls(
            self, label="Show distance map", row=2
        )
        self.confidence_control = ScalarMapControls(
            self, label="Show confidence map", row=3
        )

        self.controls = [
            self.dbm_control,
            self.wormhole_control,
            self.distance_map_control,
            self.confidence_control,
        ]

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)

        self.grid_columnconfigure(0, weight=1, minsize=10)
        self.grid_columnconfigure(1, weight=1, minsize=10)
        self.grid_columnconfigure(2, weight=1, minsize=10)

        for k, v in self.children.items():
            print(k)
            print(v.grid_info())

    def overlay_distances(self):
        print("Overlaying distances...")

    def invert_on_click_changed(self, *args):
        print("Invert on Click!")
        print(args)


def main():
    root = tk.Tk()

    controls = Controls(root)
    controls.grid(column=0, row=0)
    tk.mainloop()


if __name__ == "__main__":
    main()
