import tkinter as tk
from tkinter import ttk


class DBMControls:
    def __init__(self, master: "Controls", label: str, id_for_event: str):
        self.frame = master

        self.id_for_event = id_for_event
        self.enabled = tk.BooleanVar(master, value=False)
        self.enabled_btn = ttk.Checkbutton(
            master,
            text=label,
            variable=self.enabled,
            onvalue=True,
            command=self.send_update,
        )

        self.alpha = tk.DoubleVar(master)
        self.alpha_slider = ttk.Scale(
            master,
            orient=tk.HORIZONTAL,
            from_=0.0,
            to=1.0,
            variable=self.alpha,
            command=self.send_update,
        )

    def grid(self, row: int, firstcolumn: int):
        self.enabled_btn.grid(row=row, column=firstcolumn, sticky="W", padx=5, pady=5)
        self.alpha_slider.grid(
            row=row, column=firstcolumn + 1, sticky="W", padx=5, pady=5
        )

    def send_update(self, *args):
        self.frame.notify(
            {
                "control_id": self.id_for_event,
                "enabled": self.enabled.get(),
                "alpha": self.alpha.get(),
            }
        )


class ScalarMapControls:
    def __init__(self, master: "Controls", label: str, id_for_event: str):
        self.frame = master
        self.id_for_event = id_for_event

        self.enabled = tk.BooleanVar(self.frame, value=False)
        self.enabled_btn = ttk.Checkbutton(
            self.frame,
            text=label,
            variable=self.enabled,
            onvalue=True,
            command=self.send_update,
        )

        self.alpha = tk.DoubleVar(self.frame)
        self.alpha_slider = ttk.Scale(
            self.frame,
            orient=tk.HORIZONTAL,
            from_=0.0,
            to=1.0,
            variable=self.alpha,
            command=self.send_update,
        )

        self.power = tk.DoubleVar(self.frame)
        self.power_entry = ttk.Entry(self.frame, textvariable=self.power, width=5)
        self.power_entry.bind("<Return>", self.send_update)

    def grid(self, row: int, firstcolumn: int):
        self.enabled_btn.grid(row=row, column=firstcolumn, sticky="W", padx=5, pady=5)
        self.alpha_slider.grid(
            row=row, column=firstcolumn + 1, sticky="W", padx=5, pady=5
        )
        self.power_entry.grid(
            row=row, column=firstcolumn + 2, sticky="W", padx=5, pady=5
        )

    def send_update(self, *args):
        self.frame.notify(
            {
                "control_id": self.id_for_event,
                "enabled": self.enabled.get(),
                "alpha": self.alpha.get(),
                "power": self.power.get(),
            }
        )


class Controls(tk.Frame):
    def __init__(self, master, *args, **kwargs) -> None:
        super().__init__(master, *args, **kwargs)
        self.master = master

        self.dbm_control = DBMControls(self, label="Show DBM", id_for_event="dbm")
        self.dbm_control.enabled.set(True)
        self.dbm_control.alpha.set(1.0)
        self.wormhole_control = DBMControls(
            self, label="Show Wormholes", id_for_event="wormhole"
        )
        self.distance_map_control = ScalarMapControls(
            self, label="Show distance map", id_for_event="dist_map"
        )
        self.confidence_control = ScalarMapControls(
            self, label="Show confidence map", id_for_event="conf_map"
        )

        self.controls = [
            self.dbm_control,
            self.wormhole_control,
            self.distance_map_control,
            self.confidence_control,
        ]

        self.dbm_control.grid(row=0, firstcolumn=0)
        self.wormhole_control.grid(row=1, firstcolumn=0)
        self.distance_map_control.grid(row=2, firstcolumn=0)
        self.confidence_control.grid(row=3, firstcolumn=0)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)

        self.grid_columnconfigure(0, weight=1, minsize=10)
        self.grid_columnconfigure(1, weight=1, minsize=10)
        self.grid_columnconfigure(2, weight=1, minsize=10)

        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def notify(self, event):
        for observer in self._observers:
            observer.update_params(event)

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
