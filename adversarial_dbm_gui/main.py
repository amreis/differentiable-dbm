from .classifiers import logistic
from .data import load_mnist

import tkinter as tk
from tkinter import ttk

from .components import menubar


class MainWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root

        self.f = ttk.Frame(self.root)
        self.f.grid()

        self.menu = menubar.MenuBar(self.f)
        self.root["menu"] = self.menu

        self.b = ttk.Button(self.f, text="Start!", command=self.start)
        self.b.grid(column=1, row=0, padx=5, pady=5)
        self.l = ttk.Label(self.f, text="No Answer")
        self.l.grid(column=0, row=0, padx=5, pady=5)
        self.p = ttk.Progressbar(
            self.f, orient="horizontal", mode="determinate", maximum=20
        )
        self.p.grid(column=0, row=1, padx=5, pady=5)

        self.interrupt = False

    def start(self):
        self.b.configure(text="Stop", command=self.stop)
        self.l["text"] = "Working..."
        self.interrupt = False
        self.root.after(1, self.step)

    def step(self, count=0):
        self.p["value"] = count
        if self.interrupt:
            self.result(None)
            return
        self.root.after(100)
        if count == 20:
            self.result(42)
            return
        self.root.after(1, lambda: self.step(count + 1))

    def stop(self):
        self.interrupt = True

    def result(self, answer):
        self.p["value"] = 0
        self.b.configure(text="Start!", command=self.start)
        self.l["text"] = f"Answer: {answer}" if answer else "No Answer"


def calculate(*args):
    print(args)


def main():
    root = tk.Tk()
    root.option_add("*tearOff", tk.FALSE)
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
