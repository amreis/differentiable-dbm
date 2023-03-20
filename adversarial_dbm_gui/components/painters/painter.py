import abc
import tkinter as tk

import matplotlib.pyplot as plt
import matplotlib.artist


class Painter(abc.ABC):
    def __init__(self, ax: plt.Axes, master: tk.Frame) -> None:
        super().__init__()

        self.ax = ax
        self.options = None
        self.master = master
        self.frame = tk.Frame(master)
        self.grid = self.frame.grid  # delegating grid() to the parent frame.
        self.drawing: matplotlib.artist.Artist = None
        self.options = dict()

    @abc.abstractmethod
    def draw(self):
        ...
