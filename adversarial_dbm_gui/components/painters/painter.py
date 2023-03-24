import abc
import tkinter as tk

import matplotlib.artist
import matplotlib.pyplot as plt


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

        self._redraw_observers = []

    def attach_for_redraw(self, observer):
        self._redraw_observers.append(observer)

    def detach_for_redraw(self, observer):
        self._redraw_observers.remove(observer)

    @abc.abstractmethod
    def draw(self):
        for obs in self._redraw_observers:
            obs.redraw()
