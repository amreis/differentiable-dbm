import tkinter as tk
from tkinter import ttk


class MenuBar(tk.Menu):
    def __init__(self, master=None, *args, **kwargs) -> None:
        super().__init__(master, *args, **kwargs)

        self.menu_file = tk.Menu(self)
        self.menu_edit = tk.Menu(self)

        self.add_cascade(menu=self.menu_file, label="File")
        self.add_cascade(menu=self.menu_edit, label="Edit")
