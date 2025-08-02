# main.py
import tkinter as tk
from .dataset_viewer import DatasetViewer

def open_viewer_standalone(parent=None, key_list=None):
    if parent is None:
        win = tk.Tk()
    else:
        win = tk.Toplevel(parent)
    viewer = DatasetViewer(win, key_list)
    win.mainloop()
