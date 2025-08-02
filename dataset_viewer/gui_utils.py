# gui_utils.py
import tkinter as tk
from tkinter import messagebox

def show_error(parent, msg, status_label=None):
    messagebox.showerror("Error", msg, parent=parent)
    if status_label:
        status_label.config(text=msg)

def show_info(parent, msg, status_label=None):
    messagebox.showinfo("Info", msg, parent=parent)
    if status_label:
        status_label.config(text=msg)

def set_status(label: tk.Label, msg: str):
    if label:
        label.config(text=msg)
