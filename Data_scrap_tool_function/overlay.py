# overlay.py
import tkinter as tk

class SnipOverlay(tk.Toplevel):
    def __init__(self, master, monitor, on_complete):
        """
        Transparent overlay limited to a selected monitor.
        User drags to choose a rectangle; on_complete is called with global region.
        """
        super().__init__(master)
        self.on_complete = on_complete
        self.monitor = monitor
        left = monitor.get("left", 0)
        top = monitor.get("top", 0)
        width = monitor.get("width", 0)
        height = monitor.get("height", 0)
        self.geometry(f"{width}x{height}+{left}+{top}")
        self.attributes("-topmost", True)
        self.config(cursor="cross")
        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.start_x = self.start_y = None
        self.rect = None

        label = tk.Label(self, text="Drag to select region. ESC to cancel.", fg="white", bg="black")
        label.place(x=10, y=10)
        self.attributes("-alpha", 0.25)

        self.bind("<ButtonPress-1>", self.on_button_press)
        self.bind("<B1-Motion>", self.on_move)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<Escape>", lambda e: self.destroy())

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=2
        )

    def on_move(self, event):
        curX = self.canvas.canvasx(event.x)
        curY = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def on_release(self, event):
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        left_rel = int(min(self.start_x, end_x))
        top_rel = int(min(self.start_y, end_y))
        width = int(abs(end_x - self.start_x))
        height = int(abs(end_y - self.start_y))
        mon = self.monitor
        global_region = (left_rel + mon.get("left", 0), top_rel + mon.get("top", 0), width, height)
        self.on_complete(global_region)
        self.destroy()