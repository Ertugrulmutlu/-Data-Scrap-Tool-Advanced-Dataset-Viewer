# Data_scrap_tool.py
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import mss
from PIL import Image, ImageTk
import numpy as np
import os
import tkinter.simpledialog as simpledialog
import pathlib
import datetime
import cv2  # eklendi

import Data_scrap_tool_function.data_collector as dc
import dataset_viewer.viewer as viewer
from Data_scrap_tool_function.capture import list_monitors, get_preview_image, PREVIEW_SIZE
from Data_scrap_tool_function.overlay import SnipOverlay

WINDOW_SIZE = "950x600"

def parse_key_list(text):
    parts = [p.strip() for p in text.replace(';', ',').replace('|', ',').split(',') if p.strip()]
    cleaned = []
    for p in parts:
        if len(p) == 1:
            cleaned.append(p)
    return cleaned

class RecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Modular Snip Action Recorder")
        self.root.geometry(WINDOW_SIZE)
        self.root.resizable(False, False)

        self.status_var = tk.StringVar(value="No region selected.")
        self.preview_imgtk = None
        self.monitor_map = {}
        self.selected_region = None

        self.key_list = dc.KEY_LIST.copy()

        # video kaydı için alanlar
        self.video_writer = None
        self.video_fps = 30
        self.video_filename = None
        self._video_base_name = "dataset"  # placeholder

        # frame counter (video içindeki frame index)
        self.video_frame_counter = 0

        self._build_ui()
        threading.Thread(target=self._preview_loop, daemon=True).start()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.grid(sticky="nsew")
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)

        # Left: settings + controls
        left = ttk.LabelFrame(main, text="Controls / Settings", padding=10)
        left.grid(column=0, row=0, sticky="ns", padx=(0, 10))

        # Key settings button and display
        ttk.Button(left, text="Key Settings", command=self.open_key_settings).grid(column=0, row=0, columnspan=3, pady=(0,6), sticky="ew")
        self.tracking_label = ttk.Label(left, text=f"Tracking keys: {self.key_list}")
        self.tracking_label.grid(column=0, row=1, columnspan=3, sticky="w", pady=2)

        # Monitor selection
        ttk.Label(left, text="1. Select monitor:").grid(column=0, row=2, sticky="w", pady=2)
        self.monitor_var = tk.StringVar()
        monitors = list_monitors()
        names = []
        for name, mon in monitors:
            names.append(name)
            self.monitor_map[name] = mon
        self.monitor_combo = ttk.Combobox(
            left, values=names, textvariable=self.monitor_var, state="readonly", width=20
        )
        self.monitor_combo.grid(column=1, row=2, padx=5, pady=2)
        if names:
            self.monitor_var.set(names[1] if len(names) > 1 else names[0])

        # Snip region
        ttk.Label(left, text="2. Snip region:").grid(column=0, row=3, sticky="w", pady=2)
        ttk.Button(left, text="Snip Region", command=self.launch_snip).grid(column=1, row=3, padx=5, pady=2)

        ttk.Separator(left, orient="horizontal").grid(column=0, row=4, columnspan=3, sticky="ew", pady=8)

        # Stop & Save
        ttk.Label(left, text="3. Stop & Save:").grid(column=0, row=5, sticky="w", pady=2)
        self.stop_btn = ttk.Button(left, text="Stop and Save", command=self.stop_recording, state="disabled")
        self.stop_btn.grid(column=1, row=5, padx=5, pady=2)

        ttk.Button(left, text="Open Dataset Viewer", command=lambda: viewer.open_viewer_standalone(self.root, self.key_list))\
                    .grid(column=0, row=6, columnspan=3, pady=(15, 2), sticky="ew")

        # Status
        ttk.Label(left, textvariable=self.status_var, foreground="blue").grid(column=0, row=7, columnspan=3, sticky="w", pady=(10, 0))

        # Right: preview
        right = ttk.LabelFrame(main, text="Live Preview", padding=10)
        right.grid(column=1, row=0, sticky="nsew")
        preview_container = tk.Frame(right, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1])
        preview_container.grid_propagate(False)
        preview_container.grid(column=0, row=0, padx=5, pady=5)
        self.preview_label = tk.Label(preview_container, text="Inactive", bg="#222", fg="white")
        self.preview_label.place(x=0, y=0, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1])
        self.active_keys_label = ttk.Label(right, text="Active keys: []")
        self.active_keys_label.grid(column=0, row=1, pady=4)

    def open_key_settings(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Key Settings")
        dlg.resizable(False, False)
        dlg.grab_set()

        lb = tk.Listbox(dlg, height=6)
        for k in self.key_list:
            lb.insert(tk.END, k)
        lb.grid(row=0, column=0, columnspan=3, padx=10, pady=5)

        ttk.Label(dlg, text="New key (single char):").grid(row=1, column=0, padx=5, sticky="e")
        new_key_var = tk.StringVar()
        new_key_entry = ttk.Entry(dlg, textvariable=new_key_var, width=5)
        new_key_entry.grid(row=1, column=1, padx=5, sticky="w")

        def add_key():
            v = new_key_var.get().strip()
            if len(v) != 1:
                messagebox.showwarning("Invalid", "Enter exactly one character.")
                return
            if v in self.key_list:
                messagebox.showinfo("Exists", f"Key '{v}' already tracked.")
                return
            self.key_list.append(v)
            lb.insert(tk.END, v)
            new_key_var.set("")

        def remove_selected():
            sel = lb.curselection()
            if not sel:
                return
            idx = sel[0]
            key = lb.get(idx)
            lb.delete(idx)
            if key in self.key_list:
                self.key_list.remove(key)

        ttk.Button(dlg, text="Add", command=add_key).grid(row=1, column=2, padx=5)
        ttk.Button(dlg, text="Remove Selected", command=remove_selected).grid(row=2, column=0, columnspan=3, pady=5)

        def apply_and_close():
            if not self.key_list:
                messagebox.showwarning("Invalid", "Need at least one key.")
                return
            dc.KEY_LIST = self.key_list.copy()
            dc.stop_keyboard_listener()  # Önce durdur!
            dc.start_keyboard_listener() # Sonra yeniyle başlat!
            self.tracking_label.config(text=f"Tracking keys: {self.key_list}")
            self.update_status(f"Tracking keys updated: {self.key_list}")
            dlg.destroy()

        btn_frame = ttk.Frame(dlg)
        btn_frame.grid(row=3, column=0, columnspan=3, pady=(0,10))
        ttk.Button(btn_frame, text="Apply", command=apply_and_close).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Close", command=dlg.destroy).grid(row=0, column=1, padx=5)

    def launch_snip(self):
        sel = self.monitor_var.get()
        mon = self.monitor_map.get(sel)
        if mon is None:
            messagebox.showerror("Error", "No monitor selected.")
            return
        SnipOverlay(self.root, monitor=mon, on_complete=self._on_region_selected)
        self.update_status("Selecting region...")
        self.stop_btn.config(state="disabled")

    def _on_region_selected(self, region):
        self.selected_region = region
        dc.selected_region_global = region
        dc.recording = True
        dc.start_keyboard_listener()
        self.enable_stop()
        # video kaydını placeholder ile başlat; gerçek base name stop'ta ayarlanacak
        self._video_base_name = "dataset"
        self.video_frame_counter = 0  # reset frame counter
        dc.current_video_frame = 0
        self.start_video_recording(self._video_base_name)
        self.update_status(f"Region selected: {self.selected_region}. Recording started.")

    def enable_stop(self):
        self.stop_btn.config(state="normal")

    def start_video_recording(self, base_name: str):
        if self.video_writer is not None:
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_base = base_name.strip() if base_name and base_name.strip() else "dataset"
        self._video_base_name = safe_base
        filename = f"{safe_base}_{timestamp}.mp4"
        target_dir = "recordings"  # değişti: videolar buraya
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width, height = PREVIEW_SIZE
        self.video_writer = cv2.VideoWriter(path, fourcc, self.video_fps, (width, height))
        self.video_filename = path
        self.video_frame_counter = 0
        dc.current_video_frame = 0
        self.update_status(f"Video recording started: {os.path.basename(path)}")

    def stop_video_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            video_path = self.video_filename
            self.video_filename = None
            self.update_status(f"Video saved -> {video_path}")
            return video_path
        return None

    def stop_recording(self):
        if not dc.recording:
            return

        dc.recording = False
        dc.stop_keyboard_listener()

        name = simpledialog.askstring(
            "Save As",
            "Enter dataset name (leave empty to use default/overwrite):",
            parent=self.root
        )

        # normalize: uzantı ne olursa olsun sadece stem al (a, a.npy, a.npz => "a")
        if name and name.strip():
            filename_base = pathlib.Path(name.strip()).stem
        else:
            filename_base = "dataset"

        filename = f"{filename_base}.npz"  # tek uzantı .npz

        target_dir = "collected"
        os.makedirs(target_dir, exist_ok=True)
        target_path = pathlib.Path(target_dir) / filename

        path, count = dc.save_dataset_npz(str(target_path))
        self.update_status(f"{count} samples saved -> {path}")
        messagebox.showinfo("Done", f"{count} new samples appended/saved: {path}")

        # video kaydını durdur ve dataset ismine göre yeniden adlandır
        video_path = self.stop_video_recording()
        if video_path:
            # timestamp kısmını orijinal video adından al
            orig_name = os.path.basename(video_path)
            if "_" in orig_name:
                parts = orig_name.rsplit("_", 1)
                if len(parts) == 2 and parts[1].endswith(".mp4"):
                    timestamp_part = parts[1].replace(".mp4", "")
                else:
                    timestamp_part = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            else:
                timestamp_part = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            desired_video_name = f"{filename_base}_{timestamp_part}.mp4"
            desired_video_path = os.path.join("recordings", desired_video_name)  # recordings içinde
            try:
                os.replace(video_path, desired_video_path)
                video_path = desired_video_path
            except Exception:
                pass
            messagebox.showinfo("Video Saved", f"Live preview video saved: {video_path}")

        dc.pressed_times.clear()
        self.preview_label.config(image="", text="Inactive")
        self.active_keys_label.config(text="Active keys: []")
        dc.data.clear()

    def update_status(self, msg):
        self.status_var.set(msg)

    def _preview_loop(self):
        while True:
            if dc.recording and self.selected_region is not None:
                left, top, width, height = self.selected_region
                with mss.mss() as sct:
                    frame = np.array(sct.grab({"left": left, "top": top, "width": width, "height": height}))
                active = [k for k in dc.pressed_times.keys() if k in self.key_list]
                preview_img = get_preview_image(frame, active, self.key_list)
                im = Image.fromarray(preview_img)
                imgtk = ImageTk.PhotoImage(image=im)
                self.preview_imgtk = imgtk
                self.preview_label.config(image=imgtk, text="")
                self.active_keys_label.config(text=f"Active keys: {active}")

                # her preview frame'i videoya yaz ve frame counter güncelle
                if self.video_writer is not None:
                    try:
                        bgr = cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR)
                        self.video_writer.write(bgr)
                        # frame counter artır ve data_collector'a bildir
                        self.video_frame_counter += 1
                        dc.current_video_frame = self.video_frame_counter
                    except Exception:
                        pass
            else:
                self.active_keys_label.config(text="Active keys: []")
            time.sleep(1 / self.video_fps)

if __name__ == "__main__":
    root = tk.Tk()
    app = RecorderGUI(root)
    root.mainloop()
