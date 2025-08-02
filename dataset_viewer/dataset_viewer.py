# dataset_viewer.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import pathlib
import subprocess
import sys
import webbrowser
import urllib.parse

import socket
import time
import threading
import shutil
import os
import datetime
from pathlib import Path

from .constants import PREVIEW_SIZE, DEFAULT_KEY_LIST
from .video_player import EmbeddedVideoPlayer
from .io_utils import find_linked_video
from .gui_utils import show_error, show_info, set_status

class DatasetViewer:
    def __init__(self, root, key_list=None):
        self.root = root
        self.root.title("Dataset Viewer")
        self.root.geometry("1000x760")
        self.root.configure(bg="#f7f7f7")

        # internal
        self._img_cache = []
        self.data = []
        self.index = 0
        self.dataset_path = None

        self.key_list = key_list if key_list is not None else DEFAULT_KEY_LIST.copy()
        self.data_key_list = self.key_list

        # filter state
        self.filter_keys = set(self.data_key_list)
        self.filter_phases = {"press", "release"}
        self.include_empty_keys = True
        self.filtered_indices = []

        self.video_player_visible = False
        self._base_width = None

        self._build_ui()
        self._bind_keys()

        self.root.after(100, self._store_base_width)


    def _store_base_width(self):
        self.root.update_idletasks()
        self._base_width = self.root.winfo_width()

    def _format_key_list(self):
        return [str(k) for k in self.data_key_list]

    def _build_ui(self):
        header = ttk.Frame(self.root)
        header.pack(fill=tk.X, padx=12, pady=6)
        self.keymap_label = ttk.Label(header, text=f"Key Mapping: {self._format_key_list()}", font=("Courier", 10))
        self.keymap_label.pack(side=tk.LEFT, anchor="w")

        # control buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=12, pady=(4, 0))

        self.open_btn = ttk.Button(btn_frame, text="Open Dataset", command=self.open_dataset)
        self.open_btn.grid(row=0, column=0, padx=6, pady=4)
        ttk.Button(btn_frame, text="<< Previous", command=self.prev_item).grid(row=0, column=1, padx=6)
        ttk.Button(btn_frame, text="Next >>", command=self.next_item).grid(row=0, column=2, padx=6)
        ttk.Button(btn_frame, text="Delete", command=self.delete_current).grid(row=0, column=3, padx=6)
        ttk.Button(btn_frame, text="Edit", command=self.edit_current).grid(row=0, column=4, padx=6)
        self.play_video_btn = ttk.Button(btn_frame, text="Play Preview Video", command=self._open_linked_video)
        self.play_video_btn.grid(row=0, column=5, padx=6)
        self.filter_btn = ttk.Button(btn_frame, text="Filters", command=self._open_filter_window)
        self.filter_btn.grid(row=0, column=6, padx=6)

        # navigation
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(fill=tk.X, padx=12, pady=6)
        ttk.Label(nav_frame, text="Go to image #:").pack(side=tk.LEFT)
        self.goto_var = tk.StringVar()
        goto_entry = ttk.Entry(nav_frame, textvariable=self.goto_var, width=6)
        goto_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(nav_frame, text="Go", command=self.goto_index).pack(side=tk.LEFT)

        # content
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # left: video + image + sync
        left_composite = ttk.Frame(content_frame)
        left_composite.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.video_player = EmbeddedVideoPlayer(left_composite)

        image_container = ttk.Frame(left_composite, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1], relief=tk.SUNKEN)
        image_container.grid(row=0, column=1, sticky="n", pady=(0, 8), padx=(15, 0))
        image_container.pack_propagate(False)
        self.image_label = ttk.Label(image_container, text="No data loaded", anchor="center", background="#222", foreground="white")
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")

        self._sync_btn_container = ttk.Frame(left_composite)
        self._sync_btn_container.grid(row=1, column=1, sticky="n", pady=(2, 8))

        # right metadata
        meta_container = ttk.Frame(content_frame)
        meta_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(meta_container, text="Metadata", font=("Segoe UI", 11, "bold")).pack(anchor="nw", pady=(0, 4))
        self.meta_text = tk.Text(
            meta_container,
            height=20,
            font=("Courier", 10),
            wrap="word",
            state="disabled",
            background="#ffffff",
            relief=tk.SOLID,
            borderwidth=1,
        )
        self.meta_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        scrollbar = ttk.Scrollbar(meta_container, orient="vertical", command=self.meta_text.yview)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.meta_text.config(yscrollcommand=scrollbar.set)

        status = ttk.Frame(self.root)
        status.pack(fill=tk.X, padx=12, pady=(0, 4))
        self.status_label = ttk.Label(status, text="Ready", anchor="w", font=("Segoe UI", 9))
        self.status_label.pack(side=tk.LEFT, fill=tk.X)

        self.streamlit_btn = ttk.Button(btn_frame, text="Review with Streamlit", command=self._open_in_streamlit)
        self.streamlit_btn.grid(row=0, column=7, padx=6)
    def _bind_keys(self):
        self.root.bind("<Left>", lambda e: self.prev_item())
        self.root.bind("<Right>", lambda e: self.next_item())
        self.root.bind("<Control-o>", lambda e: self.open_dataset())
        self.root.bind("<Delete>", lambda e: self.delete_current())

    def _open_filter_window(self):
        win = tk.Toplevel(self.root)
        win.title("Filters")
        win.resizable(False, False)
        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Key filters
        ttk.Label(frame, text="Keys:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self._filter_key_vars = {}
        keys_frame = ttk.Frame(frame)
        keys_frame.grid(row=1, column=0, sticky="w", pady=(0, 8))
        for i, k in enumerate(self.data_key_list):
            var = tk.BooleanVar(value=(k in self.filter_keys))
            cb = ttk.Checkbutton(keys_frame, text=k, variable=var, command=self._apply_filters)
            cb.grid(row=0, column=i, padx=4)
            self._filter_key_vars[k] = var

        # Phase filters
        ttk.Label(frame, text="Phases:", font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky="w")
        self._filter_phase_vars = {}
        phase_frame = ttk.Frame(frame)
        phase_frame.grid(row=3, column=0, sticky="w", pady=(0, 8))
        for j, ph in enumerate(["press", "release"]):
            var = tk.BooleanVar(value=(ph in self.filter_phases))
            cb = ttk.Checkbutton(phase_frame, text=ph, variable=var, command=self._apply_filters)
            cb.grid(row=0, column=j, padx=4)
            self._filter_phase_vars[ph] = var

        # include empty-key entries
        self._include_empty_var = tk.BooleanVar(value=self.include_empty_keys)
        ttk.Checkbutton(frame, text="Include entries with no keys", variable=self._include_empty_var, command=self._apply_filters).grid(row=4, column=0, sticky="w", pady=(4, 4))

        # Buttons
        btns = ttk.Frame(frame)
        btns.grid(row=5, column=0, pady=6, sticky="ew")
        ttk.Button(btns, text="Apply", command=lambda: [self._apply_filters(), win.destroy()]).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Reset", command=self._reset_filters).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT, padx=4)

    def _reset_filters(self):
        self.filter_keys = set(self.data_key_list)
        self.filter_phases = {"press", "release"}
        self.include_empty_keys = True
        if hasattr(self, "_filter_key_vars"):
            for k, var in self._filter_key_vars.items():
                var.set(True)
        if hasattr(self, "_filter_phase_vars"):
            for ph, var in self._filter_phase_vars.items():
                var.set(True)
        if hasattr(self, "_include_empty_var"):
            self._include_empty_var.set(True)
        self._apply_filters()

    def _apply_filters(self):
        if hasattr(self, "_filter_key_vars"):
            self.filter_keys = {k for k, var in self._filter_key_vars.items() if var.get()}
        if hasattr(self, "_filter_phase_vars"):
            self.filter_phases = {ph for ph, var in self._filter_phase_vars.items() if var.get()}
        if hasattr(self, "_include_empty_var"):
            self.include_empty_keys = bool(self._include_empty_var.get())

        self.filtered_indices = []
        for idx, entry in enumerate(self.data):
            if not self._entry_matches_filters(entry):
                continue
            self.filtered_indices.append(idx)
        if self.filtered_indices:
            self.index = self.filtered_indices[0]
        else:
            self.index = 0
        self.show_current()

    def open_dataset(self):
        path = filedialog.askopenfilename(filetypes=[("NumPyZ Files", "*.npz")])
        if not path:
            return
        try:
            npz = np.load(path, allow_pickle=True)
            self.data = list(npz["data"])
            raw_keys = list(npz["keys"]) if "keys" in npz else self.key_list
            self.data_key_list = [str(k) for k in raw_keys]
            self.index = 0
            self.dataset_path = path
            self.keymap_label.config(text=f"Key Mapping: {self._format_key_list()}")
            self.filter_keys = set(self.data_key_list)
            self.filter_phases = {"press", "release"}
            self.include_empty_keys = True
            self._reset_filters()
            set_status(self.status_label, f"Loaded {len(self.data)} entries.")
        except Exception as e:
            set_status(self.status_label, f"Failed to load dataset: {e}")
            show_error(self.root, f"Error loading dataset:\n{e}", status_label=self.status_label)
            self.image_label.config(text="Error", image="")
            self._clear_meta()

    def _open_linked_video(self):
        if not self.dataset_path:
            show_info(self.root, "No dataset loaded.", status_label=self.status_label)
            return

        if self.video_player_visible:
            self.video_player.grid_remove()
            self.video_player_visible = False
            if self._base_width:
                h = self.root.winfo_height()
                self.root.geometry(f"{self._base_width}x{h}")
            return

        # Strictly match video file for the current .npz (a.npz -> a_*.mp4)
        video_path = find_linked_video(self.dataset_path)
        if not video_path:
            show_info(
                self.root,
                f"No linked video found for dataset '{pathlib.Path(self.dataset_path).name}'.",
                status_label=self.status_label
            )
            return

        self.video_player.grid(row=0, column=0, sticky="n", pady=(0, 8), padx=(0, 10))
        self.video_player_visible = True
        self.video_player.load_video(video_path)

        self.root.update_idletasks()
        if self._base_width is None:
            self._store_base_width()
        vp_w = self.video_player.winfo_reqwidth()
        curr_w = self.root.winfo_width()
        extra = vp_w + 30
        if curr_w < (self._base_width + extra):
            new_w = self._base_width + extra
            h = self.root.winfo_height()
            self.root.geometry(f"{new_w}x{h}")


    def _entry_matches_filters(self, entry):
        phase = entry[4] if len(entry) > 4 else ""
        if phase not in self.filter_phases:
            return False
        multi_hot = np.array(entry[2]) if len(entry) > 2 else np.zeros(len(self.data_key_list))
        active_keys = [str(self.data_key_list[i]) for i, v in enumerate(multi_hot) if v == 1]
        if active_keys:
            if not any(k in self.filter_keys for k in active_keys):
                return False
        else:
            if not self.include_empty_keys:
                return False
        return True

    def show_current(self):
        if not self.data:
            self.image_label.config(image="", text="No data loaded")
            self._clear_meta()
            return

        if self.filtered_indices and self.index not in self.filtered_indices:
            self.index = self.filtered_indices[0]
        entry = self.data[self.index]
        if not self._entry_matches_filters(entry):
            for idx in self.filtered_indices:
                if idx >= self.index:
                    self.index = idx
                    entry = self.data[self.index]
                    break

        keys_arr = np.array(entry[2]) if len(entry) > 2 else np.zeros(len(self.data_key_list))
        durations_arr = np.round(np.array(entry[3]) if len(entry) > 3 else np.zeros(len(self.data_key_list)), 3)
        phase = entry[4] if len(entry) > 4 else "?"

        try:
            img_arr = (entry[0] * 255).astype(np.uint8)
            img_arr = cv2.resize(img_arr, PREVIEW_SIZE, interpolation=cv2.INTER_NEAREST)
            im = Image.fromarray(img_arr)
            imgtk = ImageTk.PhotoImage(image=im)
            self.image_label.config(image=imgtk, text="")
            self._img_cache.append(imgtk)
            if len(self._img_cache) > 12:
                self._img_cache = self._img_cache[-12:]
        except Exception as e:
            self.image_label.config(text="Image Error", image="")
            set_status(self.status_label, f"Failed to render image: {e}")

        active_keys = [str(self.data_key_list[i]) for i, v in enumerate(keys_arr) if v == 1 and i < len(self.data_key_list)]

        if self.filtered_indices:
            try:
                pos = self.filtered_indices.index(self.index) + 1
            except ValueError:
                pos = 1
            total_filtered = len(self.filtered_indices)
            index_str = f"{pos}/{total_filtered}"
        else:
            index_str = f"{self.index + 1}/{len(self.data)}"

        meta_lines = [
            f"Index: {index_str}",
            f"Active Keys: {active_keys if active_keys else 'None'}",
            f"Hold Durations (s): {[float(d) for d in durations_arr.tolist()]}",
            f"Phase: {phase}",
            f"Key Mapping: {self._format_key_list()}",
        ]
        video_frame_idx = entry[5] if len(entry) > 5 else None
        if video_frame_idx is not None and video_frame_idx != -1:
            meta_lines.append(f"Linked video frame: {video_frame_idx}")
        self._set_meta("\n".join(meta_lines))

        if video_frame_idx is not None and video_frame_idx != -1:
            def do_sync():
                if not self.video_player_visible:
                    self._open_linked_video()
                self.video_player.seek_to(video_frame_idx)
            if not hasattr(self, "_sync_btn_image"):
                self._sync_btn_image = ttk.Button(self._sync_btn_container, text="Sync Video to Sample", command=do_sync)
                self._sync_btn_image.pack()
            else:
                self._sync_btn_image.config(command=do_sync)
        else:
            if hasattr(self, "_sync_btn_image"):
                self._sync_btn_image.destroy()
                delattr(self, "_sync_btn_image")

    def _set_meta(self, text: str):
        self.meta_text.config(state="normal")
        self.meta_text.delete("1.0", tk.END)
        self.meta_text.insert(tk.END, text)
        self.meta_text.config(state="disabled")

    def _clear_meta(self):
        self._set_meta("")

    def next_item(self):
        if self.filtered_indices:
            try:
                pos = self.filtered_indices.index(self.index)
                if pos < len(self.filtered_indices) - 1:
                    self.index = self.filtered_indices[pos + 1]
                else:
                    set_status(self.status_label, "Already at last filtered item.")
            except ValueError:
                self.index = self.filtered_indices[0]
        else:
            if self.data and self.index < len(self.data) - 1:
                self.index += 1
            else:
                set_status(self.status_label, "Already at last item.")
        self.show_current()

    def prev_item(self):
        if self.filtered_indices:
            try:
                pos = self.filtered_indices.index(self.index)
                if pos > 0:
                    self.index = self.filtered_indices[pos - 1]
                else:
                    set_status(self.status_label, "Already at first filtered item.")
            except ValueError:
                self.index = self.filtered_indices[0]
        else:
            if self.data and self.index > 0:
                self.index -= 1
            else:
                set_status(self.status_label, "Already at first item.")
        self.show_current()

    def goto_index(self):
        try:
            idx = int(self.goto_var.get())
            if idx < 1:
                raise ValueError("Number must be >= 1.")
            if self.filtered_indices:
                if idx > len(self.filtered_indices):
                    raise IndexError(f"Filtered image #{idx} does not exist.")
                self.index = self.filtered_indices[idx - 1]
            else:
                if idx > len(self.data):
                    raise IndexError(f"Image #{idx} does not exist.")
                self.index = idx - 1
            self.show_current()
        except ValueError:
            show_error(self.root, "Please enter a valid positive integer.", status_label=self.status_label)
        except IndexError as e:
            show_error(self.root, str(e), status_label=self.status_label)

    def delete_current(self):
        if not self.data:
            show_info(self.root, "No data to delete.", status_label=self.status_label)
            return
        if not (0 <= self.index < len(self.data)):
            return
        answer = messagebox.askyesno("Confirm", f"Are you sure you want to delete image #{self.index + 1}?")
        if not answer:
            return
        try:
            del self.data[self.index]
            if self.dataset_path:
                if self.data:
                    np.savez(self.dataset_path, data=np.array(self.data, dtype=object), keys=np.array(self._format_key_list()))
                else:
                    np.savez(self.dataset_path, data=np.array([], dtype=object), keys=np.array(self._format_key_list()))
            self._apply_filters()
            self.show_current()
            set_status(self.status_label, "Current entry deleted.")
        except Exception as e:
            show_error(self.root, f"Deletion failed: {e}", status_label=self.status_label)

    def edit_current(self):
        if not self.data or not (0 <= self.index < len(self.data)):
            return
        entry = self.data[self.index]
        key_names = [str(k) for k in self.data_key_list]

        keys = np.array(entry[2]) if len(entry) > 2 else np.zeros(len(key_names))
        durations = np.round(np.array(entry[3]) if len(entry) > 3 else np.zeros(len(key_names)), 3)
        phase = entry[4] if len(entry) > 4 else "complete"

        edit_win = tk.Toplevel(self.root)
        edit_win.title("Edit Sample")
        edit_win.transient(self.root)
        edit_win.grab_set()

        frame = ttk.Frame(edit_win, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Edit Keys:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", pady=4)
        var_keys = []
        keys_frame = ttk.Frame(frame)
        keys_frame.grid(row=1, column=0, sticky="w", pady=(0, 8))
        for i, k in enumerate(key_names):
            val = int(keys[i]) if i < len(keys) else 0
            var = tk.IntVar(value=val)
            cb = ttk.Checkbutton(keys_frame, text=k, variable=var)
            cb.pack(side=tk.LEFT, padx=4)
            var_keys.append(var)

        ttk.Label(frame, text="Edit Hold Durations (s):", font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky="w", pady=4)
        holds_frame = ttk.Frame(frame)
        holds_frame.grid(row=3, column=0, sticky="w", pady=(0, 8))
        ent_holds = []
        for i, k in enumerate(key_names):
            val = str(durations[i]) if i < len(durations) else "0.0"
            sv = tk.StringVar(value=val)
            lbl = ttk.Label(holds_frame, text=f"{k}:")
            lbl.pack(side=tk.LEFT, padx=(0, 2))
            e = ttk.Entry(holds_frame, textvariable=sv, width=6)
            e.pack(side=tk.LEFT, padx=(0, 8))
            ent_holds.append(sv)

        ttk.Label(frame, text="Phase:", font=("Segoe UI", 10, "bold")).grid(row=4, column=0, sticky="w", pady=4)
        phase_var = tk.StringVar(value=phase)
        phase_menu = ttk.OptionMenu(frame, phase_var, phase, "press", "release")
        phase_menu.grid(row=5, column=0, sticky="w", pady=(0, 8))

        btns = ttk.Frame(frame)
        btns.grid(row=6, column=0, pady=6)
        ttk.Button(btns, text="Apply", command=lambda: self._apply_edit(edit_win, entry, var_keys, ent_holds, phase_var)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="Cancel", command=edit_win.destroy).pack(side=tk.LEFT, padx=5)

    def _apply_edit(self, edit_win, old_entry, var_keys, ent_holds, phase_var):
        try:
            new_keys = np.array([v.get() for v in var_keys], dtype=np.float32)
            new_holds = np.array([float(v.get()) for v in ent_holds], dtype=np.float32)
            new_phase = phase_var.get()
            new_entry = (
                old_entry[0],
                old_entry[1] if len(old_entry) > 1 else None,
                new_keys,
                new_holds,
                new_phase,
                old_entry[5] if len(old_entry) > 5 else -1
            )
            self.data[self.index] = new_entry
            if self.dataset_path:
                np.savez(self.dataset_path, data=np.array(self.data, dtype=object), keys=np.array(self._format_key_list()))
            self.show_current()
            edit_win.destroy()
            set_status(self.status_label, "Edit applied.")
        except Exception as e:
            show_error(self.root, f"Edit application error: {e}", status_label=self.status_label)


    def _open_in_streamlit(self):
        if not self.dataset_path:
            show_info(self.root, "Please load a .npz file first.", status_label=self.status_label)
            return

        current_dir = Path(__file__).resolve().parent
        streamlit_app_path = current_dir / "streamlit_app.py"

        if not streamlit_app_path.is_file():
            show_error(
                self.root,
                f"streamlit_app.py not found: {streamlit_app_path}",
                status_label=self.status_label,
            )
            return

        dataset_arg = urllib.parse.quote(self.dataset_path, safe="")
        port = 8501
        url = f"http://localhost:{port}/?dataset={dataset_arg}"
        log_path = os.path.join(os.getcwd(), "streamlit_launcher.log")
        try:
            open(log_path, "w").close()
        except Exception:
            pass

        # Open directly if already running
        if self._wait_for_port("localhost", port, timeout=1.0):
            webbrowser.open(url)
            set_status(self.status_label, "Streamlit already running, opening.")
            return

        if not shutil.which("streamlit"):
            show_error(self.root, "Streamlit not found. Please install via `pip install streamlit`.", status_label=self.status_label)
            return

        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", str(streamlit_app_path), f"--server.port={port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(current_dir),
            )
        except Exception as e:
            show_error(self.root, f"Failed to launch Streamlit: {e}", status_label=self.status_label)
            return

        def reader(pipe, label):
            with open(log_path, "a", encoding="utf-8") as f:
                for line in pipe:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    entry = f"[{timestamp}] [{label}] {line}"
                    f.write(entry)
                    trimmed = line.strip()
                    set_status(self.status_label, f"Streamlit: {trimmed[:60]}...")

        threading.Thread(target=reader, args=(proc.stdout, "OUT"), daemon=True).start()
        threading.Thread(target=reader, args=(proc.stderr, "ERR"), daemon=True).start()

        set_status(self.status_label, "Starting Streamlit, waiting (30s)...")
        ready = self._wait_for_port("localhost", port, timeout=30.0)
        if not ready:
            show_error(
                self.root,
                f"Streamlit server did not start. Check log: {log_path}",
                status_label=self.status_label,
            )
            return

        webbrowser.open(url)
        set_status(self.status_label, "Streamlit opened.")


    def _wait_for_port(self, host: str, port: int, timeout: float):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.settimeout(0.5)
                    s.connect((host, port))
                    return True
                except Exception:
                    time.sleep(0.2)
        return False
