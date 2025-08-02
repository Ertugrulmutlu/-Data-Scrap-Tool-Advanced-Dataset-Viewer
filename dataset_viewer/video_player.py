# video_player.py
import threading
import time
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from .constants import PREVIEW_SIZE

class EmbeddedVideoPlayer(ttk.Frame):
    def __init__(self, parent, video_path=None):
        super().__init__(parent)
        self.video_path = video_path
        self.cap = None
        self.fps = 30.0
        self.total_frames = 0
        self.duration = 0
        self.playing = False
        self.current_frame_idx = 0

        self._build_ui()
        if video_path:
            self.load_video(video_path)

    def _build_ui(self):
        self.video_label = ttk.Label(self, text="No video loaded", background="black", width=60)
        self.video_label.grid(row=0, column=0, columnspan=6, sticky="nsew", pady=(0, 4))

        self.play_pause_btn = ttk.Button(self, text="Play", width=6, command=self.toggle_play)
        self.play_pause_btn.grid(row=1, column=0, padx=2)
        self.stop_btn = ttk.Button(self, text="Stop", width=6, command=self.stop)
        self.stop_btn.grid(row=1, column=1, padx=2)
        self.backward_btn = ttk.Button(self, text="<<5s", width=6, command=self.skip_backward)
        self.backward_btn.grid(row=1, column=2, padx=2)
        self.forward_btn = ttk.Button(self, text="5s>>", width=6, command=self.skip_forward)
        self.forward_btn.grid(row=1, column=3, padx=2)
        self.time_label = ttk.Label(self, text="00:00 / 00:00", width=15)
        self.time_label.grid(row=1, column=4, padx=4)
        self.reload_btn = ttk.Button(self, text="Reload", width=6, command=self._reload_video)
        self.reload_btn.grid(row=1, column=5, padx=2)

        self.seek_var = tk.DoubleVar(value=0.0)
        self.seek_scale = ttk.Scale(self, from_=0, to=1, variable=self.seek_var, command=self._on_seek)
        self.seek_scale.grid(row=2, column=0, columnspan=6, sticky="ew", pady=(4, 0))

        self.info_label = ttk.Label(self, text="No video", font=("Segoe UI", 8))
        self.info_label.grid(row=3, column=0, columnspan=6, sticky="w", pady=(2, 0))

        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()

    def load_video(self, video_path):
        if self.cap:
            self.cap.release()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Video Error", f"Cannot open video: {video_path}")
            return
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.seek_scale.config(to=max(0, self.total_frames - 1))
        self.current_frame_idx = 0
        self.playing = False
        self._update_info()
        self.seek_to(0)

    def _reload_video(self):
        if self.video_path:
            self.load_video(self.video_path)

    def toggle_play(self):
        if not self.cap:
            return
        self.playing = not self.playing
        self.play_pause_btn.config(text="Pause" if self.playing else "Play")

    def stop(self):
        self.playing = False
        self.play_pause_btn.config(text="Play")
        self.seek_to(0)

    def skip_backward(self):
        target = max(0, self.current_frame_idx - int(5 * self.fps))
        self.seek_to(target)

    def skip_forward(self):
        target = min(self.total_frames - 1, self.current_frame_idx + int(5 * self.fps))
        self.seek_to(target)

    def _on_seek(self, val):
        try:
            frame_idx = int(float(val))
        except ValueError:
            return
        self.seek_to(frame_idx)

    def seek_to(self, frame_idx):
        if not self.cap:
            return
        self.current_frame_idx = frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.seek_var.set(frame_idx)
        self._render_frame_index(frame_idx)
        self._update_time_label()

    def _format_time(self, seconds):
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m:02d}:{s:02d}"

    def _update_info(self):
        self.info_label.config(text=f"FPS: {self.fps:.2f} | Frames: {self.total_frames}")

    def _update_time_label(self):
        elapsed = self.current_frame_idx / self.fps if self.fps > 0 else 0
        self.time_label.config(text=f"{self._format_time(elapsed)} / {self._format_time(self.duration)}")

    def _render_frame_index(self, idx):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize(PREVIEW_SIZE, Image.NEAREST)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=imgtk, text="")
            self.video_label.image = imgtk
        except Exception as e:
            print("Frame render error:", e)

    def _playback_loop(self):
        while True:
            if self.playing and self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    self.playing = False
                    self.play_pause_btn.config(text="Play")
                    continue
                self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                self.seek_var.set(self.current_frame_idx)
                self._render_frame_index(self.current_frame_idx)
                self._update_time_label()
                time.sleep(1 / self.fps)
            else:
                time.sleep(0.05)
