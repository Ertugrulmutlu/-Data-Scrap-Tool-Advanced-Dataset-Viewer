# data_collector.py (updated: captures snapshot on every key state change)
import time
import numpy as np
import threading
import os
from pynput import keyboard

KEY_LIST = ['a', 's', 'd', 'f']

pressed_times = {}  # key -> press timestamp
data = []
recording = False
selected_region_global = None
_listener = None
lock = threading.Lock()
last_release_time = None

# global video frame; updated by RecorderGUI on each preview
current_video_frame = 0

SAVE_PATH = "collected/dataset"

def keys_to_multihot(active_keys):
    arr = np.zeros(len(KEY_LIST), dtype=np.float32)
    for k in active_keys:
        if k in KEY_LIST:
            arr[KEY_LIST.index(k)] = 1.0
    return arr

def start_keyboard_listener():
    global _listener
    if _listener:
        return _listener
    _listener = keyboard.Listener(on_press=_on_press_global, on_release=_on_release_global)
    _listener.start()
    return _listener

def stop_keyboard_listener():
    global _listener
    if _listener:
        _listener.stop()
        _listener = None

def _on_press_global(k):
    global last_release_time, recording, pressed_times, data, current_video_frame
    if not recording or selected_region_global is None:
        return
    try:
        key_char = k.char
    except AttributeError:
        key_char = str(k)
    if key_char not in KEY_LIST:
        return

    now = time.time()
    prev_interval = 0.0 if last_release_time is None else now - last_release_time

    newly_pressed = key_char not in pressed_times
    if newly_pressed:
        pressed_times[key_char] = now  # add key if not already pressed

    # on every press change (new key added or held key triggered) capture snapshot
    from Data_scrap_tool_function.capture import grab_region
    image = grab_region(selected_region_global)

    active_keys = list(pressed_times.keys())
    multi_hot = keys_to_multihot(active_keys)

    hold_durations = np.zeros(len(KEY_LIST), dtype=np.float32)
    for i, kname in enumerate(KEY_LIST):
        if kname in pressed_times:
            hold_durations[i] = now - pressed_times[kname]

    entry = (
        image,
        prev_interval,
        multi_hot,
        hold_durations,
        "press",
        current_video_frame
    )

    with lock:
        data.append(entry)

def _on_release_global(k):
    global last_release_time, recording, pressed_times, data, current_video_frame
    try:
        key_char = k.char
    except AttributeError:
        key_char = str(k)
    if (hasattr(k, "name") and k == keyboard.Key.esc) or key_char == "Key.esc":
        recording = False
        return
    if key_char not in KEY_LIST:
        return
    now = time.time()
    if key_char in pressed_times:
        hold = now - pressed_times[key_char]
        last_release_time = now

        from Data_scrap_tool_function.capture import grab_region
        image = grab_region(selected_region_global)

        # active keys after release
        active_keys = [kk for kk in pressed_times.keys() if kk != key_char]
        multi_hot = keys_to_multihot(active_keys)

        # hold durations: precise for released key, updated for others
        hold_durations = np.zeros(len(KEY_LIST), dtype=np.float32)
        for i, kname in enumerate(KEY_LIST):
            if kname == key_char:
                hold_durations[i] = hold
            elif kname in pressed_times and kname != key_char:
                hold_durations[i] = now - pressed_times[kname]

        entry = (
            image,
            0.0,  # release interval, can be adjusted if needed
            multi_hot,
            hold_durations,
            "release",
            current_video_frame
        )

        with lock:
            data.append(entry)

        # remove released key
        del pressed_times[key_char]

def save_dataset_npz(path=SAVE_PATH):
    """
    Saves all collected snapshots.
    """
    with lock:
        if not data:
            return path, 0
        new_arr = np.array(data, dtype=object)
        np.savez(path, data=new_arr, keys=np.array(KEY_LIST))
        return path, len(data)