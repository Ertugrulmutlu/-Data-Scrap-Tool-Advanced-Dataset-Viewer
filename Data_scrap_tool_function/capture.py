# capture.py
import mss
import cv2
import numpy as np

# Configuration constants
MODEL_INPUT = (227, 227)  # size for model input
PREVIEW_SCALE = 2         # how much to scale up for GUI preview
PREVIEW_SIZE = (MODEL_INPUT[0] * PREVIEW_SCALE, MODEL_INPUT[1] * PREVIEW_SCALE)  # exported for GUI

def list_monitors():
    """
    Enumerate connected monitors.
    Returns list of tuples: (display name, monitor dict)
    """
    with mss.mss() as sct:
        mons = sct.monitors  # index 0 is all, 1..n are individual
        result = []
        for i, m in enumerate(mons):
            name = f"Monitor {i} ({m['width']}x{m['height']})"
            result.append((name, m))
        return result

def grab_region(region):
    """
    Capture a given screen region and return normalized RGB image (float32 [0,1]).
    region: (left, top, width, height) in global coordinates.
    """
    left, top, width, height = region
    with mss.mss() as sct:
        frame = np.array(sct.grab({"left": left, "top": top, "width": width, "height": height}))
    resized = cv2.resize(frame, MODEL_INPUT)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype("float32") / 255.0
    return normalized

def get_preview_image(raw_bgr_frame, active_keys, key_list, scale=PREVIEW_SCALE):
    """
    Given a BGR frame and list of active keys, overlay indicators and scale up.
    Returns an RGB uint8 image suitable for GUI.
    """
    preview_size = (MODEL_INPUT[0] * scale, MODEL_INPUT[1] * scale)
    # Resize to base
    small = cv2.resize(raw_bgr_frame, MODEL_INPUT)
    # Overlay which keys are active
    for idx, key_name in enumerate(key_list):
        if key_name in active_keys:
            cv2.putText(
                small,
                f"{key_name}↓",
                (5 + idx * 50, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
    # Scale up nearest neighbor to keep sharp blocks
    preview = cv2.resize(small, preview_size, interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    return rgb