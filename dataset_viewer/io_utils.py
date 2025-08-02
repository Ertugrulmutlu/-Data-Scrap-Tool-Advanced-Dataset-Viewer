# io_utils.py
import pathlib
import os
import glob

def find_linked_video(dataset_path, recordings_dir=None):
    """
    Given a .npz file, return the first .mp4 file in recordings_dir whose name
    matches exactly up to the first underscore.
    Example: x.npz --> matches x_*.mp4 but NOT x1_*.mp4
    """
    if not dataset_path:
        return None
    base = pathlib.Path(dataset_path).stem
    if recordings_dir is None:
        from .constants import RECORDINGS_DIR
        recordings_dir = RECORDINGS_DIR

    if not os.path.isdir(recordings_dir):
        return None

    # All mp4 files with an underscore after base
    possible = glob.glob(os.path.join(recordings_dir, f"{base}_*.mp4"))
    for path in possible:
        name = os.path.basename(path)
        left = name.split("_")[0]
        if left == base:
            return path
    return None
