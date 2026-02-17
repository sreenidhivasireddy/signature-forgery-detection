import os
import cv2
import json
import numpy as np
from typing import Tuple, List

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def compute_mean_hw(root_dir: str, default_h: int = 273, default_w: int = 675) -> Tuple[int, int]:
    """
    Compute mean height/width across all images in root_dir/*/*.
    If no images found, returns defaults.
    """
    heights: List[int] = []
    widths: List[int] = []

    if not os.path.isdir(root_dir):
        return default_h, default_w

    for person in sorted(os.listdir(root_dir)):
        person_dir = os.path.join(root_dir, person)
        if not os.path.isdir(person_dir):
            continue

        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            fpath = os.path.join(person_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape[:2]
            heights.append(h)
            widths.append(w)

    if not heights or not widths:
        return default_h, default_w

    return int(np.mean(heights)), int(np.mean(widths))

def load_signature_data(root_dir: str, target_h: int, target_w: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images from root_dir/*/*, resize to (target_h, target_w) and label:
    - forged (1) if 'forg' in filename
    - genuine (0) otherwise

    Returns:
      X: float32 array in [0,1], shape (N, H, W, 1)
      y: int32 array, shape (N,)
    """
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    for person in sorted(os.listdir(root_dir)):
        person_dir = os.path.join(root_dir, person)
        if not os.path.isdir(person_dir):
            continue

        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            fpath = os.path.join(person_dir, fname)

            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (target_w, target_h))  # (width, height)
            X_list.append(img)

            label = 1 if "forg" in fname.lower() else 0
            y_list.append(label)

    if not X_list:
        raise RuntimeError(f"No images found in: {root_dir}")

    X = np.array(X_list, dtype=np.float32) / 255.0
    X = X[..., np.newaxis]  # (N,H,W,1)
    y = np.array(y_list, dtype=np.int32)
    return X, y

def save_metadata(path: str, h: int, w: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"height": int(h), "width": int(w)}, f, indent=2)

def load_metadata(path: str) -> Tuple[int, int]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return int(obj["height"]), int(obj["width"])
