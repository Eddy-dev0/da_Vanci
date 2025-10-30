# utils/metrics.py
import numpy as np
from skimage.color import rgb2lab

def deltaE_mean_p95(rgb_ref01: np.ndarray, rgb_pred01: np.ndarray):
    """Quick ΔE (L2 in LAB). Rückgabe: (mean, 95th percentile)"""
    lab1 = rgb2lab(rgb_ref01)
    lab2 = rgb2lab(rgb_pred01)
    de = np.linalg.norm(lab1 - lab2, axis=-1)
    return float(de.mean()), float(np.percentile(de, 95))
