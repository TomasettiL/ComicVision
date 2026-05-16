import cv2
import numpy as np
from .base import BaseFilter


class KMeansFilter(BaseFilter):
    name = "kmeans"
    label = "K-Means"
    description = "Reduces the image to exactly N flat colours — bold, poster-like cell shading"
    tab_tip = "Quantises the entire image to a fixed number of colours. More aggressive and graphic than Mean Shift. Lower colour count = bold, flat poster look. Can be slow on large images — lower the colour count for faster processing."
    output_type = "color"
    params = [
        {
            "name": "num_colors",
            "label": "Number of Colours",
            "type": "range",
            "min": 2,
            "max": 24,
            "step": 1,
            "default": 8,
            "tip": "How many distinct colours the output is reduced to. 2–4 = very graphic/stencil look. 6–10 = balanced anime cell-shade style. 12+ starts to look photographic again. Lower values are also faster to compute.",
        },
        {
            "name": "temporal_smoothing",
            "label": "Temporal Smoothing",
            "type": "range",
            "min": 0.0,
            "max": 0.9,
            "step": 0.05,
            "default": 0.0,
            "tip": "Video only — blends each frame with the previous to stop palette assignments from jumping between frames. 0 = off. 0.5–0.7 is usually enough for K-Means. 0.8+ gives very stable colours at the cost of slightly slow transitions during scene changes. Has no effect on images.",
        },
    ]

    def process_frame(self, gray: np.ndarray, bgr: np.ndarray = None, **kwargs) -> np.ndarray:
        num_colors = max(1, int(kwargs.get("num_colors", 8)))
        src = bgr if bgr is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        h, w = src.shape[:2]

        pixels = src.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, num_colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        centers = centers.astype(np.uint8)
        return centers[labels.flatten()].reshape(h, w, 3)
