import cv2
import numpy as np
from .base import BaseFilter


class BilateralFilter(BaseFilter):
    name = "bilateral"
    label = "Bilateral Blur"
    description = "Smooths flat regions while keeping edges sharp — like a smart blur that knows where the lines are"
    tab_tip = "Unlike a normal blur, bilateral filtering respects edges: it averages nearby pixels only if they are a similar colour. Great for smoothing skin, sky, and flat areas without softening outlines. Stack before edge layers for cleaner line detection, or after K-Means/Mean Shift to polish flat colour regions."
    output_type = "color"
    params = [
        {
            "name": "d",
            "label": "Diameter",
            "type": "range",
            "min": 3,
            "max": 15,
            "step": 2,
            "default": 9,
            "tip": "Size of the pixel neighbourhood considered for each blur step. Larger = stronger smoothing but noticeably slower — values above 9 can be slow on large images or video. Keep at 5–9 for real-time use.",
        },
        {
            "name": "sigma_color",
            "label": "Colour Range",
            "type": "range",
            "min": 10,
            "max": 200,
            "step": 10,
            "default": 75,
            "tip": "How different two colours can be and still be blended together. Low = only nearly-identical colours mix (tight edge preservation). High = a wider range of colours get averaged (more smoothing, softer edges). Try 50–100 for a cartoon look.",
        },
        {
            "name": "sigma_space",
            "label": "Spatial Range",
            "type": "range",
            "min": 10,
            "max": 200,
            "step": 10,
            "default": 75,
            "tip": "How far away a pixel can be and still influence the blur. Higher = broader smoothing across the image. Works together with Colour Range — raising both gives a stronger painterly effect.",
        },
    ]

    def process_frame(self, gray: np.ndarray, bgr: np.ndarray = None, **kwargs) -> np.ndarray:
        d           = int(kwargs.get("d", 9))
        sigma_color = float(kwargs.get("sigma_color", 75))
        sigma_space = float(kwargs.get("sigma_space", 75))
        src = bgr if bgr is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return cv2.bilateralFilter(src, d, sigma_color, sigma_space)