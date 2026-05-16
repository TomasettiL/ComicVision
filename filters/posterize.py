import cv2
import numpy as np
from .base import BaseFilter


class PosterizeFilter(BaseFilter):
    name = "posterize"
    label = "Posterize"
    description = "Reduces each colour channel to a fixed number of flat bands — instant screen-print or pop-art effect"
    tab_tip = "Divides the brightness range of each colour channel into equal steps and snaps every pixel to the nearest one. Faster and more graphic than K-Means — the palette is determined by the step size, not the image content. 2–4 levels gives a bold stencil look. Stack after Bilateral or Median Blur to get cleaner bands with less noise."
    output_type = "color"
    params = [
        {
            "name": "levels",
            "label": "Colour Levels",
            "type": "range",
            "min": 2,
            "max": 16,
            "step": 1,
            "default": 6,
            "tip": "Number of distinct brightness steps per colour channel. 2 = each channel is either fully on or off (very graphic, up to 8 possible colours). 4–6 = bold comic/anime palette. 8+ = subtle, starts to look photographic again. This applies per-channel, so 4 levels gives up to 4×4×4 = 64 possible colours.",
        },
    ]

    def process_frame(self, gray: np.ndarray, bgr: np.ndarray = None, **kwargs) -> np.ndarray:
        levels = max(2, int(kwargs.get("levels", 6)))
        src = bgr if bgr is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        step = 256.0 / levels

        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.floor(hsv[:, :, 1] / step) * step  # quantize S
        hsv[:, :, 2] = np.floor(hsv[:, :, 2] / step) * step  # quantize V
        # H is left untouched — no hue shifts
        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)