import cv2
import numpy as np
from .base import BaseFilter


class MedianBlurFilter(BaseFilter):
    name = "median_blur"
    label = "Median Blur"
    description = "Removes fine texture and noise by replacing each pixel with the median of its neighbourhood — edges stay surprisingly sharp"
    tab_tip = "Unlike Gaussian blur, median blur can't produce colours that weren't already in the image, so it flattens texture without creating muddy transitions. Stack before K-Means or edge layers to clean up noise first, or use alone for a smooth painted look."
    output_type = "color"
    params = [
        {
            "name": "kernel_size",
            "label": "Kernel Size",
            "type": "range",
            "min": 3,
            "max": 21,
            "step": 2,
            "default": 5,
            "tip": "Side length of the neighbourhood sampled for each pixel. Must be odd — the slider steps by 2 to enforce this. 3–5 = subtle noise removal. 7–11 = visible smoothing, good before K-Means. 15+ = heavy painterly effect.",
        },
    ]

    def process_frame(self, gray: np.ndarray, bgr: np.ndarray = None, **kwargs) -> np.ndarray:
        k = int(kwargs.get("kernel_size", 5))
        if k % 2 == 0:
            k += 1
        k = max(3, k)
        src = bgr if bgr is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return cv2.medianBlur(src, k)