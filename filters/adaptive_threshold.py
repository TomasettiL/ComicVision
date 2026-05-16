import cv2
import numpy as np
from .base import BaseFilter


class AdaptiveThresholdFilter(BaseFilter):
    name = "adaptive_threshold"
    label = "Adaptive Threshold"
    description = "Thresholds each pixel against its local neighbourhood — picks up ink-like texture and fine hatching detail across bright and dark areas equally"
    tab_tip = "Unlike a global threshold, this compares each pixel to the average of its immediate surroundings, so it works across both bright and dark regions of the same image. Produces clean black-on-white line art. Best stacked under a colour layer for a crosshatch or ink-sketch look."
    output_type = "edge"
    params = [
        {
            "name": "block_size",
            "label": "Block Size",
            "type": "range",
            "min": 3,
            "max": 51,
            "step": 2,
            "default": 11,
            "tip": "Size of the pixel neighbourhood used to calculate the local threshold. Larger = broader regions compared, picks up bigger features. Smaller = very local comparison, picks up fine texture and noise. Steps by 2 to stay odd (required by OpenCV).",
        },
        {
            "name": "C",
            "label": "Sensitivity",
            "type": "range",
            "min": 1,
            "max": 30,
            "step": 1,
            "default": 7,
            "tip": "A constant subtracted from the local mean before thresholding. Higher = fewer pixels marked as edges (only the darkest features show). Lower = more pixels marked, picks up faint texture. High = bold sparse lines, low = dense noisy lines.",
        },
        {
            "name": "method",
            "label": "Method",
            "type": "select",
            "options": [0, 1],
            "default": 1,
            "tip": "0 = Mean: compares each pixel to the flat average of its block. 1 = Gaussian: weights the neighbourhood by a bell curve centred on the pixel, giving smoother less blocky results. Gaussian is usually better for natural images.",
        },
        {
            "name": "blur_kernel",
            "label": "Pre-blur",
            "type": "range",
            "min": 0,
            "max": 9,
            "step": 2,
            "default": 0,
            "tip": "Gaussian blur applied before thresholding to reduce noise. 0 = off (maximum detail). Higher values smooth the input so only larger features are detected. Useful if the result looks too grainy or speckled.",
        },
        {
            "name": "dilation_size",
            "label": "Line Thickness",
            "type": "range",
            "min": 0,
            "max": 10,
            "step": 1,
            "default": 0,
            "tip": "Thickens detected edge lines after detection. 0 = off. Each step expands lines by roughly one pixel in all directions. 2–4 gives a bold comic ink look. Applied last, so it works the same regardless of the other parameter settings.",
        },
    ]

    def process_frame(self, gray: np.ndarray, bgr: np.ndarray = None, **kwargs) -> np.ndarray:
        block_size = int(kwargs.get("block_size", 11))
        C          = int(kwargs.get("C", 7))
        method     = int(kwargs.get("method", 1))
        blur_kernel = int(kwargs.get("blur_kernel", 0))

        # block_size must be odd and at least 3
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, block_size)

        src = gray.copy()
        if blur_kernel > 1:
            k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            src = cv2.GaussianBlur(src, (k, k), 0)

        adaptive_method = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == 1
                           else cv2.ADAPTIVE_THRESH_MEAN_C)

        dilation_size = int(kwargs.get("dilation_size", 0))
        result = cv2.adaptiveThreshold(
            src, 255,
            adaptive_method,
            cv2.THRESH_BINARY,
            block_size,
            C,
        )
        return self.thicken_edges(result, dilation_size)