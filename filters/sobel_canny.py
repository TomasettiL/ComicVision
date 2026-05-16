import cv2
import numpy as np
from .base import BaseFilter, overlay_layers


class SobelCannyFilter(BaseFilter):
    name = "sobel_canny"
    label = "Sobel + Canny"
    description = "Canny and Sobel run independently and are composited as layers — bold outlines from Canny, fine detail from Sobel, all lines solid black"
    tab_tip = "Composites Canny and Sobel as separate layers using darken mode: any edge detected by either filter prints as solid black. Canny gives bold comic outlines; Sobel recovers fine detail. No gray blending — lines are either there or they aren't."
    params = [
        # ── Canny layer ───────────────────────────────────────────────────────
        {
            "name": "canny_blur",
            "label": "Canny Pre-Blur (odd)",
            "type": "range",
            "min": 1,
            "max": 15,
            "step": 2,
            "default": 3,
            "tip": "Smooths the image before Canny detects edges. Higher values suppress noise and fine texture, giving fewer but cleaner bold outlines. Must be an odd number (1 = off, 3 = light, 7+ = heavy).",
        },
        {
            "name": "canny_low",
            "label": "Canny Lower Threshold",
            "type": "range",
            "min": 5,
            "max": 250,
            "step": 5,
            "default": 40,
            "tip": "Edges weaker than this are dropped by Canny. Lower = more outlines included. If the result looks cluttered, raise this. A ratio of ~1:2.5 with the upper threshold works well.",
        },
        {
            "name": "canny_high",
            "label": "Canny Upper Threshold",
            "type": "range",
            "min": 10,
            "max": 500,
            "step": 5,
            "default": 100,
            "tip": "Edges stronger than this are always kept by Canny. Raise to keep only the boldest outlines. The gap between lower and upper controls how many connecting edges appear between strong ones.",
        },
        # ── Sobel layer ───────────────────────────────────────────────────────
        {
            "name": "sobel_kernel",
            "label": "Sobel Kernel Size",
            "type": "select",
            "options": [3, 5, 7],
            "default": 5,
            "tip": "Sampling window for the Sobel detail layer. 3 = finest texture, can be noisy. 5 = balanced default. 7 = broader edges that skip tiny surface detail.",
        },
        {
            "name": "sobel_threshold",
            "label": "Sobel Detail Threshold",
            "type": "range",
            "min": 5,
            "max": 200,
            "step": 5,
            "default": 60,
            "tip": "Controls which Sobel gradients are strong enough to print as a line. Lower = more fine detail included (thinner lines). Higher = only strong edges appear. This is what keeps Sobel lines thinner than Canny's bold outlines.",
        },
    ]

    def process_frame(self, gray: np.ndarray, canny_blur=3, canny_low=40, canny_high=100,
                      sobel_kernel=5, sobel_threshold=60, **kwargs) -> np.ndarray:
        # ── Canny layer: bold comic outlines ──────────────────────────────────
        ck = max(1, int(canny_blur))
        if ck % 2 == 0:
            ck += 1
        blurred = cv2.GaussianBlur(gray, (ck, ck), 0)
        canny_layer = 255 - cv2.Canny(blurred, int(canny_low), int(canny_high))

        # ── Sobel layer: fine detail, binarised at threshold ──────────────────
        sk = int(sobel_kernel)
        if sk % 2 == 0:
            sk += 1
        sobel_x = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=sk)
        sobel_y = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=sk)
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Hard threshold → binary: above threshold becomes an edge, below is background
        _, sobel_binary = cv2.threshold(magnitude, int(sobel_threshold), 255, cv2.THRESH_BINARY)
        sobel_layer = 255 - sobel_binary

        # ── Composite: darken mode (min) ──────────────────────────────────────
        # Any edge from any layer = solid black. No gray, no blending.
        # Add more layers here in the future: overlay_layers(canny_layer, sobel_layer, colour_layer, ...)
        return overlay_layers(canny_layer, sobel_layer)
