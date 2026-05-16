import cv2
import numpy as np
from .base import BaseFilter


class SobelFilter(BaseFilter):
    name = "sobel"
    label = "Sobel"
    description = "Gradient magnitude edge detection using Sobel operators — smooth, painterly outlines"
    tab_tip = "Best starting point. Measures how sharply brightness changes across the image and turns those changes into outlines. Tends to produce smooth, painterly lines that follow gradients as well as hard edges."
    params = [
        {
            "name": "kernel_size",
            "label": "Kernel Size",
            "type": "select",
            "options": [3, 5, 7],
            "default": 5,
            "tip": "Size of the sampling window in pixels. 3 = fine detail but can look speckly on flat areas. 5 = balanced default for most images. 7 = broader, smoother edges that ignore tiny texture.",
        },
        {
            "name": "intensity",
            "label": "Edge Intensity",
            "type": "range",
            "min": 0.5,
            "max": 4.0,
            "step": 0.1,
            "default": 1.0,
            "tip": "Scales how dark the edge lines appear. 1.0 = unchanged. Try 1.5–2.5 for a bold comic look. Very high values (3+) can make subtle areas look over-exposed and noisy.",
        },
        {
            "name": "threshold",
            "label": "Edge Threshold",
            "type": "range",
            "min": 0,
            "max": 200,
            "step": 5,
            "default": 0,
            "tip": "0 = off: output is a smooth gradient (best for standalone use). Set above 0 to binarise: gradients stronger than this become solid black edges, everything else becomes white. Use this when stacking Sobel with other layers so all lines are solid black with no gray.",
        },
        {
            "name": "invert",
            "label": "Invert (black lines on white)",
            "type": "checkbox",
            "default": True,
            "tip": "Checked = dark lines on a white background, like a pencil sketch or manga page. Unchecked = bright lines on black, like a neon or X-ray effect.",
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

    def process_frame(self, gray: np.ndarray, bgr: np.ndarray = None,
                      kernel_size=5, intensity=1.0, threshold=0, invert=True, **kwargs) -> np.ndarray:
        ksize = int(kernel_size)
        if ksize % 2 == 0:
            ksize += 1

        sobel_x = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
        sobel_y = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        if float(intensity) != 1.0:
            magnitude = np.clip(magnitude * float(intensity), 0, 255)

        magnitude = magnitude.astype(np.uint8)

        if int(threshold) > 0:
            # Binarise: gradients above threshold → solid edge, below → background
            _, magnitude = cv2.threshold(magnitude, int(threshold), 255, cv2.THRESH_BINARY)

        dilation_size = int(kwargs.get("dilation_size", 0))
        result = (255 - magnitude) if invert else magnitude
        return self.thicken_edges(result, dilation_size)
