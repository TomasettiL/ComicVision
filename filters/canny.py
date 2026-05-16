import cv2
import numpy as np
from .base import BaseFilter


class CannyFilter(BaseFilter):
    name = "canny"
    label = "Canny"
    description = "Canny edge detector with Gaussian pre-blur — crisp, thin outlines"
    tab_tip = "Uses two sensitivity thresholds to find only the most confident edges, producing thin, precise outlines. Ideal for clean line art. Works best after smoothing away noise with the blur controls."
    params = [
        {
            "name": "blur_kernel",
            "label": "Blur Kernel (odd)",
            "type": "range",
            "min": 1,
            "max": 15,
            "step": 2,
            "default": 3,
            "tip": "Smooths the image before edge detection. Higher values blur away fine texture and noise, resulting in fewer but cleaner lines. Must be an odd number (1 = no blur, 3 = light, 7+ = heavy).",
        },
        {
            "name": "sigma",
            "label": "Blur Sigma",
            "type": "range",
            "min": 0.1,
            "max": 5.0,
            "step": 0.1,
            "default": 1.0,
            "tip": "Controls how far the blur spreads. Raise this together with Blur Kernel for even stronger noise suppression. At low values (0.5–1.0) the effect is subtle; at 3+ it becomes very aggressive.",
        },
        {
            "name": "threshold_low",
            "label": "Lower Threshold",
            "type": "range",
            "min": 5,
            "max": 250,
            "step": 5,
            "default": 40,
            "tip": "Edges weaker than this are discarded entirely. Lower values include more faint edges and fine detail. Try 20–60 for most images. If you see too much noise, raise it.",
        },
        {
            "name": "threshold_high",
            "label": "Upper Threshold",
            "type": "range",
            "min": 10,
            "max": 500,
            "step": 5,
            "default": 100,
            "tip": "Edges stronger than this are always kept. The gap between lower and upper controls how many 'connecting' edges appear between strong ones. A ratio of 1:2.5 (e.g. 40 / 100) is a good default. Raise both to keep only bold outlines.",
        },
        {
            "name": "bold_power",
            "label": "Edge Boldness",
            "type": "range",
            "min": 1.0,
            "max": 3.0,
            "step": 0.1,
            "default": 2.0,
            "tip": "Thickens lines by boosting the contrast of the edge image. 1.0 = no change (thin lines). 2.0 = good comic weight. Higher values darken any remaining gray and make lines appear heavier — but going too high can make the output look over-inked.",
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
                      blur_kernel=3, sigma=1.0, threshold_low=40, threshold_high=100, bold_power=2.0, **kwargs) -> np.ndarray:
        k = int(blur_kernel)
        if k % 2 == 0:
            k += 1
        k = max(1, k)

        blurred = cv2.GaussianBlur(gray, (k, k), float(sigma))
        edges = cv2.Canny(blurred, int(threshold_low), int(threshold_high))
        result = 255 - edges  # invert: black lines on white

        if float(bold_power) != 1.0:
            # Gamma-compress: raises white (non-edge) areas toward white, darkens grays
            f = result.astype(np.float32) / 255.0
            f = np.power(f, float(bold_power)) * 255.0
            result = np.clip(f, 0, 255).astype(np.uint8)

        dilation_size = int(kwargs.get("dilation_size", 0))
        return self.thicken_edges(result, dilation_size)
