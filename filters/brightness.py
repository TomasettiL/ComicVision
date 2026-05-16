import cv2
import numpy as np
from .base import BaseFilter


class BrightnessFilter(BaseFilter):
    name = "brightness"
    label = "Brightness & Contrast"
    description = "Adjust luminance, contrast and gamma — without shifting colours"
    tab_tip = "All three controls operate on brightness only, leaving hue and saturation untouched. Gamma is the most useful for comic/anime work: values below 1.0 lift shadows and midtones while leaving highlights alone, which helps coloured subjects stand out from bright or washed-out backgrounds."
    output_type = "color"
    params = [
        {
            "name": "gamma",
            "label": "Gamma",
            "type": "range",
            "min": 0.3,
            "max": 2.5,
            "step": 0.05,
            "default": 1.0,
            "tip": "Non-linear brightness curve. Below 1.0 lifts shadows and midtones while barely touching highlights — good for making subjects pop off bright or washed-out backgrounds. Above 1.0 deepens shadows for a moodier look. 1.0 = no change.",
        },
        {
            "name": "brightness",
            "label": "Brightness",
            "type": "range",
            "min": -100,
            "max": 100,
            "step": 5,
            "default": 0,
            "tip": "Flat additive lift applied to every pixel equally. Positive = brighter, negative = darker. Use small values (±20) for fine-tuning after Gamma. Large positive values wash out the image; large negative values crush shadows to black.",
        },
        {
            "name": "contrast",
            "label": "Contrast",
            "type": "range",
            "min": 0.5,
            "max": 2.5,
            "step": 0.05,
            "default": 1.0,
            "tip": "Stretches or compresses tones around the midpoint. Above 1.0 pushes lights lighter and darks darker — makes subjects pop from backgrounds. Below 1.0 flattens everything toward grey. 1.2–1.5 is a good starting range for adding punch without looking harsh.",
        },
    ]

    def process_frame(self, gray: np.ndarray, bgr: np.ndarray = None, **kwargs) -> np.ndarray:
        gamma      = max(0.01, float(kwargs.get("gamma", 1.0)))
        brightness = float(kwargs.get("brightness", 0.0))
        contrast   = max(0.01, float(kwargs.get("contrast", 1.0)))

        src = bgr if bgr is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV).astype(np.float32)
        V = hsv[:, :, 2]

        # 1. Gamma — non-linear shadow lift
        V = np.power(V / 255.0, gamma) * 255.0

        # 2. Brightness — flat additive shift
        V = V + brightness

        # 3. Contrast — stretch around midpoint
        V = (V - 128.0) * contrast + 128.0

        hsv[:, :, 2] = np.clip(V, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)