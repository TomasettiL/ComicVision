import cv2
import numpy as np
from .base import BaseFilter


class SaturationBoostFilter(BaseFilter):
    name = "sat_boost"
    label = "Saturation Boost"
    description = "Lifts muted / near-grey colours to a more vivid cartoon palette without touching already-saturated colours"
    tab_tip = "Boosts saturation only on colours that fall below a threshold — leaving vivid colours alone. The boost tapers smoothly to zero at the threshold so there's no hard jump. Stack after Mean Shift or K-Means to punch up flat regions to a more vivid anime palette."
    output_type = "color"
    params = [
        {
            "name": "threshold",
            "label": "Muted Threshold",
            "type": "range",
            "min": 0,
            "max": 255,
            "step": 5,
            "default": 80,
            "tip": "Saturation value (0–255) below which a colour is considered muted and eligible for boosting. Colours at or above this value are left completely unchanged. 80 is a good starting point — lower it to only boost the most washed-out colours, raise it to boost more of the image.",
        },
        {
            "name": "boost_amount",
            "label": "Boost Amount",
            "type": "range",
            "min": 0,
            "max": 200,
            "step": 5,
            "default": 80,
            "tip": "The maximum saturation increase applied to the most muted colours (near-grey, S≈0). The boost tapers proportionally to zero as saturation approaches the threshold, so the transition is seamless. Higher values push muted regions toward vivid cartoon colours more aggressively.",
        },
    ]

    def process_frame(self, gray: np.ndarray, bgr: np.ndarray = None,
                      threshold: float = 80, boost_amount: float = 80, **kwargs) -> np.ndarray:
        src = bgr if bgr is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        t = float(threshold)
        b = float(boost_amount)

        if t <= 0 or b <= 0:
            return src

        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV).astype(np.float32)
        S = hsv[:, :, 1]

        # Tapered boost: strongest at S=0, fades to zero at S=threshold.
        # new_S = S + b × (1 − S/t)   →   continuous, no hard boundary.
        muted = S < t
        S[muted] = np.clip(S[muted] + b * (1.0 - S[muted] / t), 0, 255)
        hsv[:, :, 1] = S

        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
