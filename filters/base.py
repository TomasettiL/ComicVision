import functools
import cv2
import numpy as np
from abc import ABC, abstractmethod


def overlay_layers(*layers: np.ndarray) -> np.ndarray:
    """Composite grayscale black-on-white edge layers using darken mode.

    Each layer: 0 = edge (ink), 255 = background.
    The minimum (darkest) pixel across all layers is kept, so any edge from
    any layer prints as solid black — no gray mixing, no loss of line weight.

    Usage: result = overlay_layers(canny_layer, sobel_layer, ...)
    """
    return functools.reduce(cv2.min, layers)


class BaseFilter(ABC):
    name: str = ""
    label: str = ""
    description: str = ""
    tab_tip: str = ""

    # "edge"  → process_frame returns uint8 grayscale (0 = ink, 255 = bg)
    # "color" → process_frame returns uint8 BGR image
    output_type: str = "edge"

    params: list = []   # each entry may include an optional "tip" key

    @abstractmethod
    def process_frame(self, gray: np.ndarray, bgr: np.ndarray = None, **kwargs) -> np.ndarray:
        """Process one frame.

        Args:
            gray: uint8 grayscale frame — always provided.
            bgr:  uint8 BGR frame — provided for colour filters and when chaining
                  colour layers (bgr will be the output of the previous colour filter).
                  Edge filters can ignore this.
        Returns:
            Edge filters  → grayscale uint8 (0 = ink, 255 = background).
            Colour filters → BGR uint8 (stylised colour image).
        """
        pass

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "tab_tip": self.tab_tip,
            "output_type": self.output_type,
            "params": self.params,
        }
    
    @staticmethod
    def thicken_edges(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        if kernel_size <= 0:
            return mask
        k = kernel_size * 2 + 1 # kernel size must be odd
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.erode(mask, kernel)
