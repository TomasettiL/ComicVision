import cv2
import numpy as np
from .base import BaseFilter


class MeanShiftFilter(BaseFilter):
    name = "mean_shift"
    label = "Mean Shift"
    description = "Flattens regions of similar colour into solid blocks — the core of the anime/comic flat-colour look"
    tab_tip = "Groups pixels that are close together in both position and colour into flat regions, replacing each region with a single representative colour. The main tool for anime-style flat shading. Stack with Canny/Sobel edge layers on top for the full comic effect."
    output_type = "color"
    params = [
        {
            "name": "spatial_radius",
            "label": "Spatial Radius",
            "type": "range",
            "min": 5,
            "max": 80,
            "step": 5,
            "default": 20,
            "tip": "How far apart (in pixels) two pixels can be and still be grouped into the same flat region. Higher = larger, broader colour blocks. Lower = smaller regions, more detail preserved. Try 20–40 for a clear anime look.",
        },
        {
            "name": "color_radius",
            "label": "Colour Radius",
            "type": "range",
            "min": 5,
            "max": 80,
            "step": 5,
            "default": 30,
            "tip": "How different two colours can be and still be merged into the same region. Higher = more colours get flattened together (fewer, bolder blocks). Lower = colours stay more distinct. Try 20–40 alongside spatial radius.",
        },
        {
            "name": "temporal_smoothing",
            "label": "Temporal Smoothing",
            "type": "range",
            "min": 0.0,
            "max": 0.9,
            "step": 0.05,
            "default": 0.0,
            "tip": "Video only — blends each frame with the previous one to prevent colour region boundaries from flickering. 0 = off. 0.5 = subtle stability. 0.7–0.8 = strong anti-flicker. Higher values stop region edges from 'breathing' between frames but can cause slight colour lag during fast cuts. Has no effect on images.",
        },
    ]

    def process_frame(self, gray: np.ndarray, bgr: np.ndarray = None, **kwargs) -> np.ndarray:
        spatial_radius = int(kwargs.get("spatial_radius", 20))
        color_radius   = int(kwargs.get("color_radius", 30))
        src = bgr if bgr is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return cv2.pyrMeanShiftFiltering(src, spatial_radius, color_radius)
