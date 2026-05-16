from .sobel import SobelFilter
from .canny import CannyFilter
from .mean_shift import MeanShiftFilter
from .kmeans import KMeansFilter
from .sat_boost import SaturationBoostFilter
from .adaptive_threshold import AdaptiveThresholdFilter
from .bilateral import BilateralFilter
from .median_blur import MedianBlurFilter
from .posterize import PosterizeFilter
from .brightness import BrightnessFilter

# Registry — add new filters here and they appear in the layer UI automatically.
# output_type = "edge"  → grayscale, composited with darken mode
# output_type = "color" → BGR, chained sequentially as the colour base
FILTERS: dict = {
    f.name: f
    for f in [
        SobelFilter(),
        CannyFilter(),
        MeanShiftFilter(),
        KMeansFilter(),
        SaturationBoostFilter(),
        AdaptiveThresholdFilter(),
        BilateralFilter(),
        MedianBlurFilter(),
        PosterizeFilter(),
        BrightnessFilter(),
    ]
}
