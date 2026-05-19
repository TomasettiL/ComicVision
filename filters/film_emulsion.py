import cv2
import numpy as np
from .base import BaseFilter

# ── Emulsion spectral sensitivity (R, G, B) ───────────────────────────────────
# Raw sensitivity weights; combined with the optical filter and normalized before use.
EMULSIONS: dict[str, tuple[float, float, float]] = {
    "Collodion (1860s)":      (0.02, 0.08, 0.90),  # near-blind to red/green
    "Orthochromatic (1900s)": (0.05, 0.42, 0.53),  # red-blind, classic silent film
    "Early Pan (1920s)":      (0.18, 0.48, 0.34),  # first balanced sensitivity
    "Panchromatic (1940s)":   (0.27, 0.53, 0.20),  # modern standard
    "Infrared HIE":           (0.75, 0.22, 0.03),  # extreme red/IR — black skies
}

# ── Optical filter transmittance (R, G, B) ────────────────────────────────────
# Simulates a glass filter placed in front of the lens.
OPTICAL_FILTERS: dict[str, tuple[float, float, float]] = {
    "None":       (1.00, 1.00, 1.00),
    "Yellow K2":  (0.95, 0.85, 0.10),  # haze cutting, mild sky drama
    "Orange G":   (0.98, 0.55, 0.03),  # stronger sky/cloud separation
    "Red 25A":    (0.95, 0.10, 0.02),  # extreme sky contrast
    "Red 29":     (0.92, 0.04, 0.01),  # near-black skies
    "Green 58":   (0.05, 0.92, 0.06),  # lightens foliage and skin
    "Blue 47":    (0.08, 0.15, 0.95),  # reverses landscape contrast
}


def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Full piecewise sRGB gamma removal per IEC 61966-2-1.

    Uses the exact 0.04045 threshold — not a simple power law.
    Input:  float32 array in [0, 1]
    Output: float32 array in [0, 1] (linear light)
    """
    low  = img / 12.92
    high = ((img + 0.055) / 1.055) ** 2.4
    return np.where(img <= 0.04045, low, high)


def _linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Full piecewise sRGB gamma encoding per IEC 61966-2-1.

    Uses the exact 0.0031308 threshold — not a simple power law.
    Input:  float32 array in [0, 1] (linear light)
    Output: float32 array in [0, 1] (gamma-encoded)
    """
    img  = np.clip(img, 0.0, 1.0)
    low  = img * 12.92
    high = 1.055 * (img ** (1.0 / 2.4)) - 0.055
    return np.where(img <= 0.0031308, low, high)


class FilmEmulsionFilter(BaseFilter):
    name        = "film_emulsion"
    label       = "Film Emulsion"
    description = (
        "Converts colour to grayscale using historically accurate "
        "photographic emulsion and optical filter simulation"
    )
    tab_tip = (
        "Simulates how orthochromatic and panchromatic film stock perceives colour "
        "differently from the human eye. Early emulsions were nearly blind to red — "
        "hence why Victorian portraits show pale lips and dark skies. "
        "Pair with an optical filter to further shape tonal contrast. "
        "Stack edge layers on top for a period photograph or graphic novel look."
    )
    output_type = "color"

    params = [
        {
            "name": "emulsion",
            "label": "Film Emulsion",
            "type": "select",
            "options": list(EMULSIONS.keys()),
            "default": "Panchromatic (1940s)",
            "tip": (
                "Collodion (1860s): nearly blind to red/green — dark skies, pale lips. "
                "Orthochromatic (1900s): red-blind, classic silent-film look. "
                "Early Pan (1920s): first balanced stock. "
                "Panchromatic (1940s): modern standard, closest to human vision. "
                "Infrared HIE: extreme red/IR bias — near-black skies, glowing foliage."
            ),
        },
        {
            "name": "optical_filter",
            "label": "Optical Filter",
            "type": "select",
            "options": list(OPTICAL_FILTERS.keys()),
            "default": "None",
            "tip": (
                "Glass filter placed in front of the lens — blocks certain wavelengths "
                "to shift tonal relationships before the film sees the scene. "
                "Yellow K2: mild sky drama, cuts haze. "
                "Orange G: stronger cloud separation, good for architecture. "
                "Red 25A / 29: extreme sky contrast, near-black skies. "
                "Green 58: lightens foliage and faces, darkens sky. "
                "Blue 47: pale sky, dark foliage — reverses landscape contrast. "
                "Pairing Red 25A with Orthochromatic emulsion recreates "
                "the high-drama look of 1930s landscape photography."
            ),
        },
    ]

    def process_frame(self, gray: np.ndarray, bgr: np.ndarray = None, **kwargs) -> np.ndarray:
        emulsion_key    = kwargs.get("emulsion",       "Panchromatic (1940s)")
        optical_key     = kwargs.get("optical_filter", "None")

        # Validate — fall back to defaults if an unrecognised key arrives
        if emulsion_key not in EMULSIONS:
            emulsion_key = "Panchromatic (1940s)"
        if optical_key not in OPTICAL_FILTERS:
            optical_key = "None"

        src = bgr if bgr is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # ── 1. Linearize ──────────────────────────────────────────────────────
        # OpenCV stores channels as BGR; flip to RGB for channel-order clarity.
        # Divide by 255 and apply full piecewise sRGB inverse gamma.
        rgb_linear = _srgb_to_linear(src[:, :, ::-1].astype(np.float32) / 255.0)

        # ── 2. Combined weight: optical_filter × emulsion, normalized ─────────
        # Applying the optical filter as a transmittance multiplier then
        # dot-producting with emulsion sensitivity is algebraically equivalent
        # to computing one combined weight vector and doing a single dot product.
        f = np.array(OPTICAL_FILTERS[optical_key], dtype=np.float32)
        e = np.array(EMULSIONS[emulsion_key],      dtype=np.float32)
        w = f * e
        w_sum = w.sum()
        w = w / w_sum if w_sum > 0.0 else np.full(3, 1.0 / 3.0, dtype=np.float32)

        # ── 3. Weighted dot product → linear luminance ────────────────────────
        luminance = (rgb_linear * w).sum(axis=2)  # HxW, float32

        # ── 4. Re-encode to sRGB ──────────────────────────────────────────────
        encoded  = _linear_to_srgb(luminance)
        out_gray = (encoded * 255.0 + 0.5).clip(0.0, 255.0).astype(np.uint8)

        # Return 3-channel BGR — the compositor expects color layers to be BGR
        return cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)
