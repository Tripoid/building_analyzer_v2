"""
Facade Restoration — multi-pass inpainting pipeline.

Pass 1 — Surface defects (LaMa or OpenCV TELEA):
  crack, water_damage, efflorescence, moss, rust

Pass 2 — Structural / large defects (color-sampled fill + TELEA edges):
  exposed_brick, spalling, peeling — filled with plaster color sampled from
  a ring of intact facade pixels around each defect, then edge-blended.

Pass 3 — Broken glass (synthetic glass fill):
  broken_glass — color sampled from intact window pixels; if none available,
  uses a neutral gray-blue with subtle noise and a sky-reflection gradient.
"""

import cv2
import gc
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Defects handled by LaMa / TELEA inpainting (surface-level, localised damage)
SURFACE_DEFECTS = {"crack", "water_damage", "efflorescence", "moss", "rust"}

# Defects restored via color-sampling + edge-blend (structural / potentially large)
STRUCTURAL_DEFECTS = {"exposed_brick", "spalling", "peeling"}

_lama: object = None   # lazy singleton


def _load_lama():
    global _lama
    if _lama is not None:
        return _lama
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError as e:
        raise RuntimeError(
            "simple-lama-inpainting not installed. Run: pip install simple-lama-inpainting"
        ) from e
    logger.info("Loading LaMa inpainting model (first run downloads ~200 MB)…")
    _lama = SimpleLama()
    logger.info("LaMa model ready.")
    return _lama


# ─────────────────────────────────────────────────────────────────────────────
# Pass-1 helpers (surface inpainting)
# ─────────────────────────────────────────────────────────────────────────────

def _inpaint_surface(img_rgb: np.ndarray, combined_mask: np.ndarray) -> np.ndarray:
    """LaMa inpainting with TELEA fallback, returns RGB result."""
    h, w = img_rgb.shape[:2]
    total_px = h * w

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(combined_mask * 255, kernel, iterations=2)
    mask_ratio = float(np.count_nonzero(dilated)) / total_px

    if not np.any(dilated):
        return img_rgb.copy()

    # Small or very large → TELEA
    if mask_ratio < 0.02 or mask_ratio > 0.40:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        restored_bgr = cv2.inpaint(bgr, dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        return cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)

    try:
        lama = _load_lama()
        pil_image = Image.fromarray(img_rgb)
        pil_mask = Image.fromarray(dilated)
        result_pil = lama(pil_image, pil_mask)
        result_np = np.array(result_pil)

        # Feathered composite
        feather = cv2.GaussianBlur(dilated.astype(np.float32), (15, 15), 0)
        alpha = np.clip(feather / 255.0, 0.0, 1.0)
        alpha3 = np.stack([alpha] * 3, axis=-1)
        final = (
            img_rgb.astype(np.float32) * (1.0 - alpha3) +
            result_np.astype(np.float32) * alpha3
        ).clip(0, 255).astype(np.uint8)
        return final

    except Exception as e:
        logger.warning(f"LaMa unavailable ({e}) — falling back to OpenCV TELEA.")
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        restored_bgr = cv2.inpaint(bgr, dilated, inpaintRadius=21, flags=cv2.INPAINT_TELEA)
        return cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# Pass-2 helpers (structural color-sampled fill)
# ─────────────────────────────────────────────────────────────────────────────

def _sample_surrounding_color(
    img_rgb: np.ndarray, mask: np.ndarray, ring_radius: int = 40
) -> np.ndarray:
    """Median RGB of pixels in a ring just outside the mask boundary."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_radius * 2 + 1, ring_radius * 2 + 1)
    )
    expanded = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    ring = expanded & ~mask.astype(bool)

    if np.any(ring):
        return np.median(img_rgb[ring].astype(np.float32), axis=0)
    return np.median(img_rgb.reshape(-1, 3).astype(np.float32), axis=0)


def _restore_structural(img_rgb: np.ndarray, combined_mask: np.ndarray) -> np.ndarray:
    """
    Fill structural defect areas (exposed brick, spalling, peeling) with
    color sampled from the surrounding intact facade, then TELEA-blend edges.
    """
    if not np.any(combined_mask):
        return img_rgb.copy()

    result = img_rgb.copy().astype(np.float32)
    base_color = _sample_surrounding_color(img_rgb, combined_mask, ring_radius=40)

    # Plaster-like noise: mild luminance variation
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, 10.0, img_rgb.shape).astype(np.float32)

    fill = np.clip(
        np.full(img_rgb.shape, base_color, dtype=np.float32) + noise, 0, 255
    )
    result[combined_mask.astype(bool)] = fill[combined_mask.astype(bool)]
    result_u8 = result.clip(0, 255).astype(np.uint8)

    # TELEA pass to blend hard edges
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_mask = (
        cv2.dilate(combined_mask.astype(np.uint8), edge_kernel, iterations=2) -
        cv2.erode(combined_mask.astype(np.uint8), edge_kernel, iterations=1)
    ).clip(0, 255).astype(np.uint8)

    if np.any(edge_mask):
        bgr = cv2.cvtColor(result_u8, cv2.COLOR_RGB2BGR)
        bgr = cv2.inpaint(bgr, edge_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        result_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return result_u8


# ─────────────────────────────────────────────────────────────────────────────
# Pass-3 helpers (synthetic glass fill)
# ─────────────────────────────────────────────────────────────────────────────

def _restore_glass(
    img_rgb: np.ndarray,
    glass_mask: np.ndarray,
    window_mask: np.ndarray,
) -> np.ndarray:
    """
    Fill broken glass areas with a synthetic glass appearance.

    Color is sampled from intact window pixels if available; otherwise
    falls back to a neutral gray-blue. A subtle top-to-bottom brightness
    gradient mimics a sky reflection on glass.
    """
    if not np.any(glass_mask):
        return img_rgb.copy()

    result = img_rgb.copy().astype(np.float32)
    broken = glass_mask.astype(bool)

    # Sample intact window glass pixels (window area that is NOT broken)
    intact = window_mask.astype(bool) & ~broken
    if np.any(intact):
        glass_color = np.median(img_rgb[intact].astype(np.float32), axis=0)
    else:
        glass_color = np.array([185, 205, 218], dtype=np.float32)  # neutral sky-blue

    h, w = img_rgb.shape[:2]

    # Sky-reflection gradient: top row +15 brightness, bottom row -5
    y_coords = np.arange(h, dtype=np.float32)[:, None]
    gradient = (15.0 * (1.0 - y_coords / h) - 5.0 * (y_coords / h))

    fill = np.full(img_rgb.shape, glass_color, dtype=np.float32)
    fill += gradient[:, :, np.newaxis]  # broadcast across RGB

    # Subtle luminance noise (glass is smooth, so very small std)
    rng = np.random.default_rng(seed=7)
    fill += rng.normal(0, 4.0, img_rgb.shape).astype(np.float32)
    fill = np.clip(fill, 0, 255)

    result[broken] = fill[broken]
    result_u8 = result.clip(0, 255).astype(np.uint8)

    # Feather the fill boundary into the window frame
    dilated_m = cv2.dilate(glass_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(dilated_m.astype(np.float32), (9, 9), 0)
    alpha = np.clip(alpha, 0, 1)
    alpha3 = np.stack([alpha] * 3, axis=-1)

    blended = (
        img_rgb.astype(np.float32) * (1 - alpha3) +
        result_u8.astype(np.float32) * alpha3
    ).clip(0, 255).astype(np.uint8)

    return blended


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def restore_facade(
    img_rgb: np.ndarray,
    defect_masks: dict,
    output_path: str,
    device: str = "cuda",        # kept for API compatibility; LaMa auto-selects
    element_defect_masks: dict = None,
    geom_masks: dict = None,
    **_kwargs,
) -> str:
    """
    Full multi-pass facade restoration.

    Pass 1 — surface defects (LaMa / TELEA):
        crack, water_damage, efflorescence, moss, rust

    Pass 2 — structural defects (color-sampled fill + TELEA edge blend):
        exposed_brick, spalling, peeling

    Pass 3 — broken glass (synthetic glass fill with sky-reflection gradient):
        broken_glass

    Args:
        img_rgb:              Input image, RGB uint8.
        defect_masks:         {name: bool ndarray} — wall defect masks.
        output_path:          Where to save the JPEG result.
        device:               Ignored (LaMa auto-selects device).
        element_defect_masks: Optional element defect masks (broken_glass, …).
        geom_masks:           Optional geometry masks (window, door, …).
    """
    h, w = img_rgb.shape[:2]
    canvas = img_rgb.copy()

    # ── Pass 1: surface inpainting ───────────────────────────────────────────
    surface_combined = np.zeros((h, w), dtype=np.uint8)
    for k, v in defect_masks.items():
        if k in SURFACE_DEFECTS and v is not None and np.any(v):
            surface_combined |= v.astype(np.uint8)

    if np.any(surface_combined):
        logger.info("Pass 1: surface inpainting (LaMa / TELEA)…")
        canvas = _inpaint_surface(canvas, surface_combined)

    # ── Pass 2: structural color-fill ────────────────────────────────────────
    structural_combined = np.zeros((h, w), dtype=bool)
    for k, v in defect_masks.items():
        if k in STRUCTURAL_DEFECTS and v is not None and np.any(v):
            structural_combined |= v.astype(bool)

    if np.any(structural_combined):
        logger.info("Pass 2: structural fill (color-sampled + TELEA edges)…")
        canvas = _restore_structural(canvas, structural_combined.astype(np.uint8))

    # ── Pass 3: synthetic glass ───────────────────────────────────────────────
    if element_defect_masks is not None:
        glass_mask = element_defect_masks.get("broken_glass")
        if glass_mask is not None and np.any(glass_mask):
            window_mask = (
                geom_masks.get("window", np.zeros((h, w), dtype=bool))
                if geom_masks else np.zeros((h, w), dtype=bool)
            )
            logger.info("Pass 3: synthetic glass fill for broken windows…")
            canvas = _restore_glass(canvas, glass_mask.astype(np.uint8), window_mask)

    # ── Nothing to do ────────────────────────────────────────────────────────
    restored_any = (
        np.any(surface_combined) or
        np.any(structural_combined) or
        (element_defect_masks is not None and
         np.any(element_defect_masks.get("broken_glass", np.zeros(1))))
    )
    if not restored_any:
        logger.info("No defects detected — skipping restoration.")

    cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    logger.info(f"Restoration saved → {output_path}")

    gc.collect()
    return output_path
