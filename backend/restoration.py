"""
Facade Restoration — LaMa Inpainting + OpenCV fallback.

Small masks (<2 % of image area)  → cv2.inpaint (TELEA, instant, no GPU needed)
Large masks (≥2 %)                → LaMa (Large Mask Inpainting, ~200 MB model,
                                    downloads automatically on first run)

LaMa advantages over Stable Diffusion for facades:
  • Trained specifically for texture repair, not image generation
  • Preserves surrounding plaster / brick / concrete texture seamlessly
  • Handles arbitrarily large masks without "hallucinating" new content
  • Works at full resolution — no 512 px downscale artefacts
  • 10-20× faster than SD, runs on CPU if no GPU is available
"""

import cv2
import gc
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

_lama: object = None   # lazy singleton


def _load_lama():
    """Load LaMa model once and cache it."""
    global _lama
    if _lama is not None:
        return _lama
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError as e:
        raise RuntimeError(
            "simple-lama-inpainting not installed. "
            "Run: pip install simple-lama-inpainting"
        ) from e

    logger.info("Loading LaMa inpainting model (first run downloads ~200 MB)…")
    _lama = SimpleLama()
    logger.info("LaMa model ready.")
    return _lama


def restore_facade(
    img_rgb: np.ndarray,
    defect_masks: dict,
    output_path: str,
    device: str = "cuda",        # kept for API compatibility; LaMa auto-selects
    **_kwargs,                   # absorb legacy SD params (prompt, strength, …)
) -> str:
    """
    Restore facade by inpainting defect areas.

    Args:
        img_rgb:      Image in RGB uint8 format.
        defect_masks: {any_key: bool ndarray} — defect pixel masks.
        output_path:  Where to save the result (JPEG).
        device:       Ignored — LaMa selects device automatically.

    Returns:
        output_path
    """
    h, w = img_rgb.shape[:2]
    total_px = h * w

    # ── 1. Build combined binary mask ────────────────────────────────────────
    combined = np.zeros((h, w), dtype=np.uint8)
    for mask in defect_masks.values():
        if mask is not None and np.any(mask):
            combined |= mask.astype(np.uint8)

    if not np.any(combined):
        logger.info("No defects detected — skipping restoration.")
        cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return output_path

    # ── 2. Moderate dilation (cover crack edges, not the whole facade) ───────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(combined * 255, kernel, iterations=2)   # ~14 px border

    mask_ratio = float(np.count_nonzero(dilated)) / total_px
    logger.info(f"Defect mask coverage: {mask_ratio:.1%}")

    # ── 3. Small mask → OpenCV TELEA (no model, <1 ms) ──────────────────────
    if mask_ratio < 0.02:
        logger.info("Small mask — using OpenCV TELEA inpaint.")
        restored = cv2.inpaint(
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
            dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA
        )
        cv2.imwrite(output_path, restored)
        return output_path

    # ── 3b. Mask too large for meaningful inpainting → return original ────────
    if mask_ratio > 0.40:
        logger.warning(
            f"Defect mask covers {mask_ratio:.1%} of the image — "
            "too large for inpainting. Returning original."
        )
        cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return output_path

    # ── 4. Large mask → LaMa (OpenCV TELEA if LaMa unavailable) ─────────────
    bgr_orig = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    try:
        logger.info("Large mask — using LaMa inpainting…")
        lama = _load_lama()

        pil_image = Image.fromarray(img_rgb)
        pil_mask  = Image.fromarray(dilated)

        result_pil = lama(pil_image, pil_mask)
        result_np  = np.array(result_pil)

        # ── 5. Feathered composite ───────────────────────────────────────────
        feather = cv2.GaussianBlur(dilated.astype(np.float32), (15, 15), 0)
        alpha   = np.clip(feather / 255.0, 0.0, 1.0)
        alpha3  = np.stack([alpha] * 3, axis=-1)

        final = (
            img_rgb.astype(np.float32)   * (1.0 - alpha3) +
            result_np.astype(np.float32) * alpha3
        ).clip(0, 255).astype(np.uint8)

        cv2.imwrite(output_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
        logger.info(f"LaMa restoration saved → {output_path}")

    except Exception as e:
        logger.warning(f"LaMa unavailable ({e}) — falling back to OpenCV TELEA.")
        restored = cv2.inpaint(bgr_orig, dilated, inpaintRadius=21, flags=cv2.INPAINT_TELEA)
        cv2.imwrite(output_path, restored)

    gc.collect()
    return output_path
