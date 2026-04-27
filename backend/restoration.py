"""
Facade Restoration — Stable Diffusion Inpainting.
Uses runwayml/stable-diffusion-inpainting to fill detected defect areas
with realistic restored facade texture.
"""

import os
import gc
import cv2
import torch
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Lazy-loaded singleton
_pipeline = None


def _load_pipeline(device: str = "cuda"):
    """Load the SD inpainting pipeline (cached singleton)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    logger.info("Loading Stable Diffusion Inpainting model...")
    from diffusers import StableDiffusionInpaintPipeline

    _pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    # Optimizations
    if device == "cuda":
        try:
            _pipeline.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory-efficient attention enabled")
        except Exception:
            pass
        _pipeline.enable_attention_slicing()

    logger.info("SD Inpainting model loaded successfully.")
    return _pipeline


def restore_facade(
    img_rgb: np.ndarray,
    defect_masks: dict,
    output_path: str,
    device: str = "cuda",
    prompt: str = (
        "clean restored building facade wall, smooth plaster surface, "
        "repaired building exterior, no cracks no damage, photorealistic"
    ),
    negative_prompt: str = (
        "cracks, damage, broken, dirty, graffiti, stains, moss, "
        "rust, peeling paint, deteriorated, blurry, low quality"
    ),
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    strength: float = 0.75,
) -> str:
    """
    Restore facade by inpainting defect areas using Stable Diffusion.

    Args:
        img_rgb: Original image in RGB format
        defect_masks: Dict of {defect_type: boolean_mask}
        output_path: Where to save the restored image
        device: 'cuda' or 'cpu'

    Returns:
        Path to saved restoration image
    """
    # Build combined defect mask
    h, w = img_rgb.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for mask in defect_masks.values():
        if mask is not None and np.any(mask):
            combined_mask |= mask.astype(np.uint8)

    if not np.any(combined_mask):
        # No defects — just save original
        cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return output_path

    # Dilate mask slightly for better inpainting coverage
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated_mask = cv2.dilate(combined_mask * 255, kernel, iterations=2)

    # SD Inpainting expects 512x512 images for best quality
    # Process in tiles if image is larger, or resize
    target_size = 512

    # Resize to SD-compatible size while preserving aspect ratio
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    # Round to nearest 8 (SD requirement)
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8

    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(dilated_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Convert to PIL
    pil_image = Image.fromarray(img_resized).convert("RGB")
    pil_mask = Image.fromarray(mask_resized).convert("L")

    # Load pipeline
    pipe = _load_pipeline(device)

    logger.info(f"Running SD inpainting ({new_w}x{new_h}, {num_inference_steps} steps)...")

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            width=new_w,
            height=new_h,
        )

    restored_pil = result.images[0]

    # Resize back to original dimensions
    restored_np = np.array(restored_pil)
    restored_full = cv2.resize(restored_np, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Blend: only replace defect areas, keep original elsewhere
    # Use soft blending at mask borders for seamless transition
    mask_full = cv2.resize(dilated_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    # Feather the mask edges
    mask_blurred = cv2.GaussianBlur(mask_full.astype(np.float32), (21, 21), 0)
    mask_blurred = np.clip(mask_blurred / 255.0, 0, 1)
    mask_3ch = np.stack([mask_blurred] * 3, axis=-1)

    # Composite: original * (1-mask) + restored * mask
    final = (img_rgb.astype(np.float32) * (1 - mask_3ch) +
             restored_full.astype(np.float32) * mask_3ch).astype(np.uint8)

    # Save
    cv2.imwrite(output_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

    # Free VRAM
    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"Restoration saved to {output_path}")
    return output_path
