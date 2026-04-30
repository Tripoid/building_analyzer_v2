"""
FacadeAnalyzer — ML Pipeline for building facade analysis.

Step 1: SAM3.1 → geometry detection (windows, doors, balconies)
Step 2: rembg + SAM3.1 → wall and element defect detection
Step 3: SAM3.1 → material identification (text-prompted per class)
Step 4: Wall layer classification based on defect and material overlap

SAM3.1 is the only ML model — handles geometry, defects, and materials.
  set_image() once per image, set_text_prompt() once per class.
  Returns masks, boxes, scores directly — no separate detector needed.
"""

import cv2
import torch
import gc
import os
import uuid
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

# Per-class text prompts for SAM3.1 geometry detection.
# One entry per architectural class; set_text_prompt() is called once per entry.
GEOMETRY_SAM3_PROMPTS = {
    "window": "window glass pane window frame glazing dormer window",
    "door": "entrance door front door gate doorway",
    "balcony": "balcony terrace loggia balcony slab",
    "building": "building facade wall",
    "roof": "roof rooftop gutter eaves parapet",
    "molding": "architectural molding decorative cornice frieze pilaster",
    "pipe": "drainpipe downspout rain gutter pipe",
    "ac_unit": "air conditioner AC unit ventilation grille split system",
}

CLASS_MAP_GEOMETRY = {
    "window": ["window", "pane", "glass", "glazing", "dormer"],
    "door": ["door", "entrance", "gate", "front door"],
    "balcony": ["balcony", "terrace", "loggia"],
    "building": ["building", "facade", "wall"],
    "roof": ["roof", "rooftop", "gutter", "eaves", "parapet"],
    "molding": ["cornice", "molding", "decoration", "frieze", "pilaster", "architectural"],
    "pipe": ["drainpipe", "pipe", "downspout"],
    "ac_unit": ["air conditioner", "unit", "ventilation", "grille", "split"],
}

# Per-class text prompts for SAM3.1 defect detection.
WALL_DEFECT_SAM3_PROMPTS = {
    "crack":         "wall crack facade crack concrete crack surface fracture hairline crack",
    "peeling":       "peeling plaster flaking plaster flaking paint delaminated coating blistered render",
    "exposed_brick": "exposed brick bare brick missing plaster unplastered brick masonry",
    "water_damage":  "water stain water damage moisture stain damp patch leakage mark",
    "rust":          "rust stain corrosion spot iron rust mark oxidation stain",
    "moss":          "moss lichen algae on wall biological growth green deposit",
    "efflorescence": "efflorescence salt deposit white crystalline deposit salt stain",
    "spalling":      "spalling concrete concrete deterioration concrete defect chipped concrete",
}

ELEMENT_DEFECT_SAM3_PROMPTS = {
    "broken_glass":    "broken window glass shattered glass cracked glass pane damaged glazing",
    "damaged_wood":    "damaged wooden door rotting wood deteriorated wood decay",
    "rusty_metal":     "rusty metal railing corroded metal rusted iron oxidized metal surface",
    "damaged_railing": "cracked balcony railing broken railing deformed railing damaged balcony fence",
}

CLASS_MAP_WALL_DEFECTS = {
    "crack":        ["crack", "fracture", "fissure", "crevice", "cleft"],
    "peeling":      ["peeling", "flaking", "plaster", "delaminated", "blistered", "render", "spalled render"],
    "exposed_brick":["exposed brick", "brick", "masonry", "bare brick", "unplastered"],
    "water_damage": ["water stain", "water damage", "moisture", "stain", "leakage", "damp", "efflorescence stain"],
    "rust":         ["rust", "corrosion", "iron stain", "oxide"],
    "moss":         ["moss", "lichen", "algae", "biological", "growth", "green deposit"],
    "efflorescence":["efflorescence", "salt deposit", "crystalline", "white deposit", "salt stain"],
    "spalling":     ["spalling", "concrete", "deterioration", "defect", "chipping"],
}

CLASS_MAP_ELEMENT_DEFECTS = {
    "broken_glass":    ["broken", "glass", "shattered", "cracked glass", "damaged glass"],
    "damaged_wood":    ["damaged", "wooden", "door", "wood", "rot", "deteriorated"],
    "rusty_metal":     ["rusty", "metal", "corrosion", "iron", "rusted"],
    "damaged_railing": ["cracked", "railing", "broken railing", "damaged railing", "deformed"],
}

# Per-class text prompts for SAM3.1 material detection.
MATERIAL_SAM3_PROMPTS = {
    "concrete":           "grey concrete stone wall base foundation",
    "brick":              "terracotta red brick masonry wall surface exposed brick",
    "cement_plaster":     "plain smooth cement plaster wall facade",
    "decorative_plaster": "textured decorative structured plaster facade surface",
    "molding":            "architectural molding decorative cornice frieze ornament",
    "ceramic_tile":       "ceramic tile cladding facade glazed tile",
    "painted_surface":    "painted wall smooth painted surface color coat",
}

CLASS_MAP_MATERIALS = {
    "concrete": ["concrete"],
    "brick": ["brick"],
    "cement_plaster": ["plaster"],
    "decorative_plaster": ["textured plaster"],
    "molding": ["molding", "cornice"],
    "ceramic_tile": ["ceramic", "tile"],
    "painted_surface": ["painted"],
}


COLORS_GEOMETRY = {
    "window": [0, 255, 255],
    "door": [255, 0, 0],
    "balcony": [255, 0, 255],
    "building": [0, 0, 255],
    "roof": [128, 128, 0],
    "molding": [200, 200, 200],
    "pipe": [128, 0, 128],
    "ac_unit": [0, 128, 128],
    "unknown": [255, 255, 0],
}

COLORS_WALL_DEFECTS = {
    "crack": [255, 0, 0],
    "peeling": [241, 196, 15],
    "exposed_brick": [230, 126, 34],
    "water_damage": [52, 152, 219],
    "rust": [192, 57, 43],
    "moss": [39, 174, 96],
    "efflorescence": [236, 240, 241],
    "spalling": [149, 165, 166],
}

COLORS_ELEMENT_DEFECTS = {
    "broken_glass": [0, 255, 255],
    "damaged_wood": [255, 20, 147],
    "rusty_metal": [205, 92, 0],
    "damaged_railing": [148, 103, 189],
}

COLORS_MATERIALS = {
    "concrete": [127, 140, 141],
    "brick": [211, 84, 0],
    "cement_plaster": [243, 214, 115],
    "decorative_plaster": [255, 235, 180],
    "molding": [236, 240, 241],
    "ceramic_tile": [46, 134, 193],
    "painted_surface": [175, 215, 160],
}

# Wall layer definitions for multi-layer analysis
WALL_LAYERS = {
    "finish": {"name": "Финишный слой", "depth_mm": 3, "materials": ["painted_surface", "decorative_plaster"]},
    "base_plaster": {"name": "Базовая штукатурка", "depth_mm": 20, "materials": ["cement_plaster"]},
    "insulation": {"name": "Утеплитель", "depth_mm": 50, "materials": []},
    "structural": {"name": "Несущий слой", "depth_mm": 200, "materials": ["brick", "concrete"]},
}

# Defect → affected wall layers mapping
DEFECT_LAYER_IMPACT = {
    "crack": {"surface": ["finish"], "deep": ["finish", "base_plaster", "structural"]},
    "peeling": ["finish", "base_plaster"],
    "exposed_brick": ["finish", "base_plaster"],
    "water_damage": ["finish", "base_plaster"],
    "rust": ["finish"],
    "moss": ["finish"],
    "efflorescence": ["finish"],
    "spalling": ["finish", "base_plaster", "structural"],
}

SEVERITY_LABELS = {
    "ru": {
        "high": "Высокая",
        "medium": "Средняя",
        "low": "Низкая",
    }
}

CONDITION_LABELS = {
    "ru": {
        "good": "Хорошее",
        "satisfactory": "Удовлетворительное",
        "needs_repair": "Требует ремонта",
        "critical": "Аварийное",
    }
}

# Russian display names for defect types
DEFECT_DISPLAY_RU = {
    "crack": "Трещины",
    "peeling": "Отслоение штукатурки",
    "exposed_brick": "Оголённая кладка",
    "water_damage": "Следы протечек",
    "rust": "Коррозия",
    "moss": "Биопоражение",
    "efflorescence": "Высолы",
    "spalling": "Разрушение бетона",
    "broken_glass": "Повреждённое остекление",
    "damaged_wood": "Повреждённое дерево",
    "rusty_metal": "Ржавый металл",
    "damaged_railing": "Повреждённые ограждения",
}

# Russian display names for materials
MATERIAL_DISPLAY_RU = {
    "concrete": "Бетон",
    "brick": "Кирпичная кладка",
    "cement_plaster": "Цементная штукатурка",
    "decorative_plaster": "Декоративная штукатурка",
    "molding": "Лепнина / карнизы",
    "ceramic_tile": "Керамическая плитка",
    "painted_surface": "Окрашенная поверхность",
}

# Russian descriptions for defect types
DEFECT_DESCRIPTION_RU = {
    "crack": "Линейные повреждения поверхности различной глубины",
    "peeling": "Отслоение и разрушение штукатурного слоя",
    "exposed_brick": "Участки с полностью утраченным штукатурным покрытием",
    "water_damage": "Потёки, разводы и пятна от воздействия влаги",
    "rust": "Ржавчина и коррозионные пятна на поверхности",
    "moss": "Мох, лишайник и другие биологические отложения",
    "efflorescence": "Белые солевые отложения на поверхности",
    "spalling": "Разрушение и отколы бетонного покрытия",
    "broken_glass": "Разбитые или повреждённые стеклопакеты",
    "damaged_wood": "Гниль, трещины и разрушение деревянных элементов",
    "rusty_metal": "Коррозия металлических элементов фасада",
    "damaged_railing": "Деформация или разрушение балконных ограждений",
}


def _get_base_class(raw_label: str, class_map: dict) -> Optional[str]:
    """Match a raw label to a base class using synonym lookup."""
    raw_label = raw_label.lower()
    for base, synonyms in class_map.items():
        if any(syn in raw_label for syn in synonyms):
            return base
    return None



def _adaptive_params(image_shape: Tuple[int, int]) -> dict:
    """Return adaptive ML parameters based on image resolution."""
    h, w = image_shape[:2]
    total_px = h * w
    if total_px > 2_000_000:  # >2MP
        return {"points_per_side": 48, "min_mask_area": 500, "sam3_threshold": 0.40}
    elif total_px > 800_000:  # >0.8MP
        return {"points_per_side": 32, "min_mask_area": 300, "sam3_threshold": 0.40}
    else:
        return {"points_per_side": 16, "min_mask_area": 150, "sam3_threshold": 0.40}


class FacadeAnalyzer:
    """
    Production ML pipeline for building facade analysis.
    Loads models once and caches them in memory.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_loaded = False
        self._sam3_processor = None
        self._sam3_model = None

    @staticmethod
    def _cast_all_to_bf16(obj, _visited=None):
        """Recursively cast every nn.Module to bfloat16.

        nn.Module.bfloat16() only reaches *registered* submodules.
        SAM3's ViT stores transformer blocks in a plain Python list, so they
        are NOT registered and are skipped by the standard cast.  This helper
        walks every attribute (including plain lists/dicts) so that all blocks'
        norm2 and mlp weights are cast alongside the registered layers.
        """
        if _visited is None:
            _visited = set()
        if id(obj) in _visited:
            return
        _visited.add(id(obj))

        if isinstance(obj, torch.nn.Module):
            for p in obj.parameters(recurse=False):
                p.data = p.data.to(torch.bfloat16)
            for key in list(obj._buffers):
                if obj._buffers[key] is not None:
                    obj._buffers[key] = obj._buffers[key].to(torch.bfloat16)
            for attr in vars(obj).values():
                if isinstance(attr, torch.nn.Module):
                    FacadeAnalyzer._cast_all_to_bf16(attr, _visited)
                elif isinstance(attr, (list, tuple)):
                    for item in attr:
                        if isinstance(item, torch.nn.Module):
                            FacadeAnalyzer._cast_all_to_bf16(item, _visited)
                elif isinstance(attr, dict):
                    for v in attr.values():
                        if isinstance(v, torch.nn.Module):
                            FacadeAnalyzer._cast_all_to_bf16(v, _visited)

    def load_models(self):
        """Load and cache all ML models. Call once at startup."""
        if self.models_loaded:
            return

        logger.info("Loading SAM3.1 (text-prompted detection + segmentation)...")
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self._sam3_model = build_sam3_image_model().to(self.device)
        self._sam3_processor = Sam3Processor(self._sam3_model)

        if self.device == "cuda":
            FacadeAnalyzer._cast_all_to_bf16(self._sam3_model)

            # Verify which params (if any) are still float32 after the cast.
            # This tells us whether the cast reached all submodules.
            still_f32 = [
                (n, tuple(p.shape))
                for n, p in self._sam3_model.named_parameters()
                if p.dtype == torch.float32
            ]
            if still_f32:
                logger.warning(
                    f"SAM3: {len(still_f32)} params still float32 after cast "
                    f"(first 5): {still_f32[:5]}"
                )
                # Force cast the remaining ones using the standard API.
                self._sam3_model.bfloat16()
            else:
                logger.info("SAM3: all parameters cast to bfloat16.")

            # Runtime safety net: patch Block.forward so x is cast to the
            # block's own parameter dtype before every transformer block.
            # This handles any residual mismatch (e.g. lazy-loaded float32
            # weights loaded after our cast).
            try:
                from sam3.model.vitdet import Block as _SAM3Block
                _orig_fwd = _SAM3Block.forward
                def _dtype_safe_fwd(self_blk, x):
                    try:
                        target = next(iter(self_blk.parameters())).dtype
                        if x.dtype != target:
                            x = x.to(target)
                    except StopIteration:
                        pass
                    return _orig_fwd(self_blk, x)
                _SAM3Block.forward = _dtype_safe_fwd
                logger.info("SAM3 Block.forward patched for dtype safety.")
            except Exception as patch_err:
                logger.warning(f"Could not patch SAM3 Block: {patch_err}")

        self.models_loaded = True
        logger.info("All models loaded successfully.")

    # ─────────────────────────────────────────────────
    # STEP 0: Preprocessing
    # ─────────────────────────────────────────────────

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """CLAHE normalization + adaptive denoising + resize."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize to max 1024px
        max_dim = 1024
        h, w = img_bgr.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # CLAHE on L channel
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        img_clahe = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        # Adaptive denoising
        brightness = np.mean(cl)
        h_param = 3 if brightness > 120 else 7
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)
        img_denoised = cv2.fastNlMeansDenoisingColored(img_rgb, None, h_param, h_param, 7, 21)

        return img_denoised

    # ─────────────────────────────────────────────────
    # STEP 1: Geometry Detection (SAM3.1)
    # ─────────────────────────────────────────────────

    def detect_geometry(self, img_rgb: np.ndarray) -> Tuple[Dict[str, np.ndarray], list]:
        """Detect architectural elements using SAM3.1 text-prompted segmentation."""
        image = Image.fromarray(img_rgb)
        original_size = img_rgb.shape[:2]
        params = _adaptive_params(img_rgb.shape)
        threshold = params["sam3_threshold"]

        geom_masks = {k: np.zeros(original_size, dtype=bool) for k in GEOMETRY_SAM3_PROMPTS}
        detections = []

        try:
            inference_state = self._sam3_processor.set_image(image)

            for class_key, prompt in GEOMETRY_SAM3_PROMPTS.items():
                output = self._sam3_processor.set_text_prompt(
                    state=inference_state, prompt=prompt
                )
                masks = output.get("masks")
                scores = output.get("scores")
                boxes = output.get("boxes")
                if masks is None or len(masks) == 0:
                    continue

                for i, (mask, score) in enumerate(zip(masks, scores)):
                    if float(score) < threshold:
                        continue
                    m = np.asarray(mask, dtype=bool)
                    if np.any(m):
                        geom_masks[class_key] |= m
                        box = boxes[i].tolist() if boxes is not None and len(boxes) > i else []
                        detections.append({
                            "class": class_key,
                            "score": float(score),
                            "box": box,
                        })
        except Exception:
            import traceback
            logger.error("detect_geometry failed:\n" + traceback.format_exc())
            raise

        logger.info(f"Geometry: detected {len(detections)} elements")
        return geom_masks, detections

    # ─────────────────────────────────────────────────
    # STEP 2: Wall Defect Detection
    # ─────────────────────────────────────────────────

    def detect_defects(
        self, img_rgb: np.ndarray, geom_masks: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Detect wall and element defects using rembg + SAM3.1."""
        from rembg import remove
        original_size = img_rgb.shape[:2]
        params = _adaptive_params(img_rgb.shape)

        # 1. Isolate building silhouette
        logger.info("Generating building silhouette (rembg)...")
        rgba = remove(img_rgb)
        silhouette = rgba[:, :, 3] > 128

        # 2. Compute bare wall mask
        elements = np.zeros(original_size, dtype=bool)
        for key in ["window", "door", "balcony", "roof", "molding", "pipe", "ac_unit"]:
            if key in geom_masks:
                elements |= geom_masks[key]
        bare_wall = silhouette & ~elements

        # 3. Wall defects via SAM3.1: per-class text prompts → masks directly.
        # Use silhouette as region mask so that masks stay within the building.
        logger.info("Scanning wall defects (SAM3.1)...")
        wall_defect_masks = self._sam3_segment(
            img_rgb, WALL_DEFECT_SAM3_PROMPTS, original_size,
            region_mask=silhouette, threshold=params["sam3_threshold"],
        )

        # Sanity cap for wall defects: a single type covering >70% of silhouette is a false positive
        building_px = int(silhouette.sum()) or 1
        for key in list(wall_defect_masks.keys()):
            coverage = wall_defect_masks[key].sum() / building_px
            if coverage > 0.70:
                logger.warning(f"Wall defect '{key}' covers {coverage:.1%} — zeroing (false positive)")
                wall_defect_masks[key] = np.zeros(original_size, dtype=bool)

        # 4. Element defects via SAM3.1
        logger.info("Scanning element defects (SAM3.1)...")
        element_defect_masks = self._sam3_segment(
            img_rgb, ELEMENT_DEFECT_SAM3_PROMPTS, original_size,
            threshold=params["sam3_threshold"],
        )

        # Spatial routing: element defects only within their geometry
        mask_windows = geom_masks.get("window", np.zeros(original_size, dtype=bool))
        mask_doors = geom_masks.get("door", np.zeros(original_size, dtype=bool))
        mask_balconies = geom_masks.get("balcony", np.zeros(original_size, dtype=bool))

        if "broken_glass" in element_defect_masks:
            element_defect_masks["broken_glass"] &= mask_windows
        if "damaged_wood" in element_defect_masks:
            element_defect_masks["damaged_wood"] &= mask_doors
        if "rusty_metal" in element_defect_masks:
            element_defect_masks["rusty_metal"] &= (mask_balconies | mask_windows)
        if "damaged_railing" in element_defect_masks:
            element_defect_masks["damaged_railing"] &= mask_balconies

        # Sanity cap: element defect covering >25% of total silhouette is a false positive
        building_px = int(silhouette.sum()) or 1
        for key in list(element_defect_masks.keys()):
            coverage = element_defect_masks[key].sum() / building_px
            if coverage > 0.25:
                logger.warning(f"Element defect '{key}' covers {coverage:.1%} — zeroing (false positive)")
                element_defect_masks[key] = np.zeros(original_size, dtype=bool)

        torch.cuda.empty_cache()
        gc.collect()

        return wall_defect_masks, element_defect_masks

    # ─────────────────────────────────────────────────
    # STEP 3: Material Identification
    # ─────────────────────────────────────────────────

    def analyze_materials(
        self, img_rgb: np.ndarray, geom_masks: Dict[str, np.ndarray],
        wall_defect_masks: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """Identify facade materials using SAM3.1 text-prompted segmentation."""
        from rembg import remove
        original_size = img_rgb.shape[:2]

        rgba = remove(img_rgb)
        silhouette = rgba[:, :, 3] > 128

        # Bare wall = silhouette minus architectural elements
        elements = np.zeros(original_size, dtype=bool)
        for key in ["window", "door", "balcony", "roof", "pipe", "ac_unit"]:
            if key in geom_masks:
                elements |= geom_masks[key]
        bare_wall = silhouette & ~elements

        logger.info("SAM3.1 material segmentation...")
        raw = self._sam3_segment(img_rgb, MATERIAL_SAM3_PROMPTS, original_size, region_mask=bare_wall)

        final_materials: Dict[str, np.ndarray] = {
            k: np.zeros(original_size, dtype=bool) for k in MATERIAL_SAM3_PROMPTS
        }

        # Z-index assembly: structural → base → surface → decorative → top
        concrete_m = raw["concrete"] & bare_wall
        final_materials["concrete"] |= concrete_m

        # Cement plaster fills remaining bare wall
        plaster_m = bare_wall & ~concrete_m
        final_materials["cement_plaster"] |= plaster_m

        # Brick overrides plaster
        brick_m = raw["brick"] & plaster_m
        if np.any(brick_m):
            final_materials["brick"] |= brick_m
            final_materials["cement_plaster"] &= ~brick_m

        # Decorative plaster overrides cement plaster
        dec_m = raw["decorative_plaster"] & final_materials["cement_plaster"]
        if np.any(dec_m):
            final_materials["decorative_plaster"] |= dec_m
            final_materials["cement_plaster"] &= ~dec_m

        # Painted surface overrides cement plaster
        paint_m = raw["painted_surface"] & final_materials["cement_plaster"]
        if np.any(paint_m):
            final_materials["painted_surface"] |= paint_m
            final_materials["cement_plaster"] &= ~paint_m

        # Ceramic tile overrides lower layers
        tile_m = raw["ceramic_tile"] & bare_wall
        if np.any(tile_m):
            final_materials["ceramic_tile"] |= tile_m
            for k in ["cement_plaster", "decorative_plaster", "brick", "painted_surface"]:
                final_materials[k] &= ~tile_m

        # Molding — highest z-index
        final_materials["molding"] |= raw["molding"] & bare_wall

        # Augment brick from exposed_brick defect mask
        if wall_defect_masks:
            exposed = wall_defect_masks.get("exposed_brick", np.zeros(original_size, dtype=bool))
            if np.any(exposed):
                final_materials["cement_plaster"] &= ~exposed
                final_materials["brick"] |= (exposed & silhouette)

        torch.cuda.empty_cache()
        gc.collect()

        logger.info("Material analysis complete (SAM3.1).")
        return final_materials

    # ─────────────────────────────────────────────────
    # STEP 4: Wall Layer Classification
    # ─────────────────────────────────────────────────

    def classify_wall_layers(
        self, defect_masks: Dict[str, np.ndarray], material_masks: Dict[str, np.ndarray],
        original_size: Tuple[int, int]
    ) -> Dict[str, dict]:
        """Classify which wall layers each defect impacts."""
        layer_analysis = {}

        for defect_type, mask in defect_masks.items():
            if not np.any(mask):
                continue

            area_px = int(mask.sum())
            impact = DEFECT_LAYER_IMPACT.get(defect_type, ["finish"])

            # For cracks, determine depth
            if defect_type == "crack" and isinstance(impact, dict):
                # Heuristic: cracks overlapping exposed brick = deep
                brick_mask = material_masks.get("brick", np.zeros(original_size, dtype=bool))
                overlap = np.sum(mask & brick_mask)
                if overlap > area_px * 0.1:
                    affected_layers = impact["deep"]
                    crack_depth = "deep"
                else:
                    affected_layers = impact["surface"]
                    crack_depth = "surface"
                layer_analysis[defect_type] = {
                    "area_px": area_px,
                    "affected_layers": affected_layers,
                    "crack_depth": crack_depth,
                }
            else:
                if isinstance(impact, dict):
                    affected_layers = impact.get("surface", ["finish"])
                else:
                    affected_layers = impact
                layer_analysis[defect_type] = {
                    "area_px": area_px,
                    "affected_layers": affected_layers,
                }

        return layer_analysis

    # ─────────────────────────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────────────────────────

    def generate_visualizations(
        self, img_rgb: np.ndarray,
        geom_masks: Dict[str, np.ndarray],
        wall_defect_masks: Dict[str, np.ndarray],
        element_defect_masks: Dict[str, np.ndarray],
        material_masks: Dict[str, np.ndarray],
        output_dir: str,
    ) -> Dict[str, str]:
        """Generate 4 visualization images and save to output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        original_size = img_rgb.shape[:2]
        paths = {}

        # Save original preprocessed image (needed by LayersTab viewer)
        original_path = os.path.join(output_dir, "original.jpg")
        cv2.imwrite(original_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        paths["original"] = original_path

        # 1. Heatmap — defect density
        heatmap = np.zeros(original_size, dtype=np.float32)
        for mask in wall_defect_masks.values():
            heatmap += mask.astype(np.float32)
        for mask in element_defect_masks.values():
            heatmap += mask.astype(np.float32)
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap = np.zeros(original_size, dtype=np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.5, heatmap_colored, 0.5, 0)
        path = os.path.join(output_dir, "heatmap.jpg")
        cv2.imwrite(path, overlay)
        paths["heatmap"] = path

        # 2. Defects — marked with colored boxes
        defects_canvas = img_rgb.copy()
        all_defects = {**COLORS_WALL_DEFECTS, **COLORS_ELEMENT_DEFECTS}
        all_masks = {**wall_defect_masks, **element_defect_masks}
        for dtype, color in all_defects.items():
            mask = all_masks.get(dtype)
            if mask is not None and np.any(mask):
                colored = np.zeros_like(defects_canvas)
                colored[mask] = color
                defects_canvas = cv2.addWeighted(defects_canvas, 1.0, colored, 0.5, 0)
        path = os.path.join(output_dir, "defects.jpg")
        cv2.imwrite(path, cv2.cvtColor(defects_canvas, cv2.COLOR_RGB2BGR))
        paths["defects"] = path

        # 3. Segments — material segmentation
        segments_canvas = np.zeros((*original_size, 3), dtype=np.uint8)
        for mat_name, color in COLORS_MATERIALS.items():
            mask = material_masks.get(mat_name)
            if mask is not None and np.any(mask):
                segments_canvas[mask] = color
        # Add element context — erode SAM masks so they show only the core of
        # windows/doors (not the bloated SAM area) and don't hide material colors.
        elem_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        for key in ["window", "door", "balcony"]:
            mask = geom_masks.get(key)
            if mask is not None and np.any(mask):
                core = cv2.erode(mask.astype(np.uint8), elem_kernel, iterations=4)
                segments_canvas[core.astype(bool)] = [40, 40, 40]
        path = os.path.join(output_dir, "segments.jpg")
        cv2.imwrite(path, cv2.cvtColor(segments_canvas, cv2.COLOR_RGB2BGR))
        paths["segments"] = path

        # 4. Repair overlay — priority repair zones
        repair_canvas = img_rgb.copy()
        priority_colors = {
            "crack": [255, 0, 0],
            "peeling": [255, 165, 0],
            "spalling": [255, 0, 0],
            "exposed_brick": [230, 126, 34],
            "water_damage": [52, 152, 219],
        }
        for dtype, color in priority_colors.items():
            mask = wall_defect_masks.get(dtype)
            if mask is not None and np.any(mask):
                # Draw border around repair zone
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                bgr_color = (color[2], color[1], color[0])
                cv2.drawContours(repair_canvas, contours, -1, bgr_color, 3)
                # Light fill
                colored = np.zeros_like(repair_canvas)
                colored[mask] = color
                repair_canvas = cv2.addWeighted(repair_canvas, 1.0, colored, 0.25, 0)
        path = os.path.join(output_dir, "overlay.jpg")
        cv2.imwrite(path, cv2.cvtColor(repair_canvas, cv2.COLOR_RGB2BGR))
        paths["overlay"] = path

        # 5. Per-layer overlays for interactive viewer
        logger.info("Generating per-layer overlays...")
        layers_info = []

        # Defect layers
        all_defect_colors = {**COLORS_WALL_DEFECTS, **COLORS_ELEMENT_DEFECTS}
        all_defect_masks_combined = {**wall_defect_masks, **element_defect_masks}
        for dtype, color in all_defect_colors.items():
            mask = all_defect_masks_combined.get(dtype)
            if mask is not None and np.any(mask):
                layer_canvas = img_rgb.copy()
                colored = np.zeros_like(layer_canvas)
                colored[mask] = color
                layer_canvas = cv2.addWeighted(layer_canvas, 1.0, colored, 0.5, 0)
                # Draw contours
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                bgr_color = (color[2], color[1], color[0])
                cv2.drawContours(layer_canvas, contours, -1, bgr_color, 2)
                layer_path = os.path.join(output_dir, f"layer_defect_{dtype}.jpg")
                cv2.imwrite(layer_path, cv2.cvtColor(layer_canvas, cv2.COLOR_RGB2BGR))
                paths[f"layer_defect_{dtype}"] = layer_path
                layers_info.append({
                    "id": f"layer_defect_{dtype}",
                    "type": "defect",
                    "key": dtype,
                    "name": DEFECT_DISPLAY_RU.get(dtype, dtype),
                    "color": f"rgb({color[0]},{color[1]},{color[2]})",
                })

        # Material layers
        for mat_name, color in COLORS_MATERIALS.items():
            mask = material_masks.get(mat_name)
            if mask is not None and np.any(mask):
                layer_canvas = img_rgb.copy()
                colored = np.zeros_like(layer_canvas)
                colored[mask] = color
                layer_canvas = cv2.addWeighted(layer_canvas, 0.6, colored, 0.4, 0)
                layer_path = os.path.join(output_dir, f"layer_material_{mat_name}.jpg")
                cv2.imwrite(layer_path, cv2.cvtColor(layer_canvas, cv2.COLOR_RGB2BGR))
                paths[f"layer_material_{mat_name}"] = layer_path
                layers_info.append({
                    "id": f"layer_material_{mat_name}",
                    "type": "material",
                    "key": mat_name,
                    "name": MATERIAL_DISPLAY_RU.get(mat_name, mat_name),
                    "color": f"rgb({color[0]},{color[1]},{color[2]})",
                })

        # 6. Restoration — multi-pass: LaMa surface + structural fill + glass synthesis
        logger.info("Generating restoration visualization (multi-pass)...")
        try:
            from restoration import restore_facade
            restoration_path = os.path.join(output_dir, "restoration.jpg")

            restore_facade(
                img_rgb=img_rgb,
                defect_masks=wall_defect_masks,          # Pass 1 + 2 (surface + structural)
                output_path=restoration_path,
                device=str(self.device),
                element_defect_masks=element_defect_masks,  # Pass 3 (broken glass)
                geom_masks=geom_masks,                      # For glass color sampling
            )
            paths["restoration"] = restoration_path
        except Exception as e:
            logger.warning(f"Restoration failed (non-critical): {e}")
            fallback_path = os.path.join(output_dir, "restoration.jpg")
            cv2.imwrite(fallback_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            paths["restoration"] = fallback_path

        return paths, layers_info

    # ─────────────────────────────────────────────────
    # FULL ANALYSIS PIPELINE
    # ─────────────────────────────────────────────────

    def analyze(self, image_bytes: bytes, output_dir: str = "/tmp/facade_results") -> dict:
        """Run the full analysis pipeline and return structured results."""
        analysis_id = str(uuid.uuid4())
        result_dir = os.path.join(output_dir, analysis_id)
        os.makedirs(result_dir, exist_ok=True)

        # Step 0
        logger.info("Step 0: Preprocessing...")
        img_rgb = self.preprocess(image_bytes)
        original_size = img_rgb.shape[:2]
        total_px = original_size[0] * original_size[1]

        # Step 1
        logger.info("Step 1: Geometry detection...")
        geom_masks, detections = self.detect_geometry(img_rgb)

        # Step 2
        logger.info("Step 2: Defect detection...")
        wall_defect_masks, element_defect_masks = self.detect_defects(img_rgb, geom_masks)

        # Step 3
        logger.info("Step 3: Material analysis...")
        material_masks = self.analyze_materials(img_rgb, geom_masks, wall_defect_masks)

        # Step 4: Wall layer classification
        logger.info("Step 4: Wall layer classification...")
        layer_analysis = self.classify_wall_layers(wall_defect_masks, material_masks, original_size)

        # Step 5: Visualizations
        logger.info("Step 5: Generating visualizations...")
        viz_paths, layers_info = self.generate_visualizations(
            img_rgb, geom_masks, wall_defect_masks, element_defect_masks, material_masks, result_dir
        )

        # Build silhouette for area calculations
        from rembg import remove
        rgba = remove(img_rgb)
        silhouette = rgba[:, :, 3] > 128
        building_area_px = int(silhouette.sum())

        # Compile damage statistics
        damages = []
        all_defect_masks = {**wall_defect_masks, **element_defect_masks}

        # Build union mask to avoid double-counting overlapping defect pixels
        combined_damage = np.zeros(original_size, dtype=bool)
        for mask in all_defect_masks.values():
            if np.any(mask):
                combined_damage |= mask
        total_damaged_px = int(combined_damage.sum())

        for dtype, mask in all_defect_masks.items():
            if not np.any(mask):
                continue
            area_px = int(mask.sum())
            pct = (area_px / building_area_px * 100) if building_area_px > 0 else 0

            # Severity classification
            if pct > 15:
                severity = "high"
            elif pct > 5:
                severity = "medium"
            else:
                severity = "low"

            layer_info = layer_analysis.get(dtype, {})

            damages.append({
                "type": dtype,
                "type_display": DEFECT_DISPLAY_RU.get(dtype, dtype.replace("_", " ").title()),
                "percentage": round(pct, 1),
                "area_px": area_px,
                "severity": severity,
                "severity_display": SEVERITY_LABELS["ru"][severity],
                "description": DEFECT_DESCRIPTION_RU.get(dtype, ""),
                "affected_layers": layer_info.get("affected_layers", ["finish"]),
                "crack_depth": layer_info.get("crack_depth"),
            })

        damages.sort(key=lambda d: d["percentage"], reverse=True)

        # Material statistics
        materials = []
        bare_wall_px = sum(m.sum() for m in material_masks.values())
        for mat_name, mask in material_masks.items():
            area_px = int(mask.sum())
            if area_px == 0:
                continue
            pct = (area_px / bare_wall_px * 100) if bare_wall_px > 0 else 0
            # Determine material condition based on defect overlap
            defect_on_mat = 0
            for d_mask in all_defect_masks.values():
                if np.any(d_mask):
                    defect_on_mat += int((mask & d_mask).sum())
            defect_ratio = defect_on_mat / area_px if area_px > 0 else 0
            if defect_ratio < 0.05:
                mat_condition = "Хорошее"
            elif defect_ratio < 0.20:
                mat_condition = "Удовлетворительное"
            else:
                mat_condition = "Требует ремонта"

            materials.append({
                "name": mat_name,
                "name_display": MATERIAL_DISPLAY_RU.get(mat_name, mat_name.replace("_", " ").title()),
                "percentage": round(pct, 1),
                "area_px": area_px,
                "condition": mat_condition,
            })
        materials.sort(key=lambda m: m["percentage"], reverse=True)

        # Overall score (0-100): 0% damage → 100 pts, 50%+ damage → 0 pts
        damage_ratio = min(total_damaged_px / building_area_px, 1.0) if building_area_px > 0 else 0
        overall_score = max(0, min(100, round(100 * (1 - damage_ratio * 2), 1)))

        if overall_score >= 80:
            condition = "good"
        elif overall_score >= 60:
            condition = "satisfactory"
        elif overall_score >= 30:
            condition = "needs_repair"
        else:
            condition = "critical"

        return {
            "id": analysis_id,
            "overall_score": overall_score,
            "overall_condition": CONDITION_LABELS["ru"][condition],
            "total_area_px": building_area_px,
            "damaged_area_px": total_damaged_px,   # union mask — no double counting
            "damage_ratio": round(damage_ratio * 100, 1),  # 0-100 %
            "damages": damages,
            "materials": materials,
            "layer_analysis": layer_analysis,
            "geometry_detections": detections,
            "processed_images": list(viz_paths.keys()),
            "image_paths": viz_paths,
            "layers": layers_info,
        }

    # ─────────────────────────────────────────────────
    # HELPER: SAM3.1 text-prompted segmentation
    # ─────────────────────────────────────────────────

    def _sam3_segment(
        self,
        img_rgb: np.ndarray,
        prompts_map: Dict[str, str],
        original_size: Tuple[int, int],
        region_mask: Optional[np.ndarray] = None,
        threshold: float = 0.40,
    ) -> Dict[str, np.ndarray]:
        """SAM3.1 segmentation: one text prompt per class, returns masks directly.

        prompts_map: {class_key: prompt_string} — set_text_prompt() called once per entry.
        SAM3.1 returns all instances of the queried concept as pixel masks + scores.
        No NMS or label-matching needed; class identity comes from the prompt loop.
        """
        image = Image.fromarray(img_rgb)
        result_masks = {k: np.zeros(original_size, dtype=bool) for k in prompts_map}

        try:
            inference_state = self._sam3_processor.set_image(image)

            for class_key, prompt in prompts_map.items():
                output = self._sam3_processor.set_text_prompt(
                    state=inference_state, prompt=prompt
                )
                masks = output.get("masks")
                scores = output.get("scores")
                if masks is None or len(masks) == 0:
                    continue

                for mask, score in zip(masks, scores):
                    if float(score) < threshold:
                        continue
                    m = np.asarray(mask, dtype=bool)
                    if region_mask is not None:
                        m = m & region_mask
                    if np.any(m):
                        result_masks[class_key] |= m
        except Exception:
            import traceback
            logger.error("_sam3_segment failed:\n" + traceback.format_exc())
            raise

        return result_masks
