"""
FacadeAnalyzer — ML Pipeline for building facade analysis.

Step 1: Grounding DINO + SAM2 → geometry detection (windows, doors, balconies)
Step 2: rembg + Grounding DINO + SAM2 → wall and element defect detection
Step 3: CLIPSeg + SAM2 AMG → material identification (texture fusion approach)
Step 4: Wall layer classification based on defect and material overlap

Material detection uses the notebook-proven CLIPSeg+AMG fusion:
  CLIPSeg produces per-pixel probability heatmaps for each material/texture.
  SAM2 AutoMaskGenerator cuts the image into physically coherent segments.
  Each segment is assigned to the material with the highest average CLIPSeg score.
  This outperforms DINO (object detector) for distributed surface textures.
"""

import cv2
import torch
import gc
import os
import uuid
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import torchvision.ops as ops
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

GEOMETRY_PROMPT = (
    "window. door. wall. balcony. roof. molding. pipe. air conditioner."
)

CLASS_MAP_GEOMETRY = {
    "window": ["window", "pane", "glass"],
    "door": ["door", "entrance", "gate"],
    "balcony": ["balcony", "terrace"],
    "building": ["building", "facade", "wall"],
    "roof": ["roof", "rooftop"],
    "molding": ["cornice", "molding", "decoration"],
    "pipe": ["drainpipe", "pipe"],
    "ac_unit": ["air conditioner", "unit"],
}

WALL_DEFECT_PROMPTS = [
    "wall crack",
    "peeling plaster",
    "exposed brick",
    "water stain",
    "rust stain",
    "moss",
    "efflorescence",
    "spalling concrete",
]

CLASS_MAP_WALL_DEFECTS = {
    "crack": ["crack"],
    "peeling": ["peeling", "plaster"],
    "exposed_brick": ["exposed brick", "brick"],
    "water_damage": ["water stain", "stain"],
    "rust": ["rust"],
    "moss": ["moss"],
    "efflorescence": ["efflorescence"],
    "spalling": ["spalling", "concrete"],
}

ELEMENT_DEFECT_PROMPTS = [
    "broken window glass",
    "damaged wooden door",
    "rusty metal railing",
    "cracked balcony railing",
]

CLASS_MAP_ELEMENT_DEFECTS = {
    "broken_glass": ["broken", "glass"],
    "damaged_wood": ["damaged", "wooden", "door"],
    "rusty_metal": ["rusty", "metal"],
    "damaged_railing": ["cracked", "railing"],
}

# CLIPSeg prompts for material detection (CLIPSeg+AMG fusion approach)
# Index order matches MATERIAL_CLIPSEG_IDX_MAP below.
MATERIAL_PROMPTS_CLIPSEG = [
    "painted plaster smooth wall facade texture",   # 0
    "exposed bare red brick masonry wall texture",  # 1
    "grey rough concrete stone surface texture",    # 2
    "decorative white plaster molding cornice",     # 3
    "glazed ceramic tile surface texture",          # 4
]
MATERIAL_CLIPSEG_IDX_MAP = {
    0: "cement_plaster",
    1: "brick",
    2: "concrete",
    3: "molding",
    4: "ceramic_tile",
}
MATERIAL_CLIPSEG_CONFIDENCE = 0.28   # min CLIPSeg score to assign a material

# Legacy DINO prompts kept for reference (no longer used for materials)
MATERIAL_PROMPTS = MATERIAL_PROMPTS_CLIPSEG

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


def _download_sam2_weights(dest_path: str) -> None:
    """Download SAM2 weights using aria2c (16 threads) or urllib fallback."""
    import subprocess, urllib.request
    url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        if subprocess.run(["which", "aria2c"], capture_output=True).returncode == 0:
            logger.info("Downloading SAM2 weights via aria2c (16 threads)…")
            subprocess.run(
                ["aria2c", "-x", "16", "-s", "16", "-k", "1M", url, "-o", dest_path],
                check=True
            )
        else:
            logger.info("Downloading SAM2 weights via urllib (single thread)…")
            urllib.request.urlretrieve(url, dest_path)
        logger.info("SAM2 weights downloaded.")
    except Exception as e:
        raise RuntimeError(f"Failed to download SAM2 weights: {e}") from e


def _adaptive_params(image_shape: Tuple[int, int]) -> dict:
    """Return adaptive ML parameters based on image resolution."""
    h, w = image_shape[:2]
    total_px = h * w
    if total_px > 2_000_000:  # >2MP
        return {"points_per_side": 48, "min_mask_area": 500, "dino_threshold": 0.18, "dino_text_threshold": 0.18}
    elif total_px > 800_000:  # >0.8MP
        return {"points_per_side": 32, "min_mask_area": 300, "dino_threshold": 0.18, "dino_text_threshold": 0.18}
    else:
        return {"points_per_side": 16, "min_mask_area": 150, "dino_threshold": 0.18, "dino_text_threshold": 0.18}


class FacadeAnalyzer:
    """
    Production ML pipeline for building facade analysis.
    Loads models once and caches them in memory.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_loaded = False
        self._dino_processor = None
        self._dino_model = None
        self._sam2_predictor = None
        self._sam2_amg = None
        self._clipseg_processor = None
        self._clipseg_model = None

    def load_models(self):
        """Load and cache all ML models. Call once at startup."""
        if self.models_loaded:
            return

        logger.info("Loading Grounding DINO...")
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self._dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self._dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        ).to(self.device)

        logger.info("Loading SAM2 (hiera_small)...")
        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2_hiera_small.pt")
        if not os.path.exists(weights_path):
            _download_sam2_weights(weights_path)

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        sam2_base = build_sam2("sam2_hiera_s.yaml", weights_path, device=self.device)
        self._sam2_predictor = SAM2ImagePredictor(sam2_base)
        self._sam2_amg = SAM2AutomaticMaskGenerator(
            model=sam2_base,
            points_per_side=32,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.80,
            min_mask_region_area=300,
        )

        logger.info("Loading CLIPSeg (material texture heatmaps)...")
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        self._clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self._clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        ).to(self.device)

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
    # STEP 1: Geometry Detection (DINO + SAM)
    # ─────────────────────────────────────────────────

    def detect_geometry(self, img_rgb: np.ndarray) -> Tuple[Dict[str, np.ndarray], list]:
        """Detect architectural elements using Grounding DINO + SAM."""
        image = Image.fromarray(img_rgb)
        original_size = image.size[::-1]  # (H, W)
        params = _adaptive_params(img_rgb.shape)

        # DINO detection
        dino_inputs = self._dino_processor(
            images=image, text=GEOMETRY_PROMPT, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            dino_outputs = self._dino_model(**dino_inputs)

        results = self._dino_processor.post_process_grounded_object_detection(
            dino_outputs, dino_inputs.input_ids,
            threshold=params["dino_threshold"],
            text_threshold=params["dino_text_threshold"],
            target_sizes=[original_size]
        )[0]

        all_scores = results["scores"].cpu()
        valid = all_scores > params["dino_threshold"]
        raw_boxes = results["boxes"].cpu()[valid]
        raw_scores = all_scores[valid]
        labels_out = results.get("text_labels", results.get("labels"))
        raw_labels = [labels_out[i] for i, v in enumerate(valid) if v]

        # NMS per category
        geom_masks = {k: np.zeros(original_size, dtype=bool) for k in CLASS_MAP_GEOMETRY}
        detections = []

        if len(raw_boxes) > 0:
            class_keys = list(CLASS_MAP_GEOMETRY.keys())
            category_idxs = torch.tensor([
                class_keys.index(_get_base_class(l, CLASS_MAP_GEOMETRY))
                if _get_base_class(l, CLASS_MAP_GEOMETRY) in class_keys else 99
                for l in raw_labels
            ])
            keep = ops.batched_nms(raw_boxes, raw_scores, category_idxs, iou_threshold=0.5).numpy()
            boxes = raw_boxes[keep].numpy()
            scores = raw_scores[keep].numpy()
            labels = [raw_labels[i] for i in keep]

            # SAM2 segmentation — one box at a time for tight, precise masks
            if len(boxes) > 0:
                self._sam2_predictor.set_image(img_rgb)
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                sorted_indices = np.argsort(areas)[::-1]

                for i in sorted_indices:
                    base_class = _get_base_class(str(labels[i]), CLASS_MAP_GEOMETRY) or "unknown"
                    with torch.no_grad():
                        pred_masks, _, _ = self._sam2_predictor.predict(
                            box=boxes[i], multimask_output=False
                        )
                    mask = pred_masks[0].astype(bool)
                    if base_class in geom_masks:
                        geom_masks[base_class] |= mask
                    detections.append({
                        "class": base_class,
                        "score": float(scores[i]),
                        "box": boxes[i].tolist(),
                    })

        logger.info(f"Geometry: detected {len(detections)} elements")
        return geom_masks, detections

    # ─────────────────────────────────────────────────
    # STEP 2: Wall Defect Detection
    # ─────────────────────────────────────────────────

    def detect_defects(
        self, img_rgb: np.ndarray, geom_masks: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Detect wall and element defects using rembg + Grounding DINO + SAM."""
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

        # 3. Wall defects via Grounding DINO + SAM
        # Use silhouette (not bare_wall) as region mask: SAM window masks expand
        # well beyond actual glass, leaving bare_wall nearly empty and filtering
        # out real cracks/peeling. Text prompts provide semantic specificity.
        logger.info("Scanning wall defects (DINO + SAM)...")
        wall_defect_prompt = (
            "wall crack. peeling plaster. exposed brick. "
            "water stain. rust stain. moss. efflorescence. spalling concrete."
        )
        wall_defect_masks = self._dino_sam_segment(
            img_rgb, wall_defect_prompt, CLASS_MAP_WALL_DEFECTS,
            original_size, region_mask=silhouette,
            threshold=params["dino_threshold"],
            text_threshold=params["dino_text_threshold"],
        )

        # Sanity cap for wall defects: a single type covering >70% of silhouette is a false positive
        building_px = int(silhouette.sum()) or 1
        for key in list(wall_defect_masks.keys()):
            coverage = wall_defect_masks[key].sum() / building_px
            if coverage > 0.70:
                logger.warning(f"Wall defect '{key}' covers {coverage:.1%} — zeroing (false positive)")
                wall_defect_masks[key] = np.zeros(original_size, dtype=bool)

        # 4. Element defects via DINO + SAM
        logger.info("Scanning element defects (DINO + SAM)...")
        element_defect_prompt = (
            "broken window glass. damaged wooden door. "
            "rusty metal railing. cracked balcony railing."
        )
        element_defect_masks = self._dino_sam_segment(
            img_rgb, element_defect_prompt, CLASS_MAP_ELEMENT_DEFECTS,
            original_size,
            threshold=params["dino_threshold"],
            text_threshold=params["dino_text_threshold"],
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
        """
        Identify facade materials using CLIPSeg heatmaps + SAM2 AMG fusion.

        CLIPSeg produces per-pixel probability maps for each material texture.
        SAM2 AMG cuts the image into physically coherent segments.
        Each segment is assigned to the material with the highest average score.
        Unclassified silhouette pixels default to cement_plaster.
        """
        from rembg import remove
        original_size = img_rgb.shape[:2]

        # Rebuild silhouette
        rgba = remove(img_rgb)
        silhouette = rgba[:, :, 3] > 128

        # ── 1. CLIPSeg heatmaps ───────────────────────────────────────────────
        logger.info("CLIPSeg material heatmaps…")
        clipseg_probs = self._get_clipseg_probs(img_rgb, MATERIAL_PROMPTS_CLIPSEG, original_size)

        # ── 2. SAM2 AMG — physical segments ───────────────────────────────────
        logger.info("SAM2 AMG auto-segmentation…")
        with torch.no_grad():
            amg_masks = self._sam2_amg.generate(img_rgb)
        logger.info(f"AMG produced {len(amg_masks)} segments")

        # ── 3. Fusion: assign each segment to its dominant material ───────────
        final_materials: Dict[str, np.ndarray] = {
            k: np.zeros(original_size, dtype=bool) for k in CLASS_MAP_MATERIALS
        }
        classified = np.zeros(original_size, dtype=bool)

        for ann in amg_masks:
            seg: np.ndarray = ann["segmentation"]

            # Skip segments fully outside the building silhouette
            if not np.any(seg & silhouette):
                continue

            valid = seg & silhouette
            scores = [float(clipseg_probs[i][seg].mean()) for i in range(len(MATERIAL_PROMPTS_CLIPSEG))]
            best_idx = int(np.argmax(scores))

            if scores[best_idx] >= MATERIAL_CLIPSEG_CONFIDENCE:
                mat_key = MATERIAL_CLIPSEG_IDX_MAP.get(best_idx)
                if mat_key and mat_key in final_materials:
                    final_materials[mat_key] |= valid
                    classified |= valid

        # ── 4. Default: unclassified silhouette pixels → cement_plaster ───────
        final_materials["cement_plaster"] |= (silhouette & ~classified)

        # ── 5. Augment brick from exposed_brick defect mask ───────────────────
        if wall_defect_masks:
            exposed = wall_defect_masks.get("exposed_brick", np.zeros(original_size, dtype=bool))
            if np.any(exposed):
                # Move exposed areas from plaster → brick
                final_materials["cement_plaster"] &= ~exposed
                final_materials["brick"] |= (exposed & silhouette)

        torch.cuda.empty_cache()
        gc.collect()

        logger.info("Material analysis complete (CLIPSeg+AMG).")
        return final_materials

    def _get_clipseg_probs(
        self, img_rgb: np.ndarray, prompts: List[str], original_size: Tuple[int, int]
    ) -> np.ndarray:
        """Run CLIPSeg and return per-prompt probability maps, shape (N, H, W)."""
        image = Image.fromarray(img_rgb)
        inputs = self._clipseg_processor(
            text=prompts,
            images=[image] * len(prompts),
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self._clipseg_model(**inputs)
        probs = torch.nn.functional.interpolate(
            outputs.logits.unsqueeze(1),
            size=original_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return torch.sigmoid(probs).cpu().numpy()  # (N, H, W)

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
    # HELPER: Grounding DINO + SAM segmentation
    # ─────────────────────────────────────────────────

    def _dino_sam_segment(
        self,
        img_rgb: np.ndarray,
        prompt: str,
        class_map: dict,
        original_size: Tuple[int, int],
        region_mask: Optional[np.ndarray] = None,
        threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> Dict[str, np.ndarray]:
        """Generic Grounding DINO + SAM segmentation with optional region masking."""
        image = Image.fromarray(img_rgb)

        dino_inputs = self._dino_processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            dino_outputs = self._dino_model(**dino_inputs)

        results = self._dino_processor.post_process_grounded_object_detection(
            dino_outputs, dino_inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[original_size]
        )[0]

        all_scores = results["scores"].cpu()
        valid = all_scores > threshold
        raw_boxes = results["boxes"].cpu()[valid]
        raw_scores = all_scores[valid]
        labels_out = results.get("text_labels", results.get("labels"))
        raw_labels = [labels_out[i] for i, v in enumerate(valid) if v]

        result_masks = {k: np.zeros(original_size, dtype=bool) for k in class_map}

        if len(raw_boxes) == 0:
            return result_masks

        # NMS across categories
        class_keys = list(class_map.keys())
        category_idxs = torch.tensor([
            class_keys.index(_get_base_class(l, class_map))
            if _get_base_class(l, class_map) in class_keys else 99
            for l in raw_labels
        ])
        keep = ops.batched_nms(raw_boxes, raw_scores, category_idxs, iou_threshold=0.5).numpy()
        boxes = raw_boxes[keep].numpy()
        labels = [raw_labels[i] for i in keep]

        # SAM2 segmentation — per-box for tight masks that stay inside detected regions
        self._sam2_predictor.set_image(img_rgb)
        for box, label in zip(boxes, labels):
            base_class = _get_base_class(str(label), class_map)
            if base_class is None:
                continue
            with torch.no_grad():
                pred_masks, _, _ = self._sam2_predictor.predict(
                    box=box, multimask_output=False
                )
            final_mask = pred_masks[0].astype(bool)
            if region_mask is not None:
                final_mask &= region_mask
            if np.any(final_mask):
                result_masks[base_class] |= final_mask

        return result_masks
