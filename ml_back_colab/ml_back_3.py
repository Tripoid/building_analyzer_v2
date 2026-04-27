# ====================================================================
# ЯЧЕЙКА 3: Z-INDEX АНАЛИЗ МАТЕРИАЛОВ ФАСАДА (AlegroCode 9.0)
# ====================================================================
# Миграция с CLIPSeg + SAM2 AMG → Grounding DINO + SAM2 ImagePredictor
# Используем тот же пайплайн что и prod-бэкенд (ml_pipeline.py):
#   DINO находит регионы по текстовому промпту (бетон, кирпич, штукатурка...)
#   SAM 2 строит точные пиксельные маски для каждого найденного региона
#   Z-index сборка: структурный → базовый → финишный → декоративный
# ====================================================================
!pip install transformers accelerate torchvision git+https://github.com/facebookresearch/sam2.git -q
!pip install -U "rembg[cpu]" -q

import cv2
import torch
import gc
import os
import numpy as np
import torchvision.ops as ops
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Patch
from rembg import remove

print("=== 1. ПОДГОТОВКА ИЗОЛИРОВАННОЙ СТЕНЫ ===")
torch.cuda.empty_cache()
gc.collect()

if 'img_norm_rgb' not in locals() or 'geom_masks' not in locals():
    raise ValueError("ОШИБКА: Запустите Ячейку 1 для получения базовых данных!")

original_size = img_norm_rgb.shape[:2]

# Восстанавливаем силуэт если запускаем ячейку отдельно
if 'true_silhouette' not in locals():
    print("-> Восстановление силуэта (U-2-Net)...")
    rgba_img = remove(img_norm_rgb)
    true_silhouette = rgba_img[:, :, 3] > 128

elements_mask = np.zeros(original_size, dtype=bool)
for key in ["window", "door", "balcony", "roof", "molding"]:
    if key in geom_masks:
        elements_mask = elements_mask | geom_masks[key]
bare_wall_mask = true_silhouette & ~elements_mask


print("=== 2. ЗАГРУЗКА МОДЕЛЕЙ ===")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Grounding DINO — переиспользуем из Ячейки 1 если уже загружен
if 'dino_processor' not in locals() or 'dino_model' not in locals():
    print("-> Загрузка Grounding DINO...")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    ).to(device)
    print("✓ DINO загружен.")

# SAM 2
weights_path_sam2 = "sam2_hiera_small.pt"
if not os.path.exists(weights_path_sam2):
    print("-> Загрузка SAM 2 (aria2c, 16 потоков)...")
    !aria2c -x 16 -s 16 -k 1M \
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt" \
        -o sam2_hiera_small.pt
    print("✓ SAM 2 скачан.")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_base = build_sam2("sam2_hiera_s.yaml", weights_path_sam2, device=device)
sam2_predictor = SAM2ImagePredictor(sam2_base)
print("✓ SAM 2 загружен.")


# ── ХЕЛПЕР: DINO → боксы, SAM2 → маски ────────────────────────────────────
def dino_sam2_segment(image_rgb, prompt, class_map,
                      region_mask=None, threshold=0.20, text_threshold=0.20):
    image_pil = Image.fromarray(image_rgb)
    h, w = image_rgb.shape[:2]

    dino_inputs = dino_processor(
        images=image_pil, text=prompt, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        dino_outputs = dino_model(**dino_inputs)

    results = dino_processor.post_process_grounded_object_detection(
        dino_outputs, dino_inputs.input_ids,
        threshold=threshold, text_threshold=text_threshold,
        target_sizes=[(h, w)]
    )[0]

    scores = results["scores"].cpu()
    valid = scores > threshold
    raw_boxes = results["boxes"].cpu()[valid]
    raw_scores = scores[valid]
    labels_out = results.get("text_labels", results.get("labels"))
    raw_labels = [labels_out[i] for i, v in enumerate(valid) if v]

    result_masks = {k: np.zeros((h, w), dtype=bool) for k in class_map}
    if len(raw_boxes) == 0:
        return result_masks

    class_keys = list(class_map.keys())

    def get_class(lbl):
        lbl_l = lbl.lower()
        for base, syns in class_map.items():
            if any(s in lbl_l for s in syns):
                return base
        return None

    cat_idxs = torch.tensor([
        class_keys.index(get_class(l)) if get_class(l) in class_keys else 99
        for l in raw_labels
    ])
    keep = ops.batched_nms(raw_boxes, raw_scores, cat_idxs, iou_threshold=0.5).numpy()
    boxes = raw_boxes[keep].numpy()
    labels = [raw_labels[i] for i in keep]

    sam2_predictor.set_image(image_rgb)
    for box, label in zip(boxes, labels):
        base_class = get_class(str(label))
        if base_class is None:
            continue
        with torch.no_grad():
            masks, _, _ = sam2_predictor.predict(box=box, multimask_output=False)
        mask = masks[0].astype(bool)
        if region_mask is not None:
            mask &= region_mask
        if np.any(mask):
            result_masks[base_class] |= mask

    return result_masks


# ── ШАГ 2: СКАНИРОВАНИЕ МАТЕРИАЛОВ (DINO + SAM2) ──────────────────────────
print("=== 3. СКАНИРОВАНИЕ МАТЕРИАЛОВ (Grounding DINO + SAM 2) ===")

MATERIAL_PROMPT = (
    "grey concrete stone wall base foundation. "
    "terracotta red brick masonry wall surface. "
    "plain smooth cement plaster wall facade. "
    "architectural molding cornice decoration ornament."
)

CLASS_MAP_MATERIALS = {
    "concrete": ["grey concrete stone", "concrete"],
    "brick":    ["terracotta red brick", "brick masonry"],
    "plaster":  ["plain smooth plaster", "cement plaster", "smooth wall"],
    "molding":  ["molding", "cornice", "decoration", "ornament"],
}

# Архитектурная палитра (пастельные тона)
COLORS_MATERIALS = {
    "concrete": [127, 140, 141],  # Серый бетон
    "brick":    [211,  84,   0],  # Терракотовый кирпич
    "plaster":  [243, 214, 115],  # Песочно-жёлтая штукатурка
    "molding":  [236, 240, 241],  # Светло-серая лепнина
}

print("-> Сканирование текстур фасада...")
raw_material_masks = dino_sam2_segment(
    img_norm_rgb,
    MATERIAL_PROMPT,
    CLASS_MAP_MATERIALS,
    region_mask=bare_wall_mask,
    threshold=0.18,
)

# Усиление кирпича: добавляем маску оголённой кладки из Ячейки 2 (если доступна)
if 'final_wall_defect_masks' in locals():
    exposed = final_wall_defect_masks.get("exposed_brick", np.zeros(original_size, dtype=bool))
    raw_material_masks["brick"] |= (exposed & bare_wall_mask)

del sam2_base, sam2_predictor
torch.cuda.empty_cache()
gc.collect()
print("✓ Сканирование материалов завершено.")


# ── ШАГ 3: УЛЬТИМАТИВНАЯ Z-INDEX СБОРКА ────────────────────────────────────
print("=== 4. ИЕРАРХИЧЕСКАЯ Z-INDEX СБОРКА BIM-КАРТЫ ===")

mask_only_canvas = np.zeros((*original_size, 3), dtype=np.uint8)
final_masks_materials = {k: np.zeros(original_size, dtype=bool) for k in CLASS_MAP_MATERIALS}
found_materials = []

# Контекст (окна/двери)
context_mask = elements_mask & true_silhouette
mask_only_canvas[context_mask] = [40, 40, 40]

# Слой 1: Бетон (цоколь / структурный)
concrete_m = raw_material_masks["concrete"] & bare_wall_mask
if np.any(concrete_m):
    found_materials.append("concrete")
    final_masks_materials["concrete"] |= concrete_m
    mask_only_canvas[concrete_m] = np.array(COLORS_MATERIALS["concrete"], dtype=np.uint8)

# Слой 2: Штукатурка (заполнение всей голой стены кроме бетона)
plaster_m = bare_wall_mask & ~final_masks_materials["concrete"]
if np.any(plaster_m):
    found_materials.append("plaster")
    final_masks_materials["plaster"] |= plaster_m
    mask_only_canvas[plaster_m] = np.array(COLORS_MATERIALS["plaster"], dtype=np.uint8)

# Слой 3: Кирпич (оголённый) — вытесняет штукатурку
brick_m = raw_material_masks["brick"] & plaster_m
if np.any(brick_m):
    found_materials.append("brick")
    final_masks_materials["brick"] |= brick_m
    mask_only_canvas[brick_m] = np.array(COLORS_MATERIALS["brick"], dtype=np.uint8)

# Слой 4: Лепнина — поверх всего (наивысший z-index)
molding_m = raw_material_masks["molding"] & (plaster_m | brick_m)
if np.any(molding_m):
    found_materials.append("molding")
    final_masks_materials["molding"] |= molding_m
    mask_only_canvas[molding_m] = np.array(COLORS_MATERIALS["molding"], dtype=np.uint8)


# ── ВИЗУАЛИЗАЦИЯ ────────────────────────────────────────────────────────────
overlay_canvas = cv2.addWeighted(img_norm_rgb.copy(), 0.3, mask_only_canvas, 0.7, 0)

fig, axes = plt.subplots(1, 2, figsize=(24, 12))
axes[0].imshow(overlay_canvas)
axes[0].set_title("1. Наложение Карты Материалов (Z-Index Stack)", fontsize=16, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(mask_only_canvas)
axes[1].set_title("2. BIM-Карта Материалов (DINO + SAM 2)", fontsize=16, fontweight='bold')
axes[1].axis('off')

legend_elements = [
    Patch(facecolor=np.array(COLORS_MATERIALS[c]) / 255, label=c.title())
    for c in reversed(found_materials)
]
if np.any(context_mask):
    legend_elements.append(Patch(facecolor=[40/255, 40/255, 40/255], label="Elements (Windows/Doors)"))
axes[1].legend(handles=legend_elements, loc='upper right', fontsize=12,
               facecolor='black', labelcolor='white')

plt.tight_layout()
plt.show()


# ── ОТЧЁТ ПО ПЛОЩАДЯМ ───────────────────────────────────────────────────────
print("\n=== 🧱 ВЕДОМОСТЬ ОТДЕЛКИ ФАСАДА (AlegroCode Material Report) ===")
total_wall_area = bare_wall_mask.sum()
for mat_name in found_materials:
    mat_area = final_masks_materials[mat_name].sum()
    percent = (mat_area / total_wall_area) * 100 if total_wall_area > 0 else 0
    print(f"  • {mat_name.title()}: {mat_area} px ({percent:.1f}% от площади стены)")

print("\n✅ Ультимативный Z-Index анализ строительных материалов завершён!")
print("   Экспортированы: final_masks_materials")
