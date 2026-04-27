# ====================================================================
# ЯЧЕЙКА 2: ОБНАРУЖЕНИЕ ДЕФЕКТОВ (AlegroCode 9.0 - DINO + SAM 2)
# ====================================================================
# Полная миграция с CLIPSeg + SAM3 (broken) → Grounding DINO + SAM 2
# Стена: DINO находит боксы → SAM2 строит маски (аналог prod-бэкенда)
# Элементы: тот же пайплайн, с пространственным роутингом по геометрии
# ====================================================================
!apt-get install aria2 -y -q
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

print("=== 1. ХИРУРГИЧЕСКАЯ ИЗОЛЯЦИЯ ФАСАДА (U-2-Net) ===")
torch.cuda.empty_cache()
gc.collect()

if 'img_norm_rgb' not in locals() or 'geom_masks' not in locals():
    raise ValueError("ОШИБКА: Запустите Ячейку 1, чтобы получить нормализованное фото и маски!")

original_size = img_norm_rgb.shape[:2]

print("-> Генерация идеального силуэта (отсечение фона)...")
rgba_img = remove(img_norm_rgb)
true_silhouette = rgba_img[:, :, 3] > 128

print("-> Создание маски 'Голая стена'...")
elements_mask = np.zeros(original_size, dtype=bool)
for key in ["window", "door", "balcony", "roof", "molding"]:
    if key in geom_masks:
        elements_mask = elements_mask | geom_masks[key]

bare_wall_mask = true_silhouette & ~elements_mask

# Изолированное изображение стены (чёрный фон вне стены)
isolated_bare_wall_img = img_norm_rgb.copy()
isolated_bare_wall_img[~bare_wall_mask] = [0, 0, 0]

print("=== 2. ЗАГРУЗКА МОДЕЛЕЙ ===")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Grounding DINO — переиспользуем из Ячейки 1 если уже в памяти
if 'dino_processor' not in locals() or 'dino_model' not in locals():
    print("-> Загрузка Grounding DINO...")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    ).to(device)
    print("✓ DINO загружен.")

# SAM 2 — скачиваем через aria2 для максимальной скорости
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


# ── УНИВЕРСАЛЬНЫЙ ХЕЛПЕР: DINO → боксы, SAM2 → маски ──────────────────────
def dino_sam2_segment(image_rgb, prompt, class_map,
                      region_mask=None, threshold=0.20, text_threshold=0.20):
    """
    Grounding DINO находит объекты по текстовому промпту (возвращает боксы),
    SAM 2 ImagePredictor строит точные пиксельные маски для каждого бокса.
    region_mask — опциональная булева маска для ограничения зоны поиска.
    """
    image_pil = Image.fromarray(image_rgb)
    h, w = image_rgb.shape[:2]

    # — DINO inference —
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

    # — NMS по категориям —
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

    # — SAM 2: маски по боксам —
    sam2_predictor.set_image(image_rgb)
    for box, label in zip(boxes, labels):
        base_class = get_class(str(label))
        if base_class is None:
            continue
        with torch.no_grad():
            masks, _, _ = sam2_predictor.predict(
                box=box, multimask_output=False
            )
        mask = masks[0].astype(bool)
        if region_mask is not None:
            mask &= region_mask
        if np.any(mask):
            result_masks[base_class] |= mask

    return result_masks


# ── ШАГ 2: ДЕФЕКТЫ СТЕН ────────────────────────────────────────────────────
print("=== 3. СКАНИРОВАНИЕ ДЕФЕКТОВ СТЕН (Grounding DINO + SAM 2) ===")

WALL_DEFECT_PROMPT = (
    "deep structural wall crack fracture. "
    "damaged peeling plaster surface delamination. "
    "exposed red brick from damaged crumbling wall."
)

CLASS_MAP_WALL_DEFECTS = {
    "crack":        ["crack", "fracture"],
    "peeling":      ["peeling", "plaster", "delamination"],
    "exposed_brick": ["exposed red brick", "brick", "crumbling"],
}

COLORS_WALL_DEFECTS = {
    "crack":        [255,   0,   0],
    "peeling":      [241, 196,  15],
    "exposed_brick":[230, 126,  34],
}

print("-> Тотальное сканирование патологий стен (изолированная стена)...")
final_wall_defect_masks = dino_sam2_segment(
    isolated_bare_wall_img,
    WALL_DEFECT_PROMPT,
    CLASS_MAP_WALL_DEFECTS,
    region_mask=bare_wall_mask,
    threshold=0.18,
)

found_wall = [k for k, m in final_wall_defect_masks.items() if np.any(m)]
print(f"✓ Найдено дефектов стен: {found_wall}")


# ── ШАГ 3: ДЕФЕКТЫ ЭЛЕМЕНТОВ ───────────────────────────────────────────────
print("=== 4. СКАНИРОВАНИЕ ДЕФЕКТОВ ЭЛЕМЕНТОВ (Grounding DINO + SAM 2) ===")

ELEMENT_DEFECT_PROMPT = (
    "broken shattered cracked window glass pane. "
    "damaged rotten wooden door frame surface."
)

CLASS_MAP_ELEMENT_DEFECTS = {
    "broken_glass": ["broken", "shattered", "cracked window", "glass pane"],
    "damaged_wood": ["damaged", "rotten", "wooden door", "door frame"],
}

COLORS_ELEMENT_DEFECTS = {
    "broken_glass": [  0, 255, 255],
    "damaged_wood": [255,  20, 147],
}

print("-> Сканирование дефектов остекления и дерева...")
final_element_defect_masks = dino_sam2_segment(
    img_norm_rgb,
    ELEMENT_DEFECT_PROMPT,
    CLASS_MAP_ELEMENT_DEFECTS,
    threshold=0.18,
)

# Пространственный роутинг: дефект элемента только внутри своей геометрии
mask_windows = geom_masks.get("window", np.zeros(original_size, dtype=bool))
mask_doors   = geom_masks.get("door",   np.zeros(original_size, dtype=bool))
final_element_defect_masks["broken_glass"] &= mask_windows
final_element_defect_masks["damaged_wood"] &= mask_doors

found_elem = [k for k, m in final_element_defect_masks.items() if np.any(m)]
print(f"✓ Найдено дефектов элементов: {found_elem}")

# Освобождаем GPU-память
del sam2_base, sam2_predictor
torch.cuda.empty_cache()
gc.collect()


# ── ШАГ 4: ВИЗУАЛИЗАЦИЯ И ОТЧЁТ ────────────────────────────────────────────
print("=== 5. СБОРКА КОМПЛЕКСНОЙ КАРТЫ ПАТОЛОГИЙ ===")

mask_only_canvas = np.zeros((*original_size, 3), dtype=np.uint8)
found_defects = []
all_colors = {**COLORS_WALL_DEFECTS, **COLORS_ELEMENT_DEFECTS}

draw_order = ["peeling", "exposed_brick", "crack", "damaged_wood", "broken_glass"]
for defect_type in draw_order:
    if defect_type in final_wall_defect_masks:
        mask = final_wall_defect_masks[defect_type]
    else:
        mask = final_element_defect_masks.get(defect_type, None)
    if mask is not None and np.any(mask):
        found_defects.append(defect_type)
        mask_only_canvas[mask] = np.array(all_colors[defect_type], dtype=np.uint8)

# Контекст (окна/двери без дефектов)
context_mask = np.zeros(original_size, dtype=bool)
for key, gm in geom_masks.items():
    if key != "building":
        context_mask = context_mask | gm
all_found_px = np.any(
    list(final_wall_defect_masks.values()) + list(final_element_defect_masks.values()),
    axis=0
)
context_only = context_mask & ~all_found_px & true_silhouette
mask_only_canvas[context_only] = [30, 30, 30]

overlay_canvas = cv2.addWeighted(img_norm_rgb.copy(), 0.3, mask_only_canvas, 0.7, 0)

fig, axes = plt.subplots(1, 2, figsize=(24, 12))
axes[0].imshow(overlay_canvas)
axes[0].set_title("1. Наложение Комплексной Карты Патологий", fontsize=16, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(mask_only_canvas)
axes[1].set_title("2. Физика дефектов (DINO + SAM 2)", fontsize=16, fontweight='bold')
axes[1].axis('off')

legend_elements = [
    Patch(facecolor=np.array(all_colors[c]) / 255, label=c.replace('_', ' ').title())
    for c in reversed(found_defects)
]
if np.any(context_only):
    legend_elements.append(Patch(facecolor=[30/255, 30/255, 30/255], label="Elements Context (DINO)"))
axes[1].legend(handles=legend_elements, loc='upper right', fontsize=12,
               facecolor='black', labelcolor='white')

plt.tight_layout()
plt.show()

# Отчёт
building_area_px = true_silhouette.sum()
print("\n=== 🛠️ СТРОИТЕЛЬНЫЙ ОТЧЕТ AlegroCode (Damage Report) ===")
for name, mask in {**final_wall_defect_masks, **final_element_defect_masks}.items():
    area = mask.sum()
    if area > 0:
        pct = (area / building_area_px) * 100
        print(f"  • {name.replace('_', ' ').title()}: {area} px ({pct:.1f}% здания)")

print("\n✅ Ячейка 2 завершена. Данные готовы для Ячейки 3 (материалы)!")
print("   Экспортированы: final_wall_defect_masks, final_element_defect_masks, bare_wall_mask, true_silhouette")
