# ====================================================================
# ЯЧЕЙКА 2: ОБНАРУЖЕНИЕ ДЕФЕКТОВ (SAM3.1)
# ====================================================================
# Миграция с Grounding DINO + SAM 2 → SAM3.1
# SAM3.1 объединяет распознавание по тексту и сегментацию в одной модели.
# Стена: sam3_segment находит дефекты по текстовым промптам → маски
# Элементы: тот же пайплайн с пространственным роутингом по геометрии
# ====================================================================
# УСТАНОВКА (один раз в сессии Colab):
# !git clone https://github.com/facebookresearch/sam3.git -q
# !pip install -e ./sam3 -q
# !pip install -U "rembg[cpu]" -q

import cv2
import torch
import gc
import numpy as np
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

isolated_bare_wall_img = img_norm_rgb.copy()
isolated_bare_wall_img[~bare_wall_mask] = [0, 0, 0]

print("=== 2. ЗАГРУЗКА SAM3.1 ===")
device = "cuda" if torch.cuda.is_available() else "cpu"

if 'sam3_processor' not in locals() or 'sam3_model' not in locals():
    print("-> Загрузка SAM3.1...")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    sam3_model = build_sam3_image_model()
    sam3_processor = Sam3Processor(sam3_model)
    print("✓ SAM3.1 загружен.")

SAM3_THRESHOLD = 0.40


# ── УНИВЕРСАЛЬНЫЙ ХЕЛПЕР: SAM3.1 → маски по текстовым промптам ────────────
def sam3_segment(image_rgb, prompts_map, region_mask=None, threshold=SAM3_THRESHOLD):
    """
    SAM3.1 сегментирует объекты по текстовым промптам за один проход.
    prompts_map: {class_key: prompt_string}
    Возвращает {class_key: boolean mask}.
    """
    image_pil = Image.fromarray(image_rgb)
    h, w = image_rgb.shape[:2]
    result_masks = {k: np.zeros((h, w), dtype=bool) for k in prompts_map}

    inference_state = sam3_processor.set_image(image_pil)

    for class_key, prompt in prompts_map.items():
        output = sam3_processor.set_text_prompt(state=inference_state, prompt=prompt)
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

    return result_masks


# ── ШАГ 2: ДЕФЕКТЫ СТЕН ────────────────────────────────────────────────────
print("=== 3. СКАНИРОВАНИЕ ДЕФЕКТОВ СТЕН (SAM3.1) ===")

WALL_DEFECT_PROMPTS_MAP = {
    "crack":         "wall crack facade crack concrete crack surface fracture hairline crack",
    "peeling":       "peeling plaster flaking plaster flaking paint delaminated coating blistered render",
    "exposed_brick": "exposed brick bare brick missing plaster unplastered brick masonry",
}

COLORS_WALL_DEFECTS = {
    "crack":         [255,   0,   0],
    "peeling":       [241, 196,  15],
    "exposed_brick": [230, 126,  34],
}

print("-> Тотальное сканирование патологий стен (изолированная стена)...")
final_wall_defect_masks = sam3_segment(
    isolated_bare_wall_img,
    WALL_DEFECT_PROMPTS_MAP,
    region_mask=bare_wall_mask,
)

found_wall = [k for k, m in final_wall_defect_masks.items() if np.any(m)]
print(f"✓ Найдено дефектов стен: {found_wall}")


# ── ШАГ 3: ДЕФЕКТЫ ЭЛЕМЕНТОВ ───────────────────────────────────────────────
print("=== 4. СКАНИРОВАНИЕ ДЕФЕКТОВ ЭЛЕМЕНТОВ (SAM3.1) ===")

ELEMENT_DEFECT_PROMPTS_MAP = {
    "broken_glass": "broken window glass shattered glass cracked glass pane damaged glazing",
    "damaged_wood": "damaged wooden door rotting wood deteriorated wood decay",
}

COLORS_ELEMENT_DEFECTS = {
    "broken_glass": [  0, 255, 255],
    "damaged_wood": [255,  20, 147],
}

print("-> Сканирование дефектов остекления и дерева...")
final_element_defect_masks = sam3_segment(
    img_norm_rgb,
    ELEMENT_DEFECT_PROMPTS_MAP,
)

# Пространственный роутинг: дефект элемента только внутри своей геометрии
mask_windows = geom_masks.get("window", np.zeros(original_size, dtype=bool))
mask_doors   = geom_masks.get("door",   np.zeros(original_size, dtype=bool))
final_element_defect_masks["broken_glass"] &= mask_windows
final_element_defect_masks["damaged_wood"] &= mask_doors

found_elem = [k for k, m in final_element_defect_masks.items() if np.any(m)]
print(f"✓ Найдено дефектов элементов: {found_elem}")

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
axes[1].set_title("2. Физика дефектов (SAM3.1)", fontsize=16, fontweight='bold')
axes[1].axis('off')

legend_elements = [
    Patch(facecolor=np.array(all_colors[c]) / 255, label=c.replace('_', ' ').title())
    for c in reversed(found_defects)
]
if np.any(context_only):
    legend_elements.append(Patch(facecolor=[30/255, 30/255, 30/255], label="Elements Context (SAM3.1)"))
axes[1].legend(handles=legend_elements, loc='upper right', fontsize=12,
               facecolor='black', labelcolor='white')

plt.tight_layout()
plt.show()

building_area_px = true_silhouette.sum()
print("\n=== СТРОИТЕЛЬНЫЙ ОТЧЕТ (Damage Report) ===")
for name, mask in {**final_wall_defect_masks, **final_element_defect_masks}.items():
    area = mask.sum()
    if area > 0:
        pct = (area / building_area_px) * 100
        print(f"  • {name.replace('_', ' ').title()}: {area} px ({pct:.1f}% здания)")

print("\n✅ Ячейка 2 завершена. Данные готовы для Ячейки 3 (материалы)!")
print("   Экспортированы: final_wall_defect_masks, final_element_defect_masks, bare_wall_mask, true_silhouette")
