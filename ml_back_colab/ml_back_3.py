# ====================================================================
# ЯЧЕЙКА 3: Z-INDEX АНАЛИЗ МАТЕРИАЛОВ ФАСАДА (SAM3.1)
# ====================================================================
# Миграция с Grounding DINO + SAM2 → SAM3.1
# SAM3.1 находит регионы материалов по текстовому промпту и возвращает
# точные пиксельные маски напрямую (без отдельного детектора).
# Z-index сборка: структурный → базовый → финишный → декоративный
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

print("=== 1. ПОДГОТОВКА ИЗОЛИРОВАННОЙ СТЕНЫ ===")
torch.cuda.empty_cache()
gc.collect()

if 'img_norm_rgb' not in locals() or 'geom_masks' not in locals():
    raise ValueError("ОШИБКА: Запустите Ячейку 1 для получения базовых данных!")

original_size = img_norm_rgb.shape[:2]

if 'true_silhouette' not in locals():
    print("-> Восстановление силуэта (U-2-Net)...")
    rgba_img = remove(img_norm_rgb)
    true_silhouette = rgba_img[:, :, 3] > 128

elements_mask = np.zeros(original_size, dtype=bool)
for key in ["window", "door", "balcony", "roof", "molding"]:
    if key in geom_masks:
        elements_mask = elements_mask | geom_masks[key]
bare_wall_mask = true_silhouette & ~elements_mask


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


# ── ХЕЛПЕР: SAM3.1 → маски по текстовым промптам ──────────────────────────
def sam3_segment(image_rgb, prompts_map, region_mask=None, threshold=SAM3_THRESHOLD):
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


# ── ШАГ 2: СКАНИРОВАНИЕ МАТЕРИАЛОВ (SAM3.1) ───────────────────────────────
print("=== 3. СКАНИРОВАНИЕ МАТЕРИАЛОВ (SAM3.1) ===")

MATERIAL_PROMPTS_MAP = {
    "concrete": "grey concrete stone wall base foundation",
    "brick":    "terracotta red brick masonry wall surface",
    "plaster":  "plain smooth cement plaster wall facade",
    "molding":  "architectural molding cornice decoration ornament",
}

COLORS_MATERIALS = {
    "concrete": [127, 140, 141],
    "brick":    [211,  84,   0],
    "plaster":  [243, 214, 115],
    "molding":  [236, 240, 241],
}

print("-> Сканирование текстур фасада...")
raw_material_masks = sam3_segment(
    img_norm_rgb,
    MATERIAL_PROMPTS_MAP,
    region_mask=bare_wall_mask,
)

# Усиление кирпича: добавляем маску оголённой кладки из Ячейки 2 (если доступна)
if 'final_wall_defect_masks' in locals():
    exposed = final_wall_defect_masks.get("exposed_brick", np.zeros(original_size, dtype=bool))
    raw_material_masks["brick"] |= (exposed & bare_wall_mask)

torch.cuda.empty_cache()
gc.collect()
print("✓ Сканирование материалов завершено.")


# ── ШАГ 3: УЛЬТИМАТИВНАЯ Z-INDEX СБОРКА ────────────────────────────────────
print("=== 4. ИЕРАРХИЧЕСКАЯ Z-INDEX СБОРКА BIM-КАРТЫ ===")

mask_only_canvas = np.zeros((*original_size, 3), dtype=np.uint8)
final_masks_materials = {k: np.zeros(original_size, dtype=bool) for k in MATERIAL_PROMPTS_MAP}
found_materials = []

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
axes[1].set_title("2. BIM-Карта Материалов (SAM3.1)", fontsize=16, fontweight='bold')
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
print("\n=== ВЕДОМОСТЬ ОТДЕЛКИ ФАСАДА (Material Report) ===")
total_wall_area = bare_wall_mask.sum()
for mat_name in found_materials:
    mat_area = final_masks_materials[mat_name].sum()
    percent = (mat_area / total_wall_area) * 100 if total_wall_area > 0 else 0
    print(f"  • {mat_name.title()}: {mat_area} px ({percent:.1f}% от площади стены)")

print("\n✅ Z-Index анализ строительных материалов завершён!")
print("   Экспортированы: final_masks_materials")
