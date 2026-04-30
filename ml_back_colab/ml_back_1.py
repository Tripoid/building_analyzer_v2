import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from google.colab import files

# === УСТАНОВКА SAM3 ===
# Выполни один раз в начале сессии Colab:
# !git clone https://github.com/facebookresearch/sam3.git -q
# !pip install -e ./sam3 -q

device = "cuda" if torch.cuda.is_available() else "cpu"

# === 1. КЭШИРОВАННАЯ ЗАГРУЗКА SAM3.1 ===
if 'sam3_model' not in locals():
    print("Загрузка SAM3.1...")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    sam3_model = build_sam3_image_model()
    sam3_processor = Sam3Processor(sam3_model)
    print("✓ SAM3.1 загружен.")

# === 2. НАСТРОЙКИ КЛАССОВ И ЦВЕТОВ ===
# Отдельный промпт на каждый класс — SAM3.1 возвращает маски + скоры напрямую.
CLASS_PROMPTS = {
    "window":   "window glass pane window frame glazing dormer window",
    "door":     "entrance door front door gate doorway",
    "balcony":  "balcony terrace loggia balcony slab",
    "building": "building facade wall",
}

SAM3_THRESHOLD = 0.40

COLORS_HEX = {"window": "cyan", "door": "red", "balcony": "magenta", "building": "blue"}
COLORS_RGBA = {
    "window":   np.array([0.0, 1.0, 1.0, 0.5]),
    "door":     np.array([1.0, 0.0, 0.0, 0.5]),
    "balcony":  np.array([1.0, 0.0, 1.0, 0.5]),
    "building": np.array([0.0, 0.0, 1.0, 0.1]),
}
COLORS_RGB_SOLID = {
    "window":   np.array([0, 255, 255]),
    "door":     np.array([255, 0, 0]),
    "balcony":  np.array([255, 0, 255]),
    "building": np.array([0, 0, 255]),
}

# === 3. ОБРАБОТКА ФОТОГРАФИИ ===
print("\nЗагрузите фотографию:")
uploaded = files.upload()

# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ ЯЧЕЙКИ 2
global img_norm_rgb, geom_masks

for filename in uploaded.keys():
    print("-> Применение нормализации освещения (CLAHE)...")
    img_bgr = cv2.imread(filename)

    max_dim = 1024
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img_clahe_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    img_norm_rgb = cv2.fastNlMeansDenoisingColored(
        cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2RGB), None, 5, 5, 7, 21
    )

    image = Image.fromarray(img_norm_rgb)
    original_size = img_norm_rgb.shape[:2]  # (H, W)

    # --- ИНИЦИАЛИЗАЦИЯ СЛОВАРЯ МАСОК ДЛЯ ЯЧЕЙКИ 2 ---
    geom_masks = {k: np.zeros(original_size, dtype=bool) for k in CLASS_PROMPTS}

    print("-> Сканирование геометрии (SAM3.1)...")
    inference_state = sam3_processor.set_image(image)

    all_detections = []  # [(class_key, mask, score, box)]

    for class_key, prompt in CLASS_PROMPTS.items():
        output = sam3_processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks = output.get("masks")
        scores = output.get("scores")
        boxes = output.get("boxes")
        if masks is None or len(masks) == 0:
            continue

        for i, (mask, score) in enumerate(zip(masks, scores)):
            if float(score) < SAM3_THRESHOLD:
                continue
            m = np.asarray(mask, dtype=bool)
            if np.any(m):
                geom_masks[class_key] |= m
                box = boxes[i].tolist() if boxes is not None and len(boxes) > i else None
                all_detections.append((class_key, m, float(score), box))

    print(f"SAM3.1 нашел {len(all_detections)} объектов.")

    print("-> Сборка архитектурного плана...")
    fig, axes = plt.subplots(1, 3, figsize=(28, 9))
    ax1, ax2, ax3 = axes

    ax1.imshow(img_norm_rgb)
    ax1.set_title("1. SAM3.1 Рамки + Маски", fontsize=16, fontweight='bold')
    ax1.axis('off')
    ax2.imshow(img_norm_rgb)
    ax2.set_title("2. Только SAM3.1 (Маски на фото)", fontsize=16, fontweight='bold')
    ax2.axis('off')
    pure_mask = np.zeros((*original_size, 3), dtype=np.uint8)

    # Сортируем по площади маски (большие первыми — рисуются снизу)
    all_detections.sort(key=lambda x: x[1].sum(), reverse=True)

    for class_key, m, score, box in all_detections:
        color_rgba = COLORS_RGBA.get(class_key, np.array([1.0, 1.0, 0.0, 0.5]))
        overlay = np.zeros((original_size[0], original_size[1], 4))
        overlay[m] = color_rgba
        ax1.imshow(overlay)
        ax2.imshow(overlay)

        color_solid = COLORS_RGB_SOLID.get(class_key, np.array([255, 255, 0]))
        pure_mask[m] = color_solid

        if box is not None:
            xmin, ymin, xmax, ymax = box
            color_hex = COLORS_HEX.get(class_key, "yellow")
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor=color_hex, facecolor='none'
            )
            ax1.add_patch(rect)
            ax1.text(
                xmin, ymin - 5, f"{class_key} ({score:.2f})",
                color='black', fontsize=10, fontweight='bold',
                bbox=dict(facecolor=color_hex, alpha=0.8)
            )

    ax3.imshow(pure_mask)
    ax3.set_title("3. Чистая маска (Архитектурный план)", fontsize=16, fontweight='bold')
    ax3.axis('off')
    plt.tight_layout()
    plt.show()

    torch.cuda.empty_cache()
    print("✅ Этап 1 завершен. Данные (img_norm_rgb и geom_masks) готовы для Ячейки 2!")
