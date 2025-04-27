import os
import numpy as np
from tqdm import tqdm

# Путь к входной папке с исходными .npz файлами
INPUT_DIR = "dataset_parts"
# Путь к выходной папке, куда сохраняются сбалансированные .npz файлы
OUTPUT_DIR = "balanced_dataset"

# Количество точек каждого класса в сбалансированном датасете
NUM_POINTS_PER_CLASS = 100000

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".npz")]
print(f"[INFO] Обнаружено {len(files)} файлов для обработки.")

for filename in tqdm(files):
    path = os.path.join(INPUT_DIR, filename)
    data = np.load(path)
    points = data["points"]
    labels = data["labels"]

    tooth_indices = np.where(labels == 1)[0]
    background_indices = np.where(labels == 0)[0]

    if len(tooth_indices) < NUM_POINTS_PER_CLASS:
        print(f"[WARN] В файле {filename} только {len(tooth_indices)} точек зуба (нужно {NUM_POINTS_PER_CLASS})")
        # Дублируем точки зуба
        duplicated_tooth_indices = np.random.choice(tooth_indices, NUM_POINTS_PER_CLASS, replace=True)
    else:
        duplicated_tooth_indices = np.random.choice(tooth_indices, NUM_POINTS_PER_CLASS, replace=False)

    if len(background_indices) < NUM_POINTS_PER_CLASS:
        print(f"[WARN] В файле {filename} только {len(background_indices)} точек фона (нужно {NUM_POINTS_PER_CLASS})")
        duplicated_background_indices = np.random.choice(background_indices, NUM_POINTS_PER_CLASS, replace=True)
    else:
        duplicated_background_indices = np.random.choice(background_indices, NUM_POINTS_PER_CLASS, replace=False)

    selected_indices = np.concatenate([duplicated_tooth_indices, duplicated_background_indices])
    np.random.shuffle(selected_indices)

    balanced_points = points[selected_indices]
    balanced_labels = labels[selected_indices]

    save_path = os.path.join(OUTPUT_DIR, filename)
    np.savez(save_path, points=balanced_points, labels=balanced_labels)
    print(f"[OK] Сбалансированный файл сохранён: {save_path}")

print("[DONE] Все файлы обработаны.")
