import os
import numpy as np
import trimesh
import time
from tqdm import tqdm

MESHES_PATH = "D:\\diplom 1 sem\\models\\"
LABELS_PATH = "D:\\diplom 1 sem\\labels\\"
SAVE_PATH = "D:\\diplom 1 sem\\dataset_parts\\"

os.makedirs(SAVE_PATH, exist_ok=True)

mesh_files = sorted([f for f in os.listdir(MESHES_PATH) if f.endswith(".obj")])

already_saved = set([f.replace(".npz", ".obj") for f in os.listdir(SAVE_PATH)])
mesh_files = [f for f in mesh_files if f not in already_saved]

if not mesh_files:
    print(" Все модели уже обработаны!")
else:
    print(f" Найдено {len(mesh_files)} новых моделей. Начинаем обработку...\n")

start_time = time.time()

for i, mesh_file in enumerate(tqdm(mesh_files, desc="Обработка моделей", unit="model")):
    file_start_time = time.time()

    label_file = mesh_file.replace(".obj", ".txt")
    mesh_path = os.path.join(MESHES_PATH, mesh_file)
    label_path = os.path.join(LABELS_PATH, label_file)
    save_file = os.path.join(SAVE_PATH, mesh_file.replace(".obj", ".npz"))

    print(f" Обрабатывается модель: {mesh_file} | Файл меток: {label_file}")

    if not os.path.exists(label_path):
        print(f" [Пропуск] Нет меток для {mesh_file}")
        continue

    try:
        mesh = trimesh.load_mesh(mesh_path, process=False)
        points = mesh.vertices
        labels = np.loadtxt(label_path, dtype=int)

        if len(points) != len(labels):
            print(f" [Ошибка] Несоответствие точек и меток в {mesh_file}: {len(points)} vs {len(labels)}")
            continue

        np.savez(save_file, points=points, labels=labels)
        file_time = time.time() - file_start_time
        print(f" [Готово] Сохранено в: {save_file} | Время: {file_time:.2f} сек.")

    except Exception as e:
        print(f" [Ошибка] {mesh_file}: {e}")

total_time = time.time() - start_time
print("\n Обработка завершена!")
print(f" Всего обработано: {len(mesh_files)} моделей")
print(f" Общее время выполнения: {total_time:.2f} секунд")
