import trimesh
import numpy as np
import torch
import os
import time
import logging

# Настройка логирования
logging.basicConfig(
    filename="dataset_creation.log", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Папка с моделями
MESH_DIR = "D:\\diplom 1 sem\\models"
LABELS_DIR = "D:\\diplom 1 sem\\labels"
OUTPUT_FILE = "D:\\diplom 1 sem\\dataset.npz"

# Получаем список файлов
mesh_files = sorted([f for f in os.listdir(MESH_DIR) if f.endswith(".obj")])
label_files = sorted([f for f in os.listdir(LABELS_DIR) if f.endswith(".txt")])

logging.info(f"Найдено {len(mesh_files)} моделей и {len(label_files)} файлов с метками.")

# Проверяем соответствие количества файлов
if len(mesh_files) != len(label_files):
    logging.warning("Количество моделей и файлов с метками не совпадает!")

# Списки для хранения данных
all_points = []
all_labels = []

start_time = time.time()

for mesh_file, label_file in zip(mesh_files, label_files):
    mesh_path = os.path.join(MESH_DIR, mesh_file)
    label_path = os.path.join(LABELS_DIR, label_file)

    logging.info(f"Обрабатываю: {mesh_file} и {label_file}")

    try:
        # Загружаем модель
        mesh = trimesh.load_mesh(mesh_path)
        points = np.array(mesh.vertices, dtype=np.float32)

        # Загружаем метки
        labels = np.loadtxt(label_path, dtype=np.int64)

        # Проверяем размеры
        if len(points) != len(labels):
            logging.error(f"Ошибка: {mesh_file} и {label_file} имеют разное количество точек! Пропускаем.")
            continue

        all_points.append(points)
        all_labels.append(labels)
        logging.info(f"Успешно загружено: {points.shape[0]} точек.")

    except Exception as e:
        logging.error(f"Ошибка при обработке {mesh_file} и {label_file}: {e}")

# Объединяем данные
if all_points and all_labels:
    all_points = np.vstack(all_points)
    all_labels = np.concatenate(all_labels)

    # Сохраняем датасет
    np.savez(OUTPUT_FILE, points=all_points, labels=all_labels)
    logging.info(f" Датасет успешно сохранен в {OUTPUT_FILE}")
    logging.info(f"Общее количество точек: {all_points.shape[0]}")
else:
    logging.error("Не удалось создать датасет, недостаточно данных.")

end_time = time.time()
logging.info(f" Время выполнения: {end_time - start_time:.2f} секунд.")

print(f" Датасет сохранен в {OUTPUT_FILE}. Лог записан в dataset_creation.log")
