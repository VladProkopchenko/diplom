import numpy as np
import trimesh

# Загружаем данные из .npz файла
data = np.load('balanced_dataset/tooth_segment_reduced_14.npz')  # Укажите путь к вашему .npz файлу
points = data['points']  # Точки модели

# Создаем объект PointCloud из точек
mesh = trimesh.PointCloud(points)

# Экспортируем в .obj файл
mesh.export('original_model.obj')

print(f'[DONE] Модель сохранена в файл original_model.obj')
