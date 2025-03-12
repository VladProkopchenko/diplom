import trimesh
import numpy as np
from scipy.spatial import KDTree

def load_and_downsample_obj(file_path, num_points=100000):
    """Загружает OBJ и уменьшает количество точек до num_points."""
    mesh = trimesh.load_mesh(file_path)
    points = np.array(mesh.vertices)

    if len(points) > num_points:
        idx = np.random.choice(len(points), num_points, replace=False)
        points = points[idx]

    return points


def create_labels(full_jaw_points, tooth_points, threshold=8.0):
    """Создаёт маску (1 – зуб, 0 – не зуб) на основе расстояния."""
    labels = np.zeros(len(full_jaw_points))  # Все точки сначала метим как 0
    
    # Создаём KD-дерево для быстрого поиска ближайших соседей
    tree = KDTree(tooth_points)

    # Находим ближайшую точку из зуба для каждой точки челюсти
    distances, _ = tree.query(full_jaw_points)

    # Если расстояние меньше порога, считаем точку частью зуба
    labels[distances < threshold] = 1

    return labels

# Загружаем модели
full_jaw = load_and_downsample_obj("D:\\diplom 1 sem\\obj model\\model.obj", num_points=50000)
tooth = load_and_downsample_obj("D:\\diplom 1 sem\\models_left_tooth\\11.obj", num_points=5000)  # Зуб меньше

# Создаём разметку
labels = create_labels(full_jaw, tooth)
print(f"Метки созданы: {np.sum(labels)} точек принадлежат зубу.")

