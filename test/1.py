import trimesh
import numpy as np
import torch
from scipy.spatial import cKDTree

def save_ply_with_labels(vertices, labels, filename):
    """Сохраняем 3D-модель в PLY с метками для зуба."""
    # Создаем массив цветов: если метка 1, цвет будет красным, если 0 — синим
    vertex_colors = np.array([[255, 0, 0, 255] if label == 1 else [0, 0, 255, 255] for label in labels])
    
    # Проверим, сколько точек меток зуба
    print(f"Количество точек, размеченных как зуб (метка 1): {np.sum(labels)}")
    
    # Создаем модель и сохраняем
    mesh = trimesh.Trimesh(vertices=vertices)
    mesh.visual.vertex_colors = vertex_colors  # Применяем цвета к вершинам
    mesh.export(filename)
    print(f"Файл сохранен как {filename}")
    
def load_obj(filename):
    """Загружаем 3D-модель и извлекаем вершины."""
    mesh = trimesh.load_mesh(filename)
    return mesh

def normalize_mesh(mesh):
    """Масштабирует 3D-модель так, чтобы она помещалась в куб размером 1x1x1."""
    centroid = mesh.vertices.mean(axis=0)  # Центр объекта
    scale = np.max(mesh.vertices) - np.min(mesh.vertices)  # Масштаб (разница между макс и мин координатами)

    mesh.vertices = (mesh.vertices - centroid) / scale  # Центрируем и масштабируем
    return mesh

def annotate_points_kdtree(china_mesh, tooth_mesh, threshold=0.01):
    """Аннотируем точки челюсти с помощью KD-дерева для быстрого поиска ближайших соседей."""
    china_vertices = china_mesh.vertices  # Вершины челюсти
    tooth_vertices = tooth_mesh.vertices  # Вершины зуба

    labels = np.zeros(len(china_vertices), dtype=int)  # Все точки изначально 0

    # Создаем KD-дерево из точек зуба
    tooth_tree = cKDTree(tooth_vertices)

    # Находим расстояние до ближайшей точки зуба
    distances, _ = tooth_tree.query(china_vertices, k=1)

    # Выводим статистику расстояний
    print(f"Минимальное расстояние: {distances.min():.6f}")
    print(f"Среднее расстояние: {distances.mean():.6f}")
    print(f"Максимальное расстояние: {distances.max():.6f}")

    # Если расстояние меньше порога, присваиваем метку 1 (это зуб)
    labels[distances < threshold] = 1

    num_tooth_points = labels.sum()
    print(f"Количество точек, размеченных как зуб: {num_tooth_points}")

    return torch.tensor(china_vertices, dtype=torch.float), torch.tensor(labels, dtype=torch.long)

# Указываем пути к файлам OBJ
china_obj = "D:\\diplom 1 sem\\models\\tooth_segment_reduced_30.obj"  # Путь к OBJ файлу с челюстью
tooth_obj = "D:\\diplom 1 sem\\models_left_tooth\\30.obj"  # Путь к OBJ файлу с коренным зубом

# Загружаем 3D-модели
china_mesh = load_obj(china_obj)
tooth_mesh = load_obj(tooth_obj)

# Проверяем центры перед нормализацией
china_centroid = china_mesh.vertices.mean(axis=0)
tooth_centroid = tooth_mesh.vertices.mean(axis=0)
print(f"Центр челюсти до нормализации: {china_centroid}")
print(f"Центр зуба до нормализации: {tooth_centroid}")

# Нормализуем модели (центрируем и приводим к единому масштабу)
#china_mesh = normalize_mesh(china_mesh)
#tooth_mesh = normalize_mesh(tooth_mesh)

# Проверяем центры после нормализации
china_centroid = china_mesh.vertices.mean(axis=0)
tooth_centroid = tooth_mesh.vertices.mean(axis=0)
print(f"Центр челюсти после нормализации: {china_centroid}")
print(f"Центр зуба после нормализации: {tooth_centroid}")

# Выводим количество точек челюсти и зуба
print(f"Количество точек челюсти: {len(china_mesh.vertices)}")
print(f"Количество точек зуба: {len(tooth_mesh.vertices)}")

# Аннотируем точки
points, labels = annotate_points_kdtree(china_mesh, tooth_mesh, threshold=0.03)

# Проверяем размер полученных данных
print(f"Количество точек челюсти: {points.shape[0]}")
print(f"Количество меток зуба: {labels.sum().item()}")  # Количество точек с меткой 1 (зуб)

# Сохраняем метки в файл
np.savetxt("D:\\diplom 1 sem\\labels\\tooth_labels30.txt", labels.numpy(), fmt="%d")
#save_ply_with_labels(points.numpy(), labels.numpy(), 'tooth_with_labels.ply')
