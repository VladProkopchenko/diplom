import trimesh
import numpy as np
import torch
from scipy.spatial import cKDTree

def save_ply_with_labels(vertices, labels, filename):
    assert len(vertices) == len(labels), "Размер вершин и меток не совпадает!"

    vertex_colors = np.array([[255, 0, 0] if label == 1 else [0, 0, 255] for label in labels], dtype=np.uint8)

    mesh = trimesh.Trimesh(vertices=vertices, process=False)  

    mesh.visual.vertex_colors = vertex_colors

    mesh.metadata['ply_raw'] = {
        'format': 'binary_little_endian',
        'vertex': {
            'x': 'float',
            'y': 'float',
            'z': 'float',
            'red': 'uchar',
            'green': 'uchar',
            'blue': 'uchar'
        }
    }

    mesh.export(filename, file_type='ply')
    print(f"Файл сохранен как {filename}")


    
def load_obj(filename):
    mesh = trimesh.load_mesh(filename, process = False)
    return mesh

#def normalize_mesh(mesh):
#    """Масштабирует 3D-модель так, чтобы она помещалась в куб размером 1x1x1."""
#    centroid = mesh.vertices.mean(axis=0)  # Центр объекта
#    scale = np.max(mesh.vertices) - np.min(mesh.vertices)  # Масштаб (разница между макс и мин координатами)
#
#    mesh.vertices = (mesh.vertices - centroid) / scale  # Центрируем и масштабируем
    return mesh

def annotate_points_kdtree(china_mesh, tooth_mesh, threshold=0.01):
    china_vertices = china_mesh.vertices  
    tooth_vertices = tooth_mesh.vertices  

    labels = np.zeros(len(china_vertices), dtype=int) 

    tooth_tree = cKDTree(tooth_vertices)

    distances, _ = tooth_tree.query(china_vertices, k=1)

    print(f"Минимальное расстояние: {distances.min():.6f}")
    print(f"Среднее расстояние: {distances.mean():.6f}")
    print(f"Максимальное расстояние: {distances.max():.6f}")

    labels[distances < threshold] = 1

    num_tooth_points = labels.sum()
    print(f"Количество точек, размеченных как зуб: {num_tooth_points}")

    return torch.tensor(china_vertices, dtype=torch.float), torch.tensor(labels, dtype=torch.long)

china_obj = "D:\\diplom 1 sem\\models\\tooth_segment_reduced_14.obj"  
tooth_obj = "D:\\diplom 1 sem\\models_left_tooth\\14.obj"  

china_mesh = load_obj(china_obj)
tooth_mesh = load_obj(tooth_obj)
china_centroid = china_mesh.vertices.mean(axis=0)
tooth_centroid = tooth_mesh.vertices.mean(axis=0)
print(f"Центр челюсти до нормализации: {china_centroid}")
print(f"Центр зуба до нормализации: {tooth_centroid}")

print(f"Количество точек челюсти: {len(china_mesh.vertices)}")
print(f"Количество точек зуба: {len(tooth_mesh.vertices)}")

points, labels = annotate_points_kdtree(china_mesh, tooth_mesh, threshold=0.03)

print(f"Количество точек челюсти: {points.shape[0]}")
print(f"Количество меток зуба: {labels.sum().item()}")  

np.savetxt("D:\\diplom 1 sem\\labels\\tooth_segment_reduced_14.txt", labels.numpy(), fmt="%d")
save_ply_with_labels(points.numpy(), labels.numpy(), 'D:\\diplom 1 sem\\ply_models\\tooth_with_labels14.ply')
