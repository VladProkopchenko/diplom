import torch
import numpy as np
import open3d as o3d
from pointnet import PointNetSeg  # Убедись, что этот модуль доступен

# Пути к файлам
MODEL_PATH = "D:\\diplom 1 sem\\pointnet.pth"
TEST_OBJ = "D:\\diplom 1 sem\\tooth_segment_reduced_37.obj"
OUTPUT_LABELS = "D:\\diplom 1 sem\\test_labels.txt"
OUTPUT_PLY = "D:\\diplom 1 sem\\test_labeled.ply"

# Определяем устройство (CUDA, если доступно)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Функция загрузки модели
def load_model(model_path, device):
    print("Загрузка модели...")
    model = PointNetSeg(in_dim=3, out_dim=2).to(device)  # Подставляем правильные аргументы
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Перевод в режим инференса
    return model

# Функция загрузки 3D модели
def load_obj(file_path, max_points=100_000):
    print("Загрузка 3D модели...")
    mesh = o3d.io.read_triangle_mesh(file_path)
    pcd = mesh.sample_points_uniformly(number_of_points=max_points)  # Уменьшаем число точек
    points = np.asarray(pcd.points)
    print(f"Загружено {points.shape[0]} точек")
    return points

# Функция предсказания меток точек
def predict_labels(model, points, device, batch_size=50_000):
    print("Предсказание меток...")
    model.eval()
    
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device).unsqueeze(0)  # [1, N, 3]
    points_tensor = points_tensor.permute(0, 2, 1)  # [1, 3, N]

    with torch.no_grad():  # Отключаем градиенты для оптимизации памяти
        outputs = model(points_tensor)  # [1, 2, N]
        labels = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()  # [N]

    return labels

# Функция сохранения результатов в PLY
def save_ply(points, labels, output_ply):
    print(f"Сохранение результата в {output_ply}...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Цвета: коренной зуб (красный), фон (синий)
    colors = np.zeros_like(points)
    colors[labels == 1] = [1, 0, 0]  # Красный
    colors[labels == 0] = [0, 0, 1]  # Синий
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_ply, pcd)
    print("Файл сохранен!")

# Основная функция
def main():
    model = load_model(MODEL_PATH, device)
    points = load_obj(TEST_OBJ, max_points=200_000)  # Уменьшаем количество точек
    labels = predict_labels(model, points, device)
    
    np.savetxt(OUTPUT_LABELS, labels, fmt="%d")
    print("Метки сохранены!")

    save_ply(points, labels, OUTPUT_PLY)

if __name__ == "__main__":
    main()