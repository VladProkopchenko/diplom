import torch
import numpy as np
import trimesh

from balanced_train import PointNetSeg

# Параметры
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 2
chunk_size = 200_000

# Пути
input_obj_path = 'models\\tooth_segment_reduced_6.obj'   # Заменить на нужный
output_obj_path = 'predicted_tooth2.obj'

# Загрузка модели
model = PointNetSeg(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('pointnet_tooth_segmentation_balanced.pth', map_location=device))
model.eval()

# Загрузка OBJ-модели
mesh = trimesh.load(input_obj_path, process=False)
points = np.asarray(mesh.vertices)  # (N, 3)
N = points.shape[0]

# Подготовка предсказаний
pred_labels = np.zeros(N, dtype=int)

# Обработка по чанкам
with torch.no_grad():
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = points[start:end]
        
        # Трансформация
        pts = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to(device)  # (1, 3, chunk_size)

        # Предсказание
        out = model(pts)  # (1, num_classes, chunk_size)
        out = out.squeeze(0).permute(1, 0)  # (chunk_size, num_classes)
        chunk_labels = out.argmax(dim=1).cpu().numpy()  # (chunk_size,)

        pred_labels[start:end] = chunk_labels

# Сохранение зуба
tooth_points = points[pred_labels == 1]
if len(tooth_points) > 0:
    tooth_mesh = trimesh.PointCloud(tooth_points)
    tooth_mesh.export(output_obj_path)
    print(f"[DONE] Сохранено {len(tooth_points)} точек зуба в {output_obj_path}")
else:
    print("[WARNING] Точек зуба не найдено.")
