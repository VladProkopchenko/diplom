import torch
import numpy as np
import trimesh

from balanced_train import PointNetSeg  # Импорт твоей модели

# Параметры
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 2  # Классов у тебя 2: фон и зуб

# 1. Загружаем модель
model = PointNetSeg(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('pointnet_tooth_segmentation_balanced.pth', map_location=device))
model.eval()

test_data = np.load('balanced_dataset/tooth_segment_reduced_15.npz')  
points = test_data['points']  

points_tensor = torch.tensor(points.T, dtype=torch.float32).unsqueeze(0).to(device) 

with torch.no_grad():
    pred = model(points_tensor)  
    pred = pred.squeeze(0).permute(1, 0)  
    pred_labels = pred.argmax(dim=1).cpu().numpy()  

tooth_points = points[pred_labels == 1]  

if len(tooth_points) > 0:
    mesh = trimesh.PointCloud(tooth_points)
    mesh.export('predicted_tooth.obj')
    print(f'[DONE] Сохранено {len(tooth_points)} точек зуба в файл predicted_tooth.obj')
else:
    print('[WARNING] Никакие точки зуба не найдены.')
