import torch
import torch.nn.functional as F
import numpy as np
import trimesh
import os
from train_pointnet import PointNetSeg  

OBJ_PATH = "D:\\diplom 1 sem\\models\\tooth_segment_reduced_14.obj"  
MODEL_PATH = "pointnet_tooth_segmentation_best.pth" 
OUTPUT_OBJ_PATH = "final.obj"  
NUM_POINTS = 200000 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PointNetSeg().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

mesh = trimesh.load_mesh(OBJ_PATH, process=False)
points = mesh.vertices

orig_points = points.copy() 
if len(points) > NUM_POINTS:
    choice = np.random.choice(len(points), NUM_POINTS, replace=False)
    sampled_points = points[choice]
    reverse_indices = choice
else:
    pad = NUM_POINTS - len(points)
    sampled_points = np.pad(points, ((0, pad), (0, 0)), mode="constant")
    reverse_indices = np.arange(len(points))  

sampled_points_tensor = torch.tensor(sampled_points, dtype=torch.float32).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    preds = model(sampled_points_tensor)
    preds = F.softmax(preds, dim=2)
    pred_labels = preds.argmax(dim=2).cpu().numpy().squeeze()

    print(f"Метки после предсказания: {np.unique(pred_labels)}")  

predicted_points = sampled_points[pred_labels == 1]

if predicted_points.size == 0:
    print("Зуб не найден!")
else:
    tooth_mesh = trimesh.PointCloud(predicted_points)
    tooth_mesh.export(OUTPUT_OBJ_PATH)
    print(f"Зуб сохранён в: {OUTPUT_OBJ_PATH}")
