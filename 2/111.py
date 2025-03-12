import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Функция для загрузки и уменьшения количества точек в 3D-модели
def load_and_downsample_obj(file_path, num_points=50000):
    """Загружает OBJ и уменьшает количество точек до num_points."""
    import trimesh
    mesh = trimesh.load_mesh(file_path)
    points = np.array(mesh.vertices)

    if len(points) > num_points:
        idx = np.random.choice(len(points), num_points, replace=False)
        points = points[idx]

    return points

# Функция для создания меток на основе расстояния
def create_labels(full_jaw_points, tooth_points, threshold=6.0):
    """Создаёт маску (1 – зуб, 0 – не зуб) на основе расстояния."""
    labels = np.zeros(len(full_jaw_points))  # Все точки сначала метим как 0
    
    # Создаём KD-дерево для быстрого поиска ближайших соседей
    from scipy.spatial import KDTree
    tree = KDTree(tooth_points)

    # Находим ближайшую точку из зуба для каждой точки челюсти
    distances, _ = tree.query(full_jaw_points)

    # Если расстояние меньше порога, считаем точку частью зуба
    labels[distances < threshold] = 1

    return labels

# Загрузим модели челюсти и зуба
#full_jaw = load_and_downsample_obj("D:\\diplom 1 sem\\obj model\\model.obj", num_points=50000)
full_jaw = load_and_downsample_obj(r"D:\\diplom 1 sem\\models\\tooth_segment_reduced_1.obj", num_points=50000)
tooth = load_and_downsample_obj(r"D:\\diplom 1 sem\\models_left_tooth\\11.obj", num_points=5000)

# Создаём разметку для модели челюсти
labels = create_labels(full_jaw, tooth, threshold=6.0)
print(f"Метки созданы: {np.sum(labels)} точек принадлежат зубу.")

# Датасет для обучения
class ToothDataset(Dataset):
    """Датасет для обучения нейросети"""
    def __init__(self, point_clouds, labels):
        self.point_clouds = [torch.tensor(pc, dtype=torch.float32) for pc in point_clouds]
        self.labels = [torch.tensor(lbl, dtype=torch.float32) for lbl in labels]
    
    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        return self.point_clouds[idx], self.labels[idx]

# Создаём датасет
dataset = ToothDataset([full_jaw], [labels])
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Простая нейросеть (PointNet-подобная)
class PointNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super(PointNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Создаём модель
model = PointNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# Обучение
for epoch in range(10):  # Количество эпох
    for batch in dataloader:
        pc, lbl = batch
        optimizer.zero_grad()
        output = model(pc)
        loss = loss_fn(output.squeeze(), lbl.squeeze())
        loss.backward()
        optimizer.step()
    
    print(f"Эпоха {epoch+1}, Потери: {loss.item():.4f}")

print("Обучение завершено!")
