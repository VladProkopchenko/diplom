import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import numpy as np
import torch.utils.data as data
import logging
from pathlib import Path

# Модель PointNet для сегментации
class PointNetSegmentation(nn.Module):
    def __init__(self):
        super(PointNetSegmentation, self).__init__()
        
        # Модуль для обработки входных точек
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        
        # Сегментационный слой
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 2)  # 2 класса: зуб и не зуб

        # Dropout и нормализация
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Прямой проход через слои
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        # Сегментация
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)  # Выход для классификации на 2 класса

        return F.log_softmax(x, dim=-1)


# Класс для загрузки и подготовки данных
class ToothSegmentationDataset(data.Dataset):
    def __init__(self, points_list, labels_list):
        """
        points_list: Список точек 3D-моделей челюстей
        labels_list: Список меток для точек (0 — не зуб, 1 — зуб)
        """
        self.points_list = points_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.points_list)

    def __getitem__(self, idx):
        return self.points_list[idx], self.labels_list[idx]


# Функции для загрузки 3D-моделей и меток
def load_obj(filename):
    """Загружаем 3D-модель и извлекаем вершины."""
    mesh = trimesh.load_mesh(filename)
    return mesh

def load_labels(label_file):
    """Загружаем метки из текстового файла."""
    return np.loadtxt(label_file, dtype=int)


def annotate_points_from_labels(china_mesh, labels):
    """Загружаем аннотированные точки и метки для 3D-модели челюсти."""
    points = torch.tensor(china_mesh.vertices, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    return points, labels


# Функция для загрузки всех файлов из папок
def load_data_from_directory(china_dir, tooth_dir, label_dir):
    china_objs = list(Path(china_dir).glob('*.obj'))  # Все OBJ файлы с челюстями
    tooth_objs = list(Path(tooth_dir).glob('*.obj'))  # Все OBJ файлы с зубами
    label_files = list(Path(label_dir).glob('*.txt'))  # Все файлы с метками
    
    if len(china_objs) != len(tooth_objs) or len(china_objs) != len(label_files):
        raise ValueError("Количество файлов челюстей, зубов и меток должно быть одинаковым.")
    
    points_list = []
    labels_list = []

    # Загружаем данные из файлов
    for i in range(len(china_objs)):
        china_mesh = load_obj(china_objs[i])
        tooth_mesh = load_obj(tooth_objs[i])
        labels = load_labels(label_files[i])

        # Получаем аннотированные точки и метки
        points, labels = annotate_points_from_labels(china_mesh, labels)

        points_list.append(points)
        labels_list.append(labels)

    return points_list, labels_list


# Логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Путь к директориям с 3D-моделями и метками
china_dir = "D:\\diplom 1 sem\\models"  # Путь к папке с челюстями
tooth_dir = "D:\\diplom 1 sem\\models_left_tooth"  # Путь к папке с зубами
label_dir = "D:\\diplom 1 sem\\labels"  # Путь к папке с метками

logger.info(f"Загрузка данных из папок {china_dir}, {tooth_dir}, {label_dir}...")

# Загрузка всех данных
points_list, labels_list = load_data_from_directory(china_dir, tooth_dir, label_dir)
logger.info(f"Загружено {len(points_list)} объектов.")

# Подготовка датасета
dataset = ToothSegmentationDataset(points_list, labels_list)
dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

# Инициализация модели, потерь и оптимизатора
model = PointNetSegmentation().cuda()  # Если есть CUDA
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Обучение модели
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Переводим в режим обучения
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    for i, (points_batch, labels_batch) in enumerate(dataloader):
        points_batch = points_batch.cuda()  # Перемещаем на GPU
        labels_batch = labels_batch.cuda()  # Перемещаем на GPU
        
        optimizer.zero_grad()  # Обнуляем градиенты

        # Прямой проход
        outputs = model(points_batch)
        
        # Рассчитываем потери
        loss = criterion(outputs, labels_batch)
        loss.backward()  # Обратный проход
        optimizer.step()  # Шаг оптимизации
        
        # Статистика
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_preds += labels_batch.size(0)
        correct_preds += (predicted == labels_batch).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct_preds / total_preds
    logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# Оценка модели
model.eval()  # Переводим в режим оценки
correct_preds = 0
total_preds = 0

with torch.no_grad():
    for points_batch, labels_batch in dataloader:
        points_batch = points_batch.cuda()
        labels_batch = labels_batch.cuda()
        
        outputs = model(points_batch)
        _, predicted = torch.max(outputs.data, 1)
        total_preds += labels_batch.size(0)
        correct_preds += (predicted == labels_batch).sum().item()

accuracy = correct_preds / total_preds
logger.info(f"Accuracy on test data: {accuracy:.4f}")

# Сохранение модели
torch.save(model.state_dict(), 'tooth_segmentation_model.pth')
logger.info("Модель сохранена как 'tooth_segmentation_model.pth'.")
