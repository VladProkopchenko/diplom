import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import numpy as np
import logging

# Настроим логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Настройка устройства (GPU или CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Класс модели PointNet с улучшениями
class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=1)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # 2 класса: зуб или фон

    def forward(self, pos):
        # Преобразуем вход в нужную форму для Conv1d (batch_size, channels, n_points)
        x = pos.transpose(1, 2)  # Это нужно, чтобы форма была (batch_size, 3, n_points)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, dim=2)[0]  # Max pooling по всем точкам
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Загрузка OBJ моделей
def load_obj(file_path):
    mesh = trimesh.load_mesh(file_path)
    points = mesh.vertices  # Получаем координаты точек
    return points

# Подготовка данных для обучения
def prepare_data(chin_points, tooth_points):
    logger.info(f"Количество точек челюсти: {len(chin_points)}")
    logger.info(f"Количество точек зуба: {len(tooth_points)}")

    # Создаем метки для точек челюсти
    labels = np.zeros(len(chin_points))
    
    # Ищем точки, которые принадлежат зубу
    tooth_indices = np.random.choice(len(chin_points), size=len(tooth_points), replace=False)
    labels[tooth_indices] = 1  # метки для зуба (1)

    # Преобразуем данные в тензоры
    chin_points_tensor = torch.tensor(chin_points, dtype=torch.float).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

    # Убедимся, что данные имеют форму (batch_size, 3, n_points)
    chin_points_tensor = chin_points_tensor.transpose(0, 1).unsqueeze(0)  # преобразуем в (1, 3, n_points)

    return chin_points_tensor, labels_tensor

# Обучение модели
def train_model(model, chin_points_tensor, labels_tensor, num_epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    class_weights = torch.tensor([1.0, 10.0]).to(device)  # Веса для классов (0 - фон, 1 - зуб)
    
    for epoch in range(num_epochs):
        model.train()
        
        optimizer.zero_grad()
        
        # Прогоняем модель
        output = model(chin_points_tensor)

        # Вычисляем потерю
        loss = F.cross_entropy(output, labels_tensor, weight=class_weights)
        
        # Обратное распространение ошибки
        loss.backward()
        optimizer.step()

        # Логируем потери
        if epoch % 10 == 0:
            logger.info(f"Эпоха {epoch}/{num_epochs}, Потери: {loss.item()}")

    return model

# Предсказание
def predict_and_save(model, chin_points_tensor, output_file="output_tooth.obj"):
    model.eval()
    
    with torch.no_grad():
        output = model(chin_points_tensor)
        _, predicted_labels = torch.max(output, dim=1)

        # Извлекаем точки, классифицированные как зуб
        tooth_points_pred = chin_points_tensor[0, :, predicted_labels == 1]  # Внимание на индексацию

        logger.info(f"Количество точек, классифицированных как зуб: {len(tooth_points_pred)}")

        # Сохраняем результат
        if len(tooth_points_pred) > 0:
            tooth_mesh = trimesh.PointCloud(tooth_points_pred.cpu().numpy())
            tooth_mesh.export(output_file)
            logger.info(f"Результат сохранен в {output_file}")
        else:
            logger.warning("Не найдено точек, классифицированных как зуб. Проверьте обучение модели.")

# Основная функция
def main():
    chin_model_path = "D:/diplom 1 sem/obj model/model.obj"
    tooth_model_path = "D:/diplom 1 sem/models_left_tooth/11.obj"
    
    # Загружаем модели
    chin_points = load_obj(chin_model_path)
    tooth_points = load_obj(tooth_model_path)
    
    # Подготовим данные
    chin_points_tensor, labels_tensor = prepare_data(chin_points, tooth_points)

    # Инициализация модели
    model = PointNet().to(device)
    
    # Обучаем модель
    model = train_model(model, chin_points_tensor, labels_tensor, num_epochs=100, lr=0.001)
    
    # Предсказание и сохранение результатов
    predict_and_save(model, chin_points_tensor, output_file="output_tooth.obj")

if __name__ == "__main__":
    main()
