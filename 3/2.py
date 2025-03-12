import trimesh
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Шаг 1: Преобразование OBJ в облака точек
def load_point_cloud(obj_file, num_points=800000):
    logger.info(f"Загрузка модели из файла: {obj_file}")
    mesh = trimesh.load_mesh(obj_file)
    points = mesh.sample(num_points)  # Берем num_points случайных точек из модели
    logger.info(f"Загружено {len(points)} точек из модели {obj_file}")
    return points

# Загружаем 3D-модели
jaw_points = load_point_cloud('D:\\diplom 1 sem\\obj model\\model.obj')  # Модель челюсти
tooth_points = load_point_cloud('D:\\diplom 1 sem\\models_left_tooth\\11.obj')  # Модель коренного зуба

# Подготовка меток для точек челюсти
logger.info("Подготовка меток для точек челюсти")
labels = np.zeros(len(jaw_points))  # Все точки сначала считаем не зубом
labels[0:50000] = 1  # Моделируем метки для 200 точек, которые будут зубом
logger.info(f"Метки подготовлены, первое количество точек зуба: {np.sum(labels == 1)}")

# Преобразуем точки и метки в тензоры
jaw_points_tensor = torch.tensor(jaw_points, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Шаг 2: Определение модели PointNet
class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        # Простая модель с использованием MLP для точек
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 2)  # 2 класса: зуб (1) и не зуб (0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # Каждый выход — это предсказание для каждой точки
        x = self.fc5(x)  # (num_points, 2), где 2 — количество классов (зуб или не зуб)
        return x

# Шаг 3: Обучение модели
logger.info("Инициализация модели PointNet и оптимизатора")
model = PointNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Прямой проход через модель
    output = model(jaw_points_tensor)  # Теперь мы передаем каждый кластер точек
    loss = F.cross_entropy(output, labels_tensor)  # Вычисляем потери для всех точек

    # Обратный проход и обновление весов
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        logger.info(f"Эпоха {epoch+1}/{num_epochs}, Потери: {loss.item()}")

# Шаг 4: Предсказание и сохранение результатов
logger.info("Предсказание и сохранение результата")
model.eval()
with torch.no_grad():
    output = model(jaw_points_tensor)  # Делаем предсказание
    predicted_labels = output.argmax(dim=1)  # Получаем метки для каждой точки

    # Проверим, сколько точек было классифицировано как зуб
    tooth_mask = predicted_labels == 1
    tooth_points_pred = jaw_points_tensor[tooth_mask]
    logger.info(f"Количество точек, классифицированных как зуб: {tooth_points_pred.size(0)}")

# Если найдено хотя бы одно предсказание зуба, сохраняем
if tooth_points_pred.size(0) > 0:
    tooth_points_pred_numpy = tooth_points_pred.detach().cpu().numpy()

    # Сохраняем точки в формате .obj
    with open('output_tooth.obj', 'w') as file:
        file.write("# https://github.com/mikedh/trimesh\n")  # Оставляем комментарий
        for point in tooth_points_pred_numpy:
            file.write(f"v {point[0]} {point[1]} {point[2]}\n")  # Записываем каждую точку как "v x y z"
    logger.info("Обработка завершена, результат сохранен в output_tooth.obj")
else:
    logger.warning("Не найдено точек, классифицированных как зуб. Проверьте обучение модели.")
