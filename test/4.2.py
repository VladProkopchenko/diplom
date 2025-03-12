import torch
import numpy as np
import trimesh

# Указываем пути к файлам
OBJ_INPUT_PATH = "D:\\diplom 1 sem\\models\\tooth_segment_reduced_5.obj"  # Исходная 3D-модель
OBJ_OUTPUT_PATH = "D:\\diplom 1 sem\\segmented_tooth.obj"  # Выходной файл
MODEL_PATH = "D:\\diplom 1 sem\\pointnet.pth"  # Файл с обученной моделью
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели
class PointNetSeg(torch.nn.Module):
    def __init__(self, in_dim=3, out_dim=2):
        super(PointNetSeg, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, out_dim)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Загрузка 3D-модели
def load_obj(file_path):
    mesh = trimesh.load_mesh(file_path)
    points = np.array(mesh.vertices, dtype=np.float32)
    return points

# Сохранение сегментированной 3D-модели
def save_obj(file_path, points):
    with open(file_path, "w") as f:
        for p in points:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")

# Основная функция сегментации
def segment_tooth():
    # Загружаем модель
    model = PointNetSeg().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Загружаем 3D-модель челюсти
    points = load_obj(OBJ_INPUT_PATH)
    print(f"Загружено {len(points)} точек из {OBJ_INPUT_PATH}")

    # Конвертация в тензор (без нормализации!)
    points_tensor = torch.tensor(points, dtype=torch.float32).to(DEVICE)

    # Прогоняем через сеть
    with torch.no_grad():
        outputs = model(points_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    # Выбираем только точки, относящиеся к коренному зубу (label=1)
    segmented_points = points[predictions == 1]

    print(f"Количество точек, отнесенных к коренному зубу: {len(segmented_points)}")

    # Сохраняем результат
    if len(segmented_points) > 0:
        save_obj(OBJ_OUTPUT_PATH, segmented_points)
        print(f"Сегментированный зуб сохранен в {OBJ_OUTPUT_PATH}")
    else:
        print("Ошибка: модель не нашла коренной зуб!")

# Запуск сегментации
if __name__ == "__main__":
    segment_tooth()
