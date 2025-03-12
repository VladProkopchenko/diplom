import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Фикс для Windows
if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Гиперпараметры
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Датасет
    class ToothDataset(Dataset):
        def __init__(self, npz_path):
            data = np.load(npz_path)
            self.points = torch.tensor(data["points"], dtype=torch.float32)
            self.labels = torch.tensor(data["labels"], dtype=torch.long)

        def __len__(self):
            return len(self.points)  # Количество всех точек

        def __getitem__(self, idx):
            return self.points[idx], self.labels[idx]  # Возвращаем одну точку и её метку

    # Загружаем датасет
    dataset = ToothDataset("D:\\diplom 1 sem\\dataset.npz")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Исправленный PointNet
    class FixedPointNet(nn.Module):
        def __init__(self):
            super(FixedPointNet, self).__init__()
            self.fc1 = nn.Linear(3, 64)
            self.fc2 = nn.Linear(64, 128)
            self.fc3 = nn.Linear(128, 256)
            self.fc4 = nn.Linear(256, 128)
            self.fc5 = nn.Linear(128, 2)  # Два класса: зуб / не зуб

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = self.fc5(x)  # Предсказание классов (без softmax)
            return x

    # Инициализация модели
    model = FixedPointNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Обучение
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_points, batch_labels in dataloader:
            batch_points, batch_labels = batch_points.to(DEVICE), batch_labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_points)  # [batch_size, 2]

            loss = criterion(outputs, batch_labels)  # CrossEntropyLoss ждет [batch, 2] и [batch]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

    # Сохранение модели
    torch.save(model.state_dict(), "D:\\diplom 1 sem\\pointnet.pth")
    print("Обучение завершено, модель сохранена!")
