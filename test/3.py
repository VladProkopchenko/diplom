import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Параметры
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
        return len(self.points) // 65536  # Обрезаем на фиксированный размер
    
    def __getitem__(self, idx):
        start = idx * 65536
        end = start + 65536
        return self.points[start:end], self.labels[start:end]

# Загружаем датасет
dataset = ToothDataset("D:\\diplom 1 sem\\dataset.npz")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Простая нейросеть
class SimplePointNet(nn.Module):
    def __init__(self):
        super(SimplePointNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Два выхода для классификации (зуб / не зуб)
        return x

# Инициализация модели
model = SimplePointNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Обучение
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_points, batch_labels in dataloader:
        batch_points, batch_labels = batch_points.to(DEVICE), batch_labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(batch_points)  # [batch_size, num_points, 2]
        
        loss = criterion(outputs.view(-1, 2), batch_labels.view(-1))  # Убираем размерность
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

# Сохранение модели
torch.save(model.state_dict(), "D:\\diplom 1 sem\\pointnet1.pth")
print("Обучение завершено, модель сохранена!")
