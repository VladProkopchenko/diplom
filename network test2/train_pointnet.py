import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import gc
from tqdm import tqdm

class ToothDataset(Dataset):
    def __init__(self, dataset_folder, num_points=200000):
        self.dataset_folder = dataset_folder
        self.files = sorted([f for f in os.listdir(dataset_folder) if f.endswith(".npz")])
        self.num_points = num_points
        print(f"[Dataset] Найдено {len(self.files)} файлов в папке {dataset_folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.dataset_folder, self.files[idx])
        data = np.load(file_path)
        points = data["points"]
        labels = data["labels"]

        if len(points) > self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
            points = points[choice]
            labels = labels[choice]
        else:
            pad = self.num_points - len(points)
            points = np.pad(points, ((0, pad), (0, 0)), mode="constant")
            labels = np.pad(labels, (0, pad), mode="constant")

        points = torch.tensor(points, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return points, labels

class PointNetSeg(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNetSeg, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, num_classes)

    def forward(self, x):
        B, N, D = x.size()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.max(x, 1, keepdim=True)[0]
        x = x.repeat(1, N, 1)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Используем устройство: {device}")

    dataset = ToothDataset("dataset_parts", num_points=200000)

    val_split = 0.2
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"[Split] Тренировочных: {train_size}, Валидационных: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = PointNetSeg().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    EPOCHS = 30

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")

        for i, (points, labels) in enumerate(train_loader):
            points = points.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(points)
            preds = preds.permute(0, 2, 1)
            loss = F.cross_entropy(preds, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            del points, labels, preds
            gc.collect()
            torch.cuda.empty_cache()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for points, labels in val_loader:
                points = points.to(device)
                labels = labels.to(device)
                preds = model(points)
                preds = preds.permute(0, 2, 1)
                loss = F.cross_entropy(preds, labels)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "pointnet_tooth_segmentation_best.pth")
            print("[Сохранено] Лучшая модель сохранена в pointnet_tooth_segmentation_best.pth")

    print("[Готово] Обучение завершено.")

if __name__ == "__main__":
    train()