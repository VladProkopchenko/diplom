import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# ===================== Модель PointNetSeg =====================
class PointNetSeg(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNetSeg, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)

        self.conv5 = nn.Conv1d(512, 256, 1)
        self.bn5 = nn.BatchNorm1d(256)

        self.conv6 = nn.Conv1d(256, num_classes, 1)

    def forward(self, x):
        B, _, N = x.size()
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))

        global_feat = torch.max(x3, 2, keepdim=True)[0]
        global_feat = global_feat.repeat(1, 1, N)

        concat = torch.cat([x1, global_feat], 1)
        x = F.relu(self.bn4(self.conv4(concat)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        return x

# ===================== Датасет =====================
class BalancedToothDataset(Dataset):
    def __init__(self, dataset_dir):
        self.files = sorted([os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".npz")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        points = data["points"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        return torch.from_numpy(points), torch.from_numpy(labels)

# ===================== Основной код =====================
if __name__ == "__main__":
    DATASET_DIR = "balanced_dataset"
    BATCH_SIZE = 2
    NUM_EPOCHS = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка данных
    full_dataset = BalancedToothDataset(DATASET_DIR)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Модель, лосс, оптимизатор
    model = PointNetSeg(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Обучение
    print("[INFO] Начинается обучение модели...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for points, labels in train_loader:
            points, labels = points.to(DEVICE), labels.to(DEVICE)
            points = points.transpose(1, 2)  # (B, 3, N)

            optimizer.zero_grad()
            preds = model(points)  # (B, num_classes, N)
            preds = preds.transpose(1, 2).contiguous().reshape(-1, 2)
            labels = labels.reshape(-1)

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(DEVICE), labels.to(DEVICE)
                points = points.transpose(1, 2)
                preds = model(points)
                preds = preds.transpose(1, 2).contiguous().reshape(-1, 2)
                labels = labels.reshape(-1)
                loss = criterion(preds, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {total_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f}")

    # Сохранение
    torch.save(model.state_dict(), "pointnet_tooth_segmentation_balanced.pth")
    print("[DONE] Обучение завершено, модель сохранена.")
