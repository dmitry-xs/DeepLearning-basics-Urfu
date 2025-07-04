import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from tabulate import tabulate

os.makedirs('plots/regularization', exist_ok=True)

# Проверяем доступность GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Загрузка данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


class BaseNet(nn.Module):
    def __init__(self, dropout_rate=0.0, use_batchnorm=False, use_l2=False):
        super(BaseNet, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_l2 = use_l2

        self.fc1 = nn.Linear(28 * 28, 512)
        self.bn1 = nn.BatchNorm1d(512) if use_batchnorm else None
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256) if use_batchnorm else None
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        if self.bn1:
            x = self.bn1(x)
        if self.dropout1:
            x = self.dropout1(x)

        x = torch.relu(self.fc2(x))
        if self.bn2:
            x = self.bn2(x)
        if self.dropout2:
            x = self.dropout2(x)

        return self.fc3(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())



configs = {
    "Без регуляризации": {"dropout_rate": 0.0, "use_batchnorm": False, "weight_decay": 0.0},
    "Dropout 0.1": {"dropout_rate": 0.1, "use_batchnorm": False, "weight_decay": 0.0},
    "Dropout 0.3": {"dropout_rate": 0.3, "use_batchnorm": False, "weight_decay": 0.0},
    "Dropout 0.5": {"dropout_rate": 0.5, "use_batchnorm": False, "weight_decay": 0.0},
    "BatchNorm": {"dropout_rate": 0.0, "use_batchnorm": True, "weight_decay": 0.0},
    "Dropout+BatchNorm": {"dropout_rate": 0.5, "use_batchnorm": True, "weight_decay": 0.0},
    "L2 регуляризация": {"dropout_rate": 0.0, "use_batchnorm": False, "weight_decay": 1e-4}
}

results = {}
epochs = 15


def train_model(config_name, config):
    model = BaseNet(
        dropout_rate=config["dropout_rate"],
        use_batchnorm=config["use_batchnorm"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_accs = []
    start_time = time.time()

    print(f"\nОбучение: {config_name}...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        train_acc = 100 * correct / total

        # Тестирование
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total
        test_accs.append(test_acc)

        print(f"Эпоха {epoch + 1}/{epochs}, "
              f"Потери: {avg_loss:.4f}, "
              f"Точность (train): {train_acc:.2f}%, "
              f"Точность (test): {test_acc:.2f}%")

    total_time = time.time() - start_time
    return {
        'model': model,
        'losses': train_losses,
        'test_accs': test_accs,
        'time': total_time,
        'params': model.count_params()
    }


for name, config in configs.items():
    results[name] = train_model(name, config)

# График потерь
plt.figure(figsize=(12, 5))
for name, res in results.items():
    plt.plot(res['losses'], label=name)
plt.title("Функция потерь на обучающей выборке")
plt.xlabel("Эпохи")
plt.ylabel("Потери")
plt.legend()
plt.grid(True)
plt.savefig('plots/regularization/losses.png')


# График точности на тесте
plt.figure(figsize=(12, 5))
for name, res in results.items():
    plt.plot(res['test_accs'], label=name)
plt.title("Точность на тестовой выборке")
plt.xlabel("Эпохи")
plt.ylabel("Точность (%)")
plt.legend()
plt.grid(True)
plt.savefig('plots/regularization/test_accuracy.png')



# Распределение весов первого слоя
plt.figure(figsize=(14, 6))
i = 1
for name, res in results.items():
    weights = res['model'].fc1.weight.data.cpu().numpy().flatten()
    plt.subplot(2, 4, i)
    plt.hist(weights, bins=50, alpha=0.7)
    plt.title(name)
    plt.xlabel("Веса")
    plt.ylabel("Частота")
    i += 1

plt.tight_layout()
plt.savefig('plots/regularization/weight_distributions.png')


table_data = []
for name, res in results.items():
    table_data.append([
        name,
        f"{res['test_accs'][-1]:.2f}%",
        f"{res['time']:.2f}с"
    ])

headers = ["Регуляризация", "Точность на тесте", "Время обучения"]
print("\nСравнительная таблица регуляризаций:")
print(tabulate(table_data, headers=headers, tablefmt="grid"))