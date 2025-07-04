import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from tabulate import tabulate


os.makedirs('plots/width', exist_ok=True)

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

class WidthModel(nn.Module):
    def __init__(self, layer_sizes):
        super(WidthModel, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        input_size = 28 * 28

        for size in layer_sizes:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size

        self.output = nn.Linear(input_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


width_configs = {
    "Узкие слои": [64, 32, 16],
    "Средние слои": [256, 128, 64],
    "Широкие слои": [1024, 512, 256],
    "Очень широкие слои": [2048, 1024, 512],
    "Расширяющиеся слои": [64, 128, 256],
    "Сужающиеся слои": [256, 128, 64],
    "Постоянная ширина": [256, 256, 256]
}


def train_model(model, model_name, epochs=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_losses = []
    train_accs = []
    test_accs = []
    start_time = time.time()

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

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

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

        print(f"Модель {model_name}, Эпоха {epoch + 1}/{epochs}, "
              f"Потери: {train_loss:.4f}, Точность (train): {train_acc:.2f}%, "
              f"Точность (test): {test_acc:.2f}%")

    total_time = time.time() - start_time
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'time': total_time,
        'params': model.count_params()
    }


results = {}
for name, sizes in width_configs.items():
    print(f"\nОбучение модели: {name} {sizes}...")
    model = WidthModel(sizes).to(device)
    results[name] = train_model(model, name)


plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
for name, res in results.items():
    plt.plot(res['train_accs'], label=name)
plt.title('Точность на обучающей выборке')
plt.xlabel('Эпоха')
plt.ylabel('Точность (%)')
plt.legend()

# График точности на тестовой выборке
plt.subplot(2, 2, 2)
for name, res in results.items():
    plt.plot(res['test_accs'], label=name)
plt.title('Точность на тестовой выборке')
plt.xlabel('Эпоха')
plt.ylabel('Точность (%)')
plt.legend()

# График времени обучения
plt.subplot(2, 2, 3)
times = [res['time'] for res in results.values()]
names = list(results.keys())
plt.bar(names, times)
plt.title('Время обучения')
plt.ylabel('Время (с)')
plt.xticks(rotation=45)

# Heatmap: точность на тесте по типам архитектур
plt.subplot(2, 2, 4)
test_accs = [[res['test_accs'][-1]] for res in results.values()]
sns.heatmap(test_accs, annot=True, yticklabels=results.keys(), xticklabels=["Test Accuracy"], cmap="Blues")
plt.title("Точность на тесте по архитектурам")

plt.tight_layout()
plt.savefig('plots/width/results_heatmap.png')
plt.show()


table_data = []
for name, res in results.items():
    table_data.append([
        name,
        f"{res['train_accs'][-1]:.2f}%",
        f"{res['test_accs'][-1]:.2f}%",
        f"{res['time']:.2f}с",
        f"{res['params']:,}".replace(",", " ")
    ])

headers = ["Конфигурация", "Точность (train)", "Точность (test)", "Время обучения", "Параметры"]
print("\nСравнительная таблица результатов:")
print(tabulate(table_data, headers=headers, tablefmt="grid"))