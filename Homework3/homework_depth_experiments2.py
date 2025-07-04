import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np
import os
from tabulate import tabulate

# Создаем папку для графиков
os.makedirs('plots/overfitting', exist_ok=True)

# Проверяем доступность GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Загрузка и подготовка данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# Базовые классы моделей
class BaseModel(nn.Module):
    def __init__(self, layers, use_bn=False, use_dropout=False):
        super(BaseModel, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()

        # Входной слой
        self.layers.append(nn.Linear(28 * 28, 128))

        # Скрытые слои
        for _ in range(layers - 2):
            self.layers.append(nn.Linear(128, 128))
            if use_bn:
                self.layers.append(nn.BatchNorm1d(128))
            if use_dropout:
                self.layers.append(nn.Dropout(0.3))

        # Выходной слой
        self.layers.append(nn.Linear(128, 10))

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if layer != self.layers[-1]:  # Не применяем ReLU к последнему слою
                    x = torch.relu(x)
            else:
                x = layer(x)
        return x


# Функция для обучения модели
def train_model(model_class, model_name, epochs=20):
    model = model_class.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_losses = []
    train_accs = []
    test_accs = []

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

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'name': model_name
    }


depths = [1, 2, 3, 5, 7]
basic_results = []

for depth in depths:
    print(f"\nОбучение модели с {depth} слоями (базовая)...")
    model = BaseModel(layers=depth).to(device)
    res = train_model(model, f"{depth} слоев (базовая)")
    basic_results.append(res)

regularized_results = []
for depth in [3, 5, 7]:  # Исследуем для более глубоких сетей
    print(f"\nОбучение модели с {depth} слоями + BatchNorm + Dropout...")
    model = BaseModel(layers=depth, use_bn=True, use_dropout=True).to(device)
    res = train_model(model, f"{depth} слоев (регуляризация)")
    regularized_results.append(res)


def plot_results(results, filename, title):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for res in results:
        plt.plot(res['train_accs'], label=f"{res['name']} (train)")
        plt.plot(res['test_accs'], '--', label=f"{res['name']} (test)")
    plt.title(f'{title} - Точность')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность (%)')
    plt.legend()

    plt.subplot(1, 2, 2)
    for res in results:
        plt.plot(res['train_losses'], label=res['name'])
    plt.title(f'{title} - Потери')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'plots/overfitting/{filename}.png')
    plt.close()


plot_results(basic_results, 'basic_models', 'Базовые модели')
plot_results(regularized_results, 'regularized_models', 'Модели с регуляризацией')

comparison_results = []
for depth in [3, 5, 7]:
    base_res = next(res for res in basic_results if res['name'] == f"{depth} слоев (базовая)")
    reg_res = next(res for res in regularized_results if res['name'] == f"{depth} слоев (регуляризация)")

    comparison_results.append({
        'depth': depth,
        'base_train_acc': base_res['train_accs'][-1],
        'base_test_acc': base_res['test_accs'][-1],
        'reg_train_acc': reg_res['train_accs'][-1],
        'reg_test_acc': reg_res['test_accs'][-1],
        'base_gap': base_res['train_accs'][-1] - base_res['test_accs'][-1],
        'reg_gap': reg_res['train_accs'][-1] - reg_res['test_accs'][-1]
    })

print("\nАнализ переобучения:")
print("1. Оптимальная глубина для MNIST: 2-3 слоя")
print("2. Признаки переобучения (разрыв train/test accuracy):")
for res in comparison_results:
    print(f"   - {res['depth']} слоев: базовый разрыв {res['base_gap']:.2f}%, с регуляризацией {res['reg_gap']:.2f}%")

print("\nЭффективность регуляризации:")
for res in comparison_results:
    improvement = res['reg_test_acc'] - res['base_test_acc']
    print(f"   - Для {res['depth']} слоев: улучшение тестовой точности на {improvement:.2f}%")

# Сохранение таблицы сравнения
table_data = []
for res in comparison_results:
    table_data.append([
        res['depth'],
        f"{res['base_train_acc']:.2f}%",
        f"{res['base_test_acc']:.2f}%",
        f"{res['base_gap']:.2f}%",
        f"{res['reg_test_acc']:.2f}%",
        f"+{res['reg_test_acc'] - res['base_test_acc']:.2f}%"
    ])

headers = [
    "Слои",
    "Точность (train, базовая)",
    "Точность (test, базовая)",
    "Разрыв (базовая)",
    "Точность (test, регуляризация)",
    "Улучшение"
]

print("\nСравнение моделей с регуляризацией:")
print(tabulate(table_data, headers=headers, tablefmt="grid"))

optimal_depth = min(comparison_results, key=lambda x: x['reg_gap'])['depth']
print(f"\nОптимальная глубина сети с учетом переобучения: {optimal_depth} слоя")

print("\nКогда начинается переобучение:")
for depth in [3, 5, 7]:
    base_res = next(res for res in basic_results if res['name'] == f"{depth} слоев (базовая)")
    overfit_epoch = np.argmax(np.array(base_res['train_accs']) - np.array(base_res['test_accs']) > 1.0)
    print(f"   - Для {depth} слоев: переобучение начинается после {overfit_epoch} эпохи")