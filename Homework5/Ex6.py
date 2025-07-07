import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from datasets import CustomImageDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Конфигурация
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-5
MODEL_NAME = 'resnet18'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Подготовка трансформаций
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Загрузка данных
train_dataset = CustomImageDataset('data/train', transform=train_transform)
val_dataset = CustomImageDataset('data/test', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# Инициализация модели
def initialize_model(model_name, num_classes):
    if model_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model.to(DEVICE)  # Переносим всю модель на устройство (GPU/CPU)


model = initialize_model(MODEL_NAME, len(train_dataset.get_class_names()))
print(f"Model loaded on {next(model.parameters()).device}")

# Функции для обучения и валидации


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc='Training'):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Переносим данные на устройство

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validating'):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Переносим данные на устройство

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# Инициализация
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss().to(DEVICE)  # Переносим функцию потерь на устройство

# Для хранения истории
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

# Обучение
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')

    # Обучение
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)

    # Валидация
    val_loss, val_acc = validate(model, val_loader, criterion)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')

# Визуализация
os.makedirs('results/training_results', exist_ok=True)

plt.figure(figsize=(12, 6))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_results/loss_curve.png', dpi=150)
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('training_results/accuracy_curve.png', dpi=150)
plt.close()

torch.save(model.state_dict(), 'results/training_results/model_weights.pth')
print("Обучение завершено. Результаты сохранены в папке 'training_results'")