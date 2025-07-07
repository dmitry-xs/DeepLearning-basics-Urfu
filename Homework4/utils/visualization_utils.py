import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

os.makedirs('plots', exist_ok=True)


def plot_learning_curves(train_losses, train_accs, test_accs, model_name):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title(f'{model_name} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()

    # Сохраняем график вместо показа
    filename = f"plots/{model_name.lower().replace(' ', '_')}_learning_curves.png"
    plt.savefig(filename)
    plt.close()



def plot_confusion_matrix(model, test_loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Сохраняем матрицу ошибок
    model_name = model.__class__.__name__
    filename = f"plots/{model_name.lower()}_confusion_matrix.png"
    plt.savefig(filename)
    plt.close()


def analyze_gradients(model, train_loader, device, model_name):
    model.train()
    gradients = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        model.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        # Собираем средние градиенты по слоям
        layer_grads = []
        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None:
                layer_grads.append(param.grad.abs().mean().item())
        gradients.append(np.mean(layer_grads))

    plt.figure()
    plt.plot(gradients)
    plt.xlabel('Batch')
    plt.ylabel('Average Gradient Magnitude')
    plt.title(f'{model_name} - Gradient Flow')
    save_plot(f"{model_name.lower().replace(' ', '_')}_gradients.png")


def save_plot(filename):
    os.makedirs('plots', exist_ok=True)
    plt.savefig(os.path.join('plots', filename))
    plt.close()


def plot_activations(layer, input_tensor, model_name):
    with torch.no_grad():
        activations = layer(input_tensor)

    plt.figure(figsize=(12, 6))
    for i in range(min(16, activations.shape[1])):  # Показываем первые 16 карт активаций
        plt.subplot(4, 4, i + 1)
        plt.imshow(activations[0, i].cpu().numpy(), cmap='viridis')
        plt.axis('off')
    plt.suptitle(f'{model_name} - First Layer Activations')
    save_plot(f"{model_name.lower().replace(' ', '_')}_activations.png")


def plot_feature_maps(model, input_tensor, model_name, device):
    # Перемещаем модель и входные данные на одно устройство
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(get_activation(name)))

    with torch.no_grad():
        _ = model(input_tensor)

    for hook in hooks:
        hook.remove()

    for name, act in activations.items():
        plt.figure(figsize=(12, 6))
        act = act.cpu()
        for i in range(min(16, act.shape[1])):
            plt.subplot(4, 4, i + 1)
            plt.imshow(act[0, i].numpy(), cmap='viridis')
            plt.axis('off')
        plt.suptitle(f'{model_name} - {name} Feature Maps')
        save_plot(f"{model_name.lower().replace(' ', '_')}_{name}_feature_maps.png")





