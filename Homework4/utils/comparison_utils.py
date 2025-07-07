import os
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def prepare_mnist(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def prepare_cifar10(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def save_results(results, filename):
    os.makedirs('results', exist_ok=True)
    with open(f'results/{filename}.json', 'w') as f:
        json.dump(results, f, indent=4)


def print_comparison(results):
    print("\nComparison Results:")
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"Final Train Accuracy: {res['train_accs'][-1]:.2f}%")
        print(f"Final Test Accuracy: {res['test_accs'][-1]:.2f}%")
        print(f"Total Training Time: {res['total_time']:.2f}s")
        print(f"Average Epoch Time: {res['avg_epoch_time']:.2f}s")
        print(f"Inference Time: {res['inference_time']:.2f}s")
        print(f"Number of Parameters: {res['num_params']}")

