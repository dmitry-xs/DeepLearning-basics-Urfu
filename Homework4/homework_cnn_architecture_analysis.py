import torch
import torch.nn as nn
from models.cnn_models import KernelSizeCNN, DepthCNN, ResNetLikeCNN
from utils.training_utils import train_and_evaluate
from utils.visualization_utils import plot_activations, plot_feature_maps, analyze_gradients
from utils.comparison_utils import prepare_mnist, prepare_cifar10, save_results


def analyze_kernel_sizes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_mnist()

    models = {
        '3x3 Kernels': KernelSizeCNN(kernel_sizes=[3, 3, 3]),
        '5x5 Kernels': KernelSizeCNN(kernel_sizes=[5, 5, 5]),
        '7x7 Kernels': KernelSizeCNN(kernel_sizes=[7, 7, 7]),
        'Mixed 1x1+3x3': KernelSizeCNN(kernel_sizes=[1, 3, 3])
    }

    results = {}
    for name, model in models.items():
        print(f"\nAnalyzing {name}...")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

        sample_data, _ = next(iter(train_loader))
        plot_activations(model.conv1, sample_data[0:1], name)

        results[name] = train_and_evaluate(
            model, train_loader, test_loader,
            epochs=10, learning_rate=0.001, device=device
        )

    save_results(results, 'kernel_size_analysis')
    analyze_receptive_fields(models)


def analyze_depth():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_cifar10()

    models = {
        'Shallow (2 layers)': DepthCNN(num_blocks=[1, 1]).to(device),
        'Medium (4 layers)': DepthCNN(num_blocks=[2, 2]).to(device),
        'Deep (6+ layers)': DepthCNN(num_blocks=[3, 3]).to(device),
        'ResNet-like': ResNetLikeCNN(num_blocks=[2, 2, 2]).to(device)
    }

    results = {}
    for name, model in models.items():
        print(f"\nAnalyzing {name}...")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

        results[name] = train_and_evaluate(
            model, train_loader, test_loader,
            epochs=10, learning_rate=0.001, device=device
        )

        sample_data, _ = next(iter(train_loader))
        plot_feature_maps(model, sample_data[0:1].to(device), name, device)

        analyze_gradients(model, train_loader, device, name)

    save_results(results, 'depth_analysis')


def analyze_receptive_fields(models):
    print("\nReceptive Field Analysis:")
    for name, model in models.items():
        rf = calculate_receptive_field(model)
        print(f"{name}: {rf}")


def calculate_receptive_field(model):
    rf = 1
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            k = layer.kernel_size[0]
            s = layer.stride[0]
            rf = rf * s + (k - 1)
    return rf


if __name__ == "__main__":
    print("=== Analyzing Kernel Sizes ===")
    #analyze_kernel_sizes()

    print("\n=== Analyzing Network Depth ===")
    analyze_depth()