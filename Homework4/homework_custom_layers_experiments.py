import torch
import torch.nn as nn
from models.custom_layers import (
    CustomConv2d, AttentionConv2d, CustomActivation, CustomPooling,
    BasicResidualBlock, BottleneckResidualBlock, WideResidualBlock
)
from utils.training_utils import train_and_evaluate
from utils.comparison_utils import prepare_cifar10, save_results


def test_custom_layers():
    print("\n=== Testing Custom Layers ===")
    custom_conv = CustomConv2d(3, 16, kernel_size=3)
    std_conv = nn.Conv2d(3, 16, kernel_size=3)
    print(f"Custom conv params: {sum(p.numel() for p in custom_conv.parameters())}")
    print(f"Standard conv params: {sum(p.numel() for p in std_conv.parameters())}")

    attention_conv = AttentionConv2d(16, 16, kernel_size=3)
    x = torch.randn(1, 16, 32, 32)
    print(f"Attention conv output shape: {attention_conv(x).shape}")

    custom_act = CustomActivation()
    x = torch.randn(5)
    print(f"Custom activation: {custom_act(x)}")

    custom_pool = CustomPooling(kernel_size=2)
    x = torch.randn(1, 3, 32, 32)
    print(f"Custom pool output shape: {custom_pool(x).shape}")


def compare_custom_layers():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_cifar10()

    models = {
        'Standard CNN': nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 10)  # 64 channels * 16x16 (32x32 с pool)
        ),
        'Custom CNN': nn.Sequential(
            CustomConv2d(3, 32, kernel_size=3),
            CustomActivation(),
            AttentionConv2d(32, 64, kernel_size=3),
            CustomPooling(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 10)  # Аналогично стандартной CNN
        )
    }

    results = {}
    for name, model in models.items():
        print(f"\nTesting {name}...")
        model = model.to(device)
        print(f"Model architecture:\n{model}")

        test_input = torch.randn(1, 3, 32, 32).to(device)
        try:
            test_output = model(test_input)
            print(f"Test forward pass successful! Output shape: {test_output.shape}")
        except Exception as e:
            print(f"Forward pass failed: {str(e)}")
            continue

        results[name] = train_and_evaluate(
            model, train_loader, test_loader,
            epochs=1, learning_rate=0.001, device=device
        )

    save_results(results, 'custom_layers_comparison')


def compare_residual_blocks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_cifar10()

    models = {
        'Basic Residual': nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            BasicResidualBlock(32, 32),
            BasicResidualBlock(32, 64, stride=2),
            BasicResidualBlock(64, 128, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10)
        ),
        'Bottleneck Residual': nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            BottleneckResidualBlock(64, 64),
            BottleneckResidualBlock(64, 128, stride=2),
            BottleneckResidualBlock(128, 256, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 10)
        ),
        'Wide Residual': nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            WideResidualBlock(32, 32, widen_factor=2),
            WideResidualBlock(32, 64, stride=2, widen_factor=2),
            WideResidualBlock(64, 128, stride=2, widen_factor=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128 * 2, 10)  # Учитываем widen_factor
        )
    }

    results = {}
    for name, model in models.items():
        print(f"\nTesting {name} network...")
        model = model.to(device)
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

        test_input = torch.randn(1, 3, 32, 32).to(device)
        try:
            test_output = model(test_input)
            print(f"Test forward pass successful! Output shape: {test_output.shape}")
        except Exception as e:
            print(f"Forward pass failed: {str(e)}")
            continue

        results[name] = train_and_evaluate(
            model, train_loader, test_loader,
            epochs=1, learning_rate=0.001, device=device
        )

    save_results(results, 'residual_blocks_comparison')

if __name__ == "__main__":
    test_custom_layers()
    compare_custom_layers()
    compare_residual_blocks()