import torch
from models.fc_models import FCNet, DeepFCNet
from models.cnn_models import SimpleCNN, ResCNN, CifarResCNN, RegularizedResCNN
from utils.training_utils import train_and_evaluate
from utils.visualization_utils import plot_learning_curves, plot_confusion_matrix, analyze_gradients
from utils.comparison_utils import prepare_mnist, prepare_cifar10, save_results, print_comparison


def compare_mnist():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_mnist()

    models = {
        'FC Network': FCNet(),
        'Simple CNN': SimpleCNN(),
        'CNN with Residual': ResCNN()
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        results[name] = train_and_evaluate(
            model, train_loader, test_loader,
            epochs=10, learning_rate=0.001, device=device
        )
        plot_learning_curves(
            results[name]['train_losses'],
            results[name]['train_accs'],
            results[name]['test_accs'],
            name
        )

    save_results(results, 'mnist_comparison')
    print_comparison(results)


def compare_cifar10():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_cifar10()
    cifar_classes = ('plane', 'car', 'bird', 'cat', 'deer',
                     'dog', 'frog', 'horse', 'ship', 'truck')

    models = {
        'Deep FC Network': DeepFCNet(),
        'CNN with Residual': CifarResCNN(),
        'Regularized ResCNN': RegularizedResCNN()
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        results[name] = train_and_evaluate(
            model, train_loader, test_loader,
            epochs=10, learning_rate=0.001, device=device
        )
        plot_learning_curves(
            results[name]['train_losses'],
            results[name]['train_accs'],
            results[name]['test_accs'],
            name
        )
        plot_confusion_matrix(model, test_loader, device, cifar_classes)
        analyze_gradients(model, train_loader, device)

    save_results(results, 'cifar_comparison')
    print_comparison(results)


if __name__ == "__main__":
    print("Comparing models on MNIST:")
    compare_mnist()

    print("\nComparing models on CIFAR-10:")
    compare_cifar10()