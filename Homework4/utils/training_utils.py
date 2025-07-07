import torch
import time
import numpy as np



def train_model(model, train_loader, criterion, optimizer, device):
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
    return train_loss, train_acc


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def measure_inference_time(model, test_loader, device, n_runs=10):
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start_time = time.time()
            for images, _ in test_loader:
                images = images.to(device)
                _ = model(images)
            times.append(time.time() - start_time)
    return np.mean(times)


def train_and_evaluate(model, train_loader, test_loader, epochs, learning_rate, device):
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accs = []
    test_accs = []
    times = []

    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate_model(model, test_loader, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        epoch_time = time.time() - epoch_start
        times.append(epoch_time)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, '
              f'Time: {epoch_time:.2f}s')

    total_time = time.time() - start_time
    avg_epoch_time = np.mean(times)
    inference_time = measure_inference_time(model, test_loader, device)

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'total_time': total_time,
        'avg_epoch_time': avg_epoch_time,
        'inference_time': inference_time,
        'num_params': sum(p.numel() for p in model.parameters())
    }