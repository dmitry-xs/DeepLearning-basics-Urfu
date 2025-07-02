import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product


from homework_model_modification import LinearRegression, LogisticRegression
from homework_datasets import CSVDataset

os.makedirs("plots", exist_ok=True)



def run_regression_experiment(lr, batch_size, optimizer_name):
    dataset = CSVDataset(
        file_path='data/WineQT.csv',
        target_column='quality',
        numeric_cols=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                      'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                      'pH', 'sulphates', 'alcohol'],
        normalize_numeric=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LinearRegression(in_features=dataset.get_feature_dim())
    criterion = nn.MSELoss()
    optimizer = None
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    epochs = 50
    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

    # Сохраняем график потерь
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label=f"LR={lr}, BS={batch_size}, OPT={optimizer_name}")
    plt.title(f"Regression Loss - {optimizer_name}, LR={lr}, BS={batch_size}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/regression_{optimizer_name}_lr{lr}_bs{batch_size}.png")
    plt.close()

    return {
        "type": "regression",
        "lr": lr,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "final_loss": losses[-1]
    }


def run_classification_experiment(lr, batch_size, optimizer_name):
    dataset = CSVDataset(
        file_path='data/titanic.csv',
        target_column='Survived',
        numeric_cols=['Age', 'SibSp', 'Parch', 'Fare'],
        categorical_cols=['Pclass', 'Embarked'],
        binary_cols=['Sex'],
        normalize_numeric=True,
        test_size=0.2,
        random_state=42,
        mode='train'
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LogisticRegression(in_features=dataset.get_feature_dim(), num_classes=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = None
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    losses = []
    accs = []

    for epoch in range(50):
        model.train()
        total_loss = 0
        total_acc = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                acc = (preds == y_batch).float().mean()

                total_loss += loss.item()
                total_acc += acc.item()

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        losses.append(avg_loss)
        accs.append(avg_acc)

    # Сохраняем график потерь
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label=f"LR={lr}, BS={batch_size}, OPT={optimizer_name}")
    plt.title(f"Classification Loss - {optimizer_name}, LR={lr}, BS={batch_size}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/classification_{optimizer_name}_lr{lr}_bs{batch_size}.png")
    plt.close()

    return {
        "type": "classification",
        "lr": lr,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "final_loss": losses[-1],
        "final_acc": accs[-1]
    }


def run_experiments():
    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [8, 32, 128]
    optimizers = ["SGD", "Adam", "RMSprop"]

    results = []

    print("Запускаем эксперименты для регрессии...")
    for lr, bs, opt in product(learning_rates, batch_sizes, optimizers):
        result = run_regression_experiment(lr, bs, opt)
        results.append(result)

    print("Запускаем эксперименты для классификации...")
    for lr, bs, opt in product(learning_rates, batch_sizes, optimizers):
        result = run_classification_experiment(lr, bs, opt)
        results.append(result)

    # Сохраняем результаты в таблицу
    df = pd.DataFrame(results)
    df.to_csv("plots/experiment_results.csv", index=False)
    print("Все эксперименты завершены. Результаты сохранены в 'plots/experiment_results.csv'")
    print("Графики сохранены в папку 'plots/'")


if __name__ == "__main__":
    run_experiments()
