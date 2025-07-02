import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from homework_datasets import CSVDataset


class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

    def l1_regularization(self, m):
        L1 = 0.0
        for w in self.parameters():
            L1 += m * torch.sum(torch.abs(w))
        return L1

    def l2_regularization(self, p):
        L2 = 0.0
        for w in self.parameters():
            L2 += p * torch.sum(w ** 2)
        return L2



class LogisticRegression(nn.Module):
    '''Модель логистической регрессии'''
    def __init__(self, in_features, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)



def accuracy(y_pred, y_true):
    if y_pred.shape[1] > 1:
        preds = y_pred.argmax(dim=1)
    else:
        preds = (y_pred > 0.5).float()
    return (preds == y_true).float().mean()


def precision(y_true, y_pred):
    true_pos = ((y_pred == 1) & (y_true == 1)).sum()
    pred_pos = (y_pred == 1).sum()
    return true_pos.float() / (pred_pos.float() + 1e-7)


def recall(y_true, y_pred):
    true_pos = ((y_pred == 1) & (y_true == 1)).sum()
    actual_pos = (y_true == 1).sum()
    return true_pos.float() / (actual_pos.float() + 1e-7)


def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec + 1e-7)


def plot_confusion_matrix(y_true, y_pred, num_classes):
    cm = torch.zeros(num_classes, num_classes)
    for t, p in zip(y_true, y_pred):
        cm[t.long(), p.long()] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(cm.numpy(), cmap='Blues')
    plt.colorbar()

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", color="red")

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(range(num_classes))
    plt.yticks(range(num_classes))
    plt.savefig('confusion_matrix.png')
    plt.close()



def train_wine_regression():

    dataset = CSVDataset(
        file_path='data/WineQT.csv',
        target_column='quality',
        numeric_cols=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                      'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                      'pH', 'sulphates', 'alcohol'],
        normalize_numeric=True
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Создаем модель
    model = LinearRegression(in_features=dataset.get_feature_dim())
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Параметры обучения
    epochs = 100
    best_loss = float('inf')
    patience = 5
    no_improvement = 0

    # Коэффициенты регуляризации
    m = 0.001
    p = 0.001

    # Обучение
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(batch_X)

            loss = criterion(y_pred, batch_y)
            loss += model.l1_regularization(m)
            loss += model.l2_regularization(p)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (i + 1)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement = 0
            torch.save(model.state_dict(), 'wine_regression_best.pth')
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f'Early stopping на эпохе {epoch}')
            break

        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

    print("Обучение линейной регрессии на WineQT завершено")



def train_credit_classification():
    # Загрузка и подготовка данных
    dataset = CSVDataset(
        file_path='data/titanic.csv',
        target_column='Survived',
        numeric_cols=['Age', 'SibSp', 'Parch', 'Fare'],
        categorical_cols=['Pclass', 'Embarked'],
        binary_cols=['Sex'],
        normalize_numeric=True,
        test_size=0.2,
        random_state=42,
        mode='train'  )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    print(dataset.data['Survived'].value_counts())
    # Создаем модель
    model = LogisticRegression(in_features=dataset.get_feature_dim(), num_classes=1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Для хранения метрик
    all_labels = []
    all_predictions = []
    all_probs = []

    # Обучение
    epochs = 50
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_acc = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(batch_X)

            # Исправление размерностей:
            # Убираем лишнюю размерность у целевых значений
            batch_y = batch_y.float().squeeze()

            loss = criterion(outputs.squeeze(), batch_y)  # Убираем размерность и у выхода модели
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                y_pred = (probs > 0.5).float()
                acc = (y_pred.squeeze() == batch_y).float().mean()

                total_loss += loss.item()
                total_acc += acc.item()

                all_labels.append(batch_y)
                all_predictions.append(y_pred.squeeze())
                all_probs.append(probs.squeeze())

        avg_loss = total_loss / (i + 1)
        avg_acc = total_acc / (i + 1)

        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}')

    # Объединяем все батчи
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)
    all_probs = torch.cat(all_probs)

    print(f"Accuracy: {(all_predictions == all_labels).float().mean():.4f}")
    print(f"Precision: {precision(all_labels, all_predictions):.4f}")
    print(f"Recall: {recall(all_labels, all_predictions):.4f}")
    print(f"F1-score: {f1_score(all_labels, all_predictions):.4f}")

    plot_confusion_matrix(all_labels, all_predictions, num_classes=2)

    # Сохраняем модель
    torch.save(model.state_dict(), 'credit_classification.pth')
    print("Обучение завершено")


if __name__ == '__main__':
    print("Обучение линейной регрессии на WineQT")
    train_wine_regression()

    print("\nОбучение логистической регрессии на Titanic")
    train_credit_classification()