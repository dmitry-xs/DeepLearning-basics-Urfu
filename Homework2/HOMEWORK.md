# Домашнее задание к уроку 2: Линейная и логистическая регрессия
## Задание 1: Модификация существующих моделей
### 1.1 Расширение линейной регрессии
Расширим класс модели линейной регрессии дополнив его методами подсчета регуляризаций.
Вычисляются по следующим формулам 
**L1 = L + m∑|wi|**
**L2 = L + ρ∑wi^2**

```python
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
```

Теперь добавим раннюю остановку.
Ранняя остановка работает следующим образом:

    Отслеживание лучшего результата – На каждой эпохе сравнивается текущая ошибка с лучшей. Если ошибка уменьшилась, модель сохраняется и  счётчик no_improvement сбрасывается.

    Если улучшения нет, счётчик увеличивается. Когда он достигает заданного значения patience (например, 5), обучение останавливается.    

Это предотвращает переобучение и экономит время, прекращая обучение, когда модель перестаёт улучшаться.

```python
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement = 0
            torch.save(model.state_dict(), 'wine_regression_best.pth')
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f'Early stopping на эпохе {epoch}')
            break
```

### 1.2 Расширение логистической регрессии
Поддержка многоклассовой классификации:
```python
  class LogisticRegression(nn.Module):
    '''Модель логистической регрессии'''
    def __init__(self, in_features, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)
```
Реализация метрик по следующим формулам
**Precision = TP / (TP + FP)**
**Recall = TP / (TP + FN)**
**F1 = 2 * (precision * recall) / (precision + recall)**

```python
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
```

##Задание 2: Работа с датасетами 

Для задачи классификации был взят датасет titanic
Для задачи регрессии Wine Quality Dataset

Результаты обучения моделей:
1) Линейная регрессия. Остановка была произведена на 20 эпохе т.к. модель перестала обучаться
```
Epoch 16, Loss: 0.4305
Epoch 17, Loss: 0.4284
Epoch 18, Loss: 0.4299
Epoch 19, Loss: 0.4282
Early stopping на эпохе 20
```
2) Логистическая регрессия:
```
Epoch 49, Loss: 0.4528, Accuracy: 0.8083
Epoch 50, Loss: 0.4423, Accuracy: 0.8124
Accuracy: 0.7744
Precision: 0.8280
Recall: 0.5131
F1-score: 0.6336
```
## Задание 3: Эксперименты и анализ
### 3.1 Исследование гиперпараметров 
При экспериментах использовались следующие параметры: 

learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [8, 32, 128]
optimizers = ["SGD", "Adam", "RMSprop"]

Результаты:

```
              type     lr  batch_size optimizer  final_loss  final_acc
0       regression  0.100           8       SGD    0.522911        NaN
1       regression  0.100           8      Adam    0.770364        NaN
2       regression  0.100           8   RMSprop    0.713304        NaN
3       regression  0.100          32       SGD    0.429266        NaN
4       regression  0.100          32      Adam    0.462481        NaN
5       regression  0.100          32   RMSprop    0.535668        NaN
6       regression  0.100         128       SGD    0.396425        NaN
7       regression  0.100         128      Adam    0.388647        NaN
8       regression  0.100         128   RMSprop    0.446513        NaN
9       regression  0.010           8       SGD    0.419561        NaN
10      regression  0.010           8      Adam    0.409828        NaN
11      regression  0.010           8   RMSprop    0.418112        NaN
12      regression  0.010          32       SGD    0.407569        NaN
13      regression  0.010          32      Adam    0.415841        NaN
14      regression  0.010          32   RMSprop    0.408927        NaN
15      regression  0.010         128       SGD    0.393185        NaN
16      regression  0.010         128      Adam    6.796136        NaN
17      regression  0.010         128   RMSprop    2.315570        NaN
18      regression  0.001           8       SGD    0.407603        NaN
19      regression  0.001           8      Adam    1.616084        NaN
20      regression  0.001           8   RMSprop    0.462872        NaN
21      regression  0.001          32       SGD    0.501181        NaN
22      regression  0.001          32      Adam   19.620490        NaN
23      regression  0.001          32   RMSprop   15.431265        NaN
24      regression  0.001         128       SGD    7.306621        NaN
25      regression  0.001         128      Adam   31.032434        NaN
26      regression  0.001         128   RMSprop   26.423528        NaN
27  classification  0.100           8       SGD    0.465604   0.781944
28  classification  0.100           8      Adam    0.518373   0.769444
29  classification  0.100           8   RMSprop    0.485713   0.773611
30  classification  0.100          32       SGD    0.459368   0.802083
31  classification  0.100          32      Adam    0.448582   0.796800
32  classification  0.100          32   RMSprop    0.432694   0.808424
33  classification  0.100         128       SGD    0.475752   0.781839
34  classification  0.100         128      Adam    0.444633   0.801049
35  classification  0.100         128   RMSprop    0.454704   0.801370
36  classification  0.010           8       SGD    0.461387   0.783333
37  classification  0.010           8      Adam    0.423508   0.819444
38  classification  0.010           8   RMSprop    0.446057   0.804167
39  classification  0.010          32       SGD    0.531490   0.725393
40  classification  0.010          32      Adam    0.451463   0.798007
41  classification  0.010          32   RMSprop    0.443437   0.807669
42  classification  0.010         128       SGD    0.598911   0.674693
43  classification  0.010         128      Adam    0.463998   0.805615
44  classification  0.010         128   RMSprop    0.446624   0.805276
45  classification  0.001           8       SGD    0.593348   0.672222
46  classification  0.001           8      Adam    0.482366   0.787500
47  classification  0.001           8   RMSprop    0.473872   0.804167
48  classification  0.001          32       SGD    0.560153   0.716033
49  classification  0.001          32      Adam    0.515822   0.747283
50  classification  0.001          32   RMSprop    0.530101   0.727959
51  classification  0.001         128       SGD    0.765280   0.386201
52  classification  0.001         128      Adam    0.607339   0.689676
53  classification  0.001         128   RMSprop    0.541569   0.731361
```

Из таблицы видно, что для регрессионных задач наилучшие результаты достигаются при использовании оптимизатора SGD с learning rate 0.01 и batch size 128 (final_loss = 0.393), а также при комбинации Adam с теми же параметрами (final_loss = 0.388). 
В классификационных задачах оптимальные параметры зависят от метрики: максимальная точность (final_acc = 0.819) достигается при Adam с lr=0.01 и batch_size=8, но баланс между точностью и стабильностью наблюдается у RMSprop с lr=0.1 и batch_size=32 (final_acc = 0.808, final_loss = 0.432).

Критически плохие результаты (final_loss > 15) возникают в регрессии при малых learning rate (0.001) в сочетании с Adam/RMSprop и большими batch size, что указывает на проблему сходимости. В классификации аналогичная тенденция: при lr=0.001 и batch_size=128 все оптимизаторы показывают худшие результаты.


