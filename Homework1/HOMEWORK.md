# Домашняя работа №1
## Задание 1: Создание и манипуляции с тензорами
### 1.1 Создание тензоров
```
1. Для заполнения тензора рандомными числами от 0 до 1 используем метод rand модуля torch 
2. При создании тензора заполненого нулями используем torch.zeros, в параметрах через запятую указываем размерность тензора 
3. Для созжания тензора из единиц используем метод ones      
4. И для тензора заполненого последовательностью чисел от 0 до 15 используем arange, в параметрах указываем 0 и 16 т.к последнее число не учитывается при генерации.
```
### 1.2 Операции с тензорами
Создадим два тензора А и В и заполним из наприимер значениями от 0 до 12.
```
1. Для транспонирования тензора А используем А.Т
2. Для матричного умножения двух тензоров используем @
3. Для того чтобы поэлементно умножить два тензора используем привычный нам символ *
4. Для того чтобы посчитать сумму значений из тензора, воспользуемся методом sum из модуля торч, в и результате получаем тензор со значением 66, и с помощью item() возвращаем только сумму.
```
### 1.3 Индексация и срезы
Для начала сосздадим тензор размером 5х5х5, для этого воспользуемся torch.randint. Будем генерировать целые числа от 0 до 9. 
```
1. Возьмем первую строку (первой матрицы). Для этого выполним срез tensor[0, 0, :], где указываем, что берем нулевую матрицу, нудевую строку, и указываем все элементы в первой строке.
2. Чтобы взять последний столбец (например всех матриц) сделаем срез tensor[:, :, -1], где указываем, что берем все матрицы, все строки, и указываем с каждой строки последний элемент.
3. Для нахождения центральной подматрицы, сделаем срез tensor[2:4, 2:4, 2:4]. Срезы делаем по всем трем осям, чтобы получить центральный куб 2×2×2.
4. И чтобы извлечь все элементы с четными индексами, сделаем срез tensor[::2, ::2, ::2], то есть по каждой оси делаем шаг 2, это позволит проскочить нечетные индексы.
```
### 1.4 Работа с формами
Для изменения форм тензора можем воспользовать как reshape, так и view
## Задание 2: Автоматическое дифференцирование
### 2.1 Простые вычисления с градиентами
Вызываем метод backward(), после определния функции, он автоматически вычисляет градиенты. Затем принтуем градиенты по всем переменным, используя grad.item().
```python
f.backward()

print(f"Градиент по x: {x.grad.item()}")
print(f"Градиент по y: {y.grad.item()}")
print(f"Градиент по z: {z.grad.item()}")
```
Для аналитической проверки, создадим функцию analytical_gradients(x, y, z), она просто возвращает частные производные по всем переменным. 
```python
def analytical_gradients(x, y, z):
    """Данной функцией аналитически проверим подсчет градиентов"""
    df_dx = 2*x + 2*y*z
    df_dy = 2*y + 2*x*z
    df_dz = 2*z + 2*x*y
    return df_dx, df_dy, df_dz
```
В итоге градиенты, при значениях 1.0, 2.0, 3.0 в обоих случаях вышли одинаковыми.
```
Градиент по x: 14.0
Градиент по y: 10.0
Градиент по z: 10.0

df/dx = 14
df/dy = 10
df/dz = 10
```
### 2.2 Градиент функции потерь
Определим функцию MSE
```python
def MSE(y_pred, y_true):
    """Данная функция считает среднеквадратичную ошибку"""
    return ((y_pred - y_true)**2).mean()
```
Также объявим X и y. Затем инициализируем параметры модели w, b. После чего объявляем модель и вычисляем потери. И в конце вычисляем градиенты.
```python
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32) # y = 2x

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

y_pred = X*w + b

mse = MSE(y_pred, y)
mse.backward()

print(f'\nГрадиент по w: {w.grad.item()}')
print(f'Градиент по b: {b.grad.item()}')
```
### 2.3 Цепное правило
Для начала напишем составную функцию из условий задания
```python
def compos_func(x: torch.Tensor):
    return torch.sin(x**2 + 1)
```
объявим х, у и вычислим градиенты.
```python
X2 = torch.tensor(2.0, requires_grad=True)

y2 = compos_func(X2)
y2.backward()

grad = X2.grad.item()
print(f'\nГрадиент = {grad:.4f}')
```
И теперь автоматически посчитаем градиент
```python
torch_autograd = torch.autograd.grad(compos_func(X2), X2)[0].item()
print(f'Автоградиент = {torch_autograd:.4f}')
```
В итоге получаем 2 одинаковых ответа
```
Градиент = 1.1346
Автоградиент = 1.1346
```
## Задание 3: Сравнение производительности CPU vs CUDA
При выполнении данного задания вышли следующие результаты.

```
Выполнение операций над матрицами размера 64x1024x1024
+------------------------+-------------+-------------+-------------+
| Операция               |   CPU (сек) |   GPU (сек) | Ускорение   |
+========================+=============+=============+=============+
| Матричное умножение    |      1.1734 |      0.0475 | 24.71x      |
+------------------------+-------------+-------------+-------------+
| Поэлементное сложение  |      0.1940 |      0.0033 | 58.44x      |
+------------------------+-------------+-------------+-------------+
| Поэлементное умножение |      0.1891 |      0.0033 | 57.56x      |
+------------------------+-------------+-------------+-------------+
| Транспонирование       |      0.0000 |      0.0000 | 1.42x       |
+------------------------+-------------+-------------+-------------+
| Сумма всех элементов   |      0.0239 |      0.0011 | 21.80x      |
+------------------------+-------------+-------------+-------------+
```
```
Выполнение операций над матрицами размера 128x512x512
+------------------------+-------------+-------------+-------------+
| Операция               |   CPU (сек) |   GPU (сек) | Ускорение   |
+========================+=============+=============+=============+
| Матричное умножение    |      0.5009 |      0.0122 | 40.93x      |
+------------------------+-------------+-------------+-------------+
| Поэлементное сложение  |      0.0953 |      0.0017 | 56.15x      |
+------------------------+-------------+-------------+-------------+
| Поэлементное умножение |      0.0972 |      0.0017 | 57.44x      |
+------------------------+-------------+-------------+-------------+
| Транспонирование       |      0.0000 |      0.0000 | 1.00x       |
+------------------------+-------------+-------------+-------------+
| Сумма всех элементов   |      0.0126 |      0.0006 | 20.21x      |
+------------------------+-------------+-------------+-------------+
```
```
Выполнение операций над матрицами размера 256x256x256
+------------------------+-------------+-------------+-------------+
| Операция               |   CPU (сек) |   GPU (сек) | Ускорение   |
+========================+=============+=============+=============+
| Матричное умножение    |      0.1131 |      0.0033 | 34.47x      |
+------------------------+-------------+-------------+-------------+
| Поэлементное сложение  |      0.0422 |      0.0009 | 48.25x      |
+------------------------+-------------+-------------+-------------+
| Поэлементное умножение |      0.0364 |      0.0009 | 41.80x      |
+------------------------+-------------+-------------+-------------+
| Транспонирование       |      0.0001 |      0.0000 | 2.36x       |
+------------------------+-------------+-------------+-------------+
| Сумма всех элементов   |      0.0066 |      0.0004 | 17.91x      |
+------------------------+-------------+-------------+-------------+
```
Проанализировав таблицы, можем сказать:
1. Все операциии, кроме транспонирования, выполняются на gpu намного быстрее, чем на cpu. Это связанно с тем, что транспонирование это операция перестановки данных в памяти, а не вычислений.
2. Влияние размера матриц на ускорение: 
    -в случае с транспонированием и сумме всех элемментов разницы практически нет
    -при поэлементных операциях с матрицами большего размера, ускорение больше
    -при матричном умножении больших матриц, ускорение также больше.
4. GPU эффективен только при больших вычислениях, так как накладные расходы на передачу данных могут перевесить выгоду от ускорения.
5. Операции с малым объемом вычислений (например, транспонирование) могут не стоить переноса на GPU, так как время копирования данных может быть больше времени выполнения самой операции.









