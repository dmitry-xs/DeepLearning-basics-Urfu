import torch

tensor1 = torch.rand(3,4)
tensor2 = torch.zeros(2,3,4)
tensor3 = torch.ones(5,5)
tensor4 = torch.arange(0, 16).reshape(4,4)

print(f'tensor1: {tensor1}')
print(f'tensor2: {tensor2}')
print(f'tensor3: {tensor3}')
print(f'tensor4: {tensor4}')



A = torch.arange(0,12).reshape(3,4)
B = torch.arange(0,12).reshape(4,3)

print(f'Транспонированый тензор А: {A.T}')
print(f'Матричное умножение двух тензоров: {A @ B}')
print(f'Поэлементное умножение тензора А и транспонированного тензора В: {A * B.T}')
print(f'Сумма элементов тензора А: {torch.sum(A).item()}')


tensor = torch.randint(10, (5,5,5))
print(f'Первая строка: {tensor[0, 0, :]}')
print(f'Последний столбец: {tensor[:, :, -1]}')
print(f'Подматрица размером 2x2 из центра тензора: {tensor[2:4, 2:4, 2:4]}')
print(f'Все элементы с четными индексами: {tensor[::2, ::2, ::2]}')


tensor = torch.arange(0,24)
print(f'Изначальный тензор: {tensor}')
print(f'Тензор 2x12 {tensor.reshape(2,12)}')
print(f'Тензор 3x8 {tensor.view(3,8)}')
print(f'Тензор 4x6 {tensor.view(4,6)}')
print(f'Тензор 2x3x4 {tensor.reshape(2,3,4)}')
print(f'Тензор 2x2x2x3 {tensor.reshape(2,2,2,3)}')

