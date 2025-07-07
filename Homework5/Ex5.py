import os
import time
import torch
import psutil
import matplotlib.pyplot as plt
from datasets import CustomImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Конфигурация эксперимента
SIZES = [64, 128, 224, 512]  # Тестируемые размеры
NUM_IMAGES = 100              # Количество изображений для обработки
REPEATS = 3                   # Количество повторений для усреднения

# Аугментации для теста (фиксированный набор)
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

# Функция для измерения памяти
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # в MB


results = {'time': [], 'memory': [], 'size': []}

for size in SIZES:
    print(f"\nТестируем размер {size}x{size}...")

    # Создаем датасет с текущим размером
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        augmentations
    ])

    dataset = CustomImageDataset(root_dir='data/train', transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Измеряем время и память
    time_total = 0
    memory_usage = []

    for _ in range(REPEATS):
        start_time = time.time()
        start_mem = get_memory_usage()

        # Обрабатываем NUM_IMAGES изображений
        for i, (image, _) in enumerate(loader):
            if i >= NUM_IMAGES:
                break

        end_time = time.time()
        end_mem = get_memory_usage()

        time_total += (end_time - start_time)
        memory_usage.append(end_mem - start_mem)

    # Сохраняем средние значения
    avg_time = time_total / REPEATS
    avg_memory = sum(memory_usage) / len(memory_usage)

    results['size'].append(size)
    results['time'].append(avg_time)
    results['memory'].append(avg_memory)

    print(f"Среднее время: {avg_time:.2f} сек")
    print(f"Среднее потребление памяти: {avg_memory:.2f} MB")

# Создаем папку для результатов
os.makedirs('results/size_experiment', exist_ok=True)

# График времени обработки
plt.figure(figsize=(10, 5))
plt.plot(results['size'], results['time'], 'bo-')
plt.title('Зависимость времени обработки от размера изображения')
plt.xlabel('Размер изображения (пиксели)')
plt.ylabel('Время обработки 100 изображений (сек)')
plt.grid(True)
plt.savefig('size_experiment/time_vs_size.png', dpi=150)
plt.close()

# График потребления памяти
plt.figure(figsize=(10, 5))
plt.plot(results['size'], results['memory'], 'ro-')
plt.title('Зависимость потребления памяти от размера изображения')
plt.xlabel('Размер изображения (пиксели)')
plt.ylabel('Потребление памяти (MB)')
plt.grid(True)
plt.savefig('size_experiment/memory_vs_size.png', dpi=150)
plt.close()

# Комбинированный график
plt.figure(figsize=(12, 6))
plt.plot(results['size'], results['time'], 'bo-', label='Время (сек)')
plt.plot(results['size'], results['memory'], 'ro-', label='Память (MB)')
plt.title('Зависимость времени и памяти от размера изображения')
plt.xlabel('Размер изображения (пиксели)')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.savefig('size_experiment/combined_results.png', dpi=150)
plt.close()

print("\nРезультаты сохранены в папке 'size_experiment'")

# Выводим таблицу результатов
print("\nРезультаты эксперимента:")
print("Размер\tВремя (сек)\tПамять (MB)")
for size, t, m in zip(results['size'], results['time'], results['memory']):
    print(f"{size}x{size}\t{t:.2f}\t\t{m:.2f}")

# Анализ зависимости
print("\nАнализ:")
for i in range(1, len(SIZES)):
    size_ratio = (SIZES[i] / SIZES[i - 1]) ** 2
    time_ratio = results['time'][i] / results['time'][i - 1]
    mem_ratio = results['memory'][i] / results['memory'][i - 1]

    print(f"\nПри увеличении размера с {SIZES[i - 1]} до {SIZES[i]} (в {size_ratio:.1f} раз):")
    print(f"- Время обработки увеличилось в {time_ratio:.1f} раз")
    print(f"- Потребление памяти увеличилось в {mem_ratio:.1f} раз")