import os
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datasets import CustomImageDataset

# Загружаем датасет
dataset = CustomImageDataset(root_dir='data/train', transform=None)

# Подсчитываем количество изображений в каждом классе
class_counts = defaultdict(int)
for _, label in dataset:
    class_counts[label] += 1

# Сортируем по именам классов
class_names = dataset.get_class_names()
sorted_counts = {class_names[k]: v for k, v in sorted(class_counts.items())}

print("Количество изображений по классам:")
for class_name, count in sorted_counts.items():
    print(f"{class_name}: {count} изображений")

# Собираем информацию о размерах изображений
widths = []
heights = []
aspect_ratios = []

for img_path in dataset.images:
    with Image.open(img_path) as img:
        width, height = img.size
        widths.append(width)
        heights.append(height)
        aspect_ratios.append(width / height)

# Вычисляем статистику
size_stats = {
    'min_width': min(widths),
    'max_width': max(widths),
    'mean_width': np.mean(widths),
    'min_height': min(heights),
    'max_height': max(heights),
    'mean_height': np.mean(heights),
    'min_aspect': min(aspect_ratios),
    'max_aspect': max(aspect_ratios),
    'mean_aspect': np.mean(aspect_ratios)
}

print("\nСтатистика размеров изображений:")
print(f"Ширина: мин={size_stats['min_width']}, макс={size_stats['max_width']}, среднее={size_stats['mean_width']:.1f}")
print(f"Высота: мин={size_stats['min_height']}, макс={size_stats['max_height']}, среднее={size_stats['mean_height']:.1f}")
print(f"Соотношение сторон: мин={size_stats['min_aspect']:.2f}, макс={size_stats['max_aspect']:.2f}, среднее={size_stats['mean_aspect']:.2f}")

# Создаем папку для сохранения графиков
os.makedirs('results/dataset_analysis', exist_ok=True)

# Гистограмма распределения изображений по классам
plt.figure(figsize=(12, 6))
plt.bar(sorted_counts.keys(), sorted_counts.values())
plt.title('Распределение изображений по классам')
plt.xlabel('Класс')
plt.ylabel('Количество изображений')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('dataset_analysis/class_distribution.png', dpi=150)
plt.close()

# График распределения размеров изображений
plt.figure(figsize=(12, 6))
plt.scatter(widths, heights, alpha=0.5)
plt.title('Распределение размеров изображений')
plt.xlabel('Ширина (пиксели)')
plt.ylabel('Высота (пиксели)')
plt.grid(True)
plt.savefig('dataset_analysis/size_distribution.png', dpi=150)
plt.close()

# Гистограмма соотношений сторон
plt.figure(figsize=(12, 6))
plt.hist(aspect_ratios, bins=30, edgecolor='black')
plt.title('Распределение соотношений сторон изображений')
plt.xlabel('Соотношение сторон (ширина/высота)')
plt.ylabel('Количество изображений')
plt.grid(True)
plt.savefig('dataset_analysis/aspect_ratio_distribution.png', dpi=150)
plt.close()

# Boxplot размеров по классам
class_sizes = defaultdict(list)
for img_path, label in zip(dataset.images, dataset.labels):
    with Image.open(img_path) as img:
        class_sizes[label].append(sum(img.size))  # Сумма ширины и высоты

plt.figure(figsize=(12, 6))
plt.boxplot([class_sizes[k] for k in sorted(class_sizes.keys())],
            labels=[class_names[k] for k in sorted(class_sizes.keys())])
plt.title('Распределение размеров изображений по классам')
plt.xlabel('Класс')
plt.ylabel('Ширина + высота (пиксели)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('dataset_analysis/class_size_boxplot.png', dpi=150)
plt.close()

print("\nГрафики сохранены в папку 'dataset_analysis'")