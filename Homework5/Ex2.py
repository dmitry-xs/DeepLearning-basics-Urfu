import os
import matplotlib.pyplot as plt
from torchvision import transforms
from datasets import CustomImageDataset
from custom_augs import RandomColorShift, RandomWaveDistortion, RandomPixelShuffle
from extra_augs import AddGaussianNoise, ElasticTransform, Posterize
from PIL import Image

# Создаем папки для результатов
os.makedirs('results/custom_augs', exist_ok=True)
os.makedirs('results/extra_augs', exist_ok=True)
os.makedirs('results/comparison', exist_ok=True)

# Загружаем датасет
dataset = CustomImageDataset(root_dir='data/train', transform=None)

# Выбираем 3 изображения для демонстрации
sample_indices = [0, len(dataset) // 2, len(dataset) - 1]

# Наши кастомные аугментации
custom_augs = {
    'ColorShift': RandomColorShift(),
    'WaveDistortion': RandomWaveDistortion(),
    'PixelShuffle': RandomPixelShuffle()
}

# Готовые аугментации из extra_augs.py
extra_augs = {
    'GaussianNoise': AddGaussianNoise(),
    'ElasticTransform': ElasticTransform(),
    'Posterize': Posterize()
}


def apply_and_save_augs(img_idx, augs_dict, save_dir):
    """Применяет и сохраняет аугментации"""
    image, label = dataset[img_idx]
    class_name = dataset.get_class_names()[label]

    # Преобразуем в тензор для extra_augs
    tensor_img = transforms.ToTensor()(image)

    # Создаем подпапку
    class_dir = os.path.join(save_dir, f'class_{class_name}')
    os.makedirs(class_dir, exist_ok=True)

    # Сохраняем оригинал
    image.save(os.path.join(class_dir, f'original_{img_idx}.jpg'))

    # Применяем и сохраняем каждую аугментацию
    for name, aug in augs_dict.items():
        if save_dir == 'custom_augs':
            # Для наших кастомных аугментаций
            augmented = aug(image)
            augmented.save(os.path.join(class_dir, f'{name}_{img_idx}.jpg'))
        else:
            # Для готовых аугментаций (работают с тензорами)
            augmented = aug(tensor_img)
            augmented = transforms.ToPILImage()(augmented)
            augmented.save(os.path.join(class_dir, f'{name}_{img_idx}.jpg'))


# Применяем аугментации
for idx in sample_indices:
    apply_and_save_augs(idx, custom_augs, 'results/custom_augs')
    apply_and_save_augs(idx, extra_augs, 'results/extra_augs')

# Создаем сравнения
for idx in sample_indices:
    _, label = dataset[idx]
    class_name = dataset.get_class_names()[label]

    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Сравнение аугментаций для класса {class_name}', fontsize=16)

    # Загружаем изображения
    custom_images = {}
    extra_images = {}

    custom_dir = os.path.join('results/custom_augs', f'class_{class_name}')
    extra_dir = os.path.join('results/extra_augs', f'class_{class_name}')

    for name in custom_augs.keys():
        custom_images[name] = Image.open(os.path.join(custom_dir, f'{name}_{idx}.jpg'))

    for name in extra_augs.keys():
        extra_images[name] = Image.open(os.path.join(extra_dir, f'{name}_{idx}.jpg'))

    # Отображаем оригинал
    plt.subplot(3, 3, 1)
    plt.imshow(dataset[idx][0])
    plt.title('Original')
    plt.axis('off')

    # Отображаем кастомные аугментации
    for i, (name, img) in enumerate(custom_images.items()):
        plt.subplot(3, 3, i + 2)
        plt.imshow(img)
        plt.title(f'Custom: {name}')
        plt.axis('off')

    # Отображаем готовые аугментации
    for i, (name, img) in enumerate(extra_images.items()):
        plt.subplot(3, 3, i + 5)
        plt.imshow(img)
        plt.title(f'Extra: {name}')
        plt.axis('off')

    # Сохраняем сравнение
    plt.tight_layout()
    plt.savefig(os.path.join('results/comparison', f'compare_{class_name}_{idx}.jpg'), dpi=150)
    plt.close()

print("Результаты сохранены в папках 'custom_augs', 'extra_augs' и 'comparison'")