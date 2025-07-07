import os
from datasets import CustomImageDataset
from torchvision.utils import save_image
from Aug_pipe import *

def apply_and_save_pipelines(dataset_path: str = 'data/train', output_dir: str = 'augmentation_results(ex4)'):
    """Применяет все конфигурации аугментаций и сохраняет результаты"""

    # Создаем папки для результатов
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'light'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'medium'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'heavy'), exist_ok=True)

    # Загружаем датасет
    dataset = CustomImageDataset(root_dir=dataset_path, transform=None)

    # Создаем конфигурации аугментаций
    light_augs = create_light_augmentations()
    medium_augs = create_medium_augmentations()
    heavy_augs = create_heavy_augmentations()

    # Выбираем по 2 изображения из каждого класса для демонстрации
    samples_per_class = 2
    selected_indices = []
    class_counts = {}

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if class_counts.get(label, 0) < samples_per_class:
            selected_indices.append(idx)
            class_counts[label] = class_counts.get(label, 0) + 1
        if len(class_counts) == len(dataset.classes) and all(v == samples_per_class for v in class_counts.values()):
            break

    # Применяем и сохраняем аугментации
    for idx in selected_indices:
        image, label = dataset[idx]
        class_name = dataset.classes[label]

        # Создаем подпапки для класса
        os.makedirs(os.path.join(output_dir, 'light', class_name), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'medium', class_name), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'heavy', class_name), exist_ok=True)

        # Сохраняем оригинал
        original_path = os.path.join(output_dir, f'original_{class_name}_{idx}.jpg')
        image.save(original_path)

        # Применяем и сохраняем light аугментации
        light_img = light_augs.apply(image)
        light_path = os.path.join(output_dir, 'light', class_name, f'aug_{idx}.jpg')
        light_img.save(light_path)

        # Применяем и сохраняем medium аугментации
        medium_img = medium_augs.apply(image)
        medium_path = os.path.join(output_dir, 'medium', class_name, f'aug_{idx}.jpg')
        medium_img.save(medium_path)

        # Применяем и сохраняем heavy аугментации
        heavy_img = heavy_augs.apply(image)
        heavy_path = os.path.join(output_dir, 'heavy', class_name, f'aug_{idx}.jpg')
        heavy_img.save(heavy_path)

    print(f"Результаты сохранены в {output_dir}")


# Запускаем процесс
apply_and_save_pipelines()

import matplotlib.pyplot as plt


def create_comparison_collages(output_dir: str = 'augmentation_results(ex4)'):
    """Создает коллажи для сравнения разных уровней аугментаций"""

    dataset = CustomImageDataset(root_dir='data/train', transform=None)
    class_names = dataset.classes

    for class_name in class_names:
        # Находим все изображения для этого класса
        class_images = []
        for idx, (_, label) in enumerate(dataset):
            if dataset.classes[label] == class_name:
                class_images.append(idx)
                if len(class_images) >= 3:  # Берем 3 изображения для примера
                    break

        # Создаем коллаж для каждого изображения
        for img_idx in class_images:
            plt.figure(figsize=(15, 5))

            # Оригинал
            plt.subplot(1, 4, 1)
            original_img = Image.open(os.path.join(output_dir, f'original_{class_name}_{img_idx}.jpg'))
            plt.imshow(original_img)
            plt.title('Original')
            plt.axis('off')

            # Light
            plt.subplot(1, 4, 2)
            light_img = Image.open(os.path.join(output_dir, 'light', class_name, f'aug_{img_idx}.jpg'))
            plt.imshow(light_img)
            plt.title('Light Augmentations')
            plt.axis('off')

            # Medium
            plt.subplot(1, 4, 3)
            medium_img = Image.open(os.path.join(output_dir, 'medium', class_name, f'aug_{img_idx}.jpg'))
            plt.imshow(medium_img)
            plt.title('Medium Augmentations')
            plt.axis('off')

            # Heavy
            plt.subplot(1, 4, 4)
            heavy_img = Image.open(os.path.join(output_dir, 'heavy', class_name, f'aug_{img_idx}.jpg'))
            plt.imshow(heavy_img)
            plt.title('Heavy Augmentations')
            plt.axis('off')

            # Сохраняем коллаж
            os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'comparisons', f'comparison_{class_name}_{img_idx}.jpg'),
                        bbox_inches='tight', dpi=150)
            plt.close()


create_comparison_collages()
print("Коллажи для сравнения сохранены в augmentation_results(ex4)/comparisons")