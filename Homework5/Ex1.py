import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from datasets import CustomImageDataset

# Создаем папку для сохранения результатов
os.makedirs('results/augmentation_results', exist_ok=True)

# 1. Создаем пайплайн аугментаций
transform_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomCrop(size=(200, 200)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
    transforms.RandomRotation(degrees=45),
    transforms.RandomGrayscale(p=0.5),
    transforms.Resize((224, 224))
])

# Отдельные трансформации для демонстрации
individual_transforms = {
    "Original": None,
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip(p=1.0),
    "RandomCrop": transforms.Compose([
        transforms.RandomCrop(size=(200, 200)),
        transforms.Resize((224, 224))
    ]),
    "ColorJitter": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
    "RandomRotation": transforms.RandomRotation(degrees=45),
    "RandomGrayscale": transforms.RandomGrayscale(p=1.0),
    "All_Together": transform_pipeline
}

# 2. Загружаем датасет
dataset = CustomImageDataset(root_dir='data/train', transform=None)

# 3. Выбираем по одному изображению из каждого класса (максимум 5 классов)
selected_images = []
classes_covered = set()

for i in range(len(dataset)):
    if len(selected_images) >= 5:
        break
    _, label = dataset[i]
    if label not in classes_covered:
        selected_images.append(i)
        classes_covered.add(label)


# 4. Функция для сохранения трансформаций
def save_transforms(image_idx):
    original_image, label = dataset[image_idx]
    class_name = dataset.get_class_names()[label]

    # Создаем подпапку для класса
    class_dir = os.path.join('results/augmentation_results', f'class_{class_name}')
    os.makedirs(class_dir, exist_ok=True)

    # Сохраняем оригинал
    original_path = os.path.join(class_dir, 'original.jpg')
    original_image.save(original_path)

    # Сохраняем отдельные трансформации
    for transform_name, transform in individual_transforms.items():
        if transform_name == "Original":
            continue  # оригинал уже сохранили

        transformed_image = transform(original_image) if transform else original_image

        # Сохраняем изображение
        transform_path = os.path.join(class_dir, f'{transform_name.lower()}.jpg')
        transformed_image.save(transform_path)

    # Создаем и сохраняем коллаж из всех трансформаций
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Augmentations for class: {class_name}', fontsize=16)

    for i, (transform_name, transform) in enumerate(individual_transforms.items()):
        plt.subplot(3, 3, i + 1)

        if transform_name == "Original":
            img = original_image
        else:
            img = transform(original_image) if transform else original_image

        plt.imshow(img)
        plt.title(transform_name)
        plt.axis('off')

    collage_path = os.path.join(class_dir, 'all_transforms_collage.jpg')
    plt.tight_layout()
    plt.savefig(collage_path, bbox_inches='tight', dpi=300)
    plt.close()


# Применяем и сохраняем для каждого выбранного изображения
for img_idx in selected_images:
    save_transforms(img_idx)

print("Все аугментации сохранены в папку 'augmentation_results(ex4)'")