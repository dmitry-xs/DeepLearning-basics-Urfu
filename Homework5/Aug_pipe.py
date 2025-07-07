import torch
from torchvision import transforms
from PIL import Image
from typing import Dict, Callable


class AugmentationPipeline:
    """Класс для создания и управления пайплайном аугментаций"""

    def __init__(self):
        self.augmentations = {}

    def add_augmentation(self, name: str, aug: Callable) -> None:
        """Добавляет аугментацию в пайплайн"""
        self.augmentations[name] = aug

    def remove_augmentation(self, name: str) -> None:
        """Удаляет аугментацию из пайплайна"""
        if name in self.augmentations:
            del self.augmentations[name]

    def apply(self, image: Image.Image) -> Image.Image:
        """Применяет все аугментации к изображению"""
        for aug in self.augmentations.values():
            image = aug(image)
        return image

    def get_augmentations(self) -> Dict[str, Callable]:
        """Возвращает словарь всех аугментаций"""
        return self.augmentations.copy()

def create_light_augmentations() -> AugmentationPipeline:
    """Легкие аугментации (минимальные изменения)"""
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation('resize', transforms.Resize((256, 256)))
    pipeline.add_augmentation('random_flip', transforms.RandomHorizontalFlip(p=0.3))
    pipeline.add_augmentation('color_jitter', transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1))
    return pipeline

def create_medium_augmentations() -> AugmentationPipeline:
    """Средние аугментации (умеренные изменения)"""
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation('resize', transforms.Resize((256, 256)))
    pipeline.add_augmentation('random_flip', transforms.RandomHorizontalFlip(p=0.5))
    pipeline.add_augmentation('color_jitter', transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    pipeline.add_augmentation('random_rotation', transforms.RandomRotation(15))
    pipeline.add_augmentation('random_crop', transforms.RandomResizedCrop(
        256, scale=(0.8, 1.0)))
    return pipeline

def create_heavy_augmentations() -> AugmentationPipeline:
    """Сильные аугментации (значительные изменения)"""
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation('resize', transforms.Resize((256, 256)))
    pipeline.add_augmentation('random_flip', transforms.RandomHorizontalFlip(p=0.7))
    pipeline.add_augmentation('color_jitter', transforms.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
    pipeline.add_augmentation('random_rotation', transforms.RandomRotation(30))
    pipeline.add_augmentation('random_crop', transforms.RandomResizedCrop(
        256, scale=(0.6, 1.0)))
    pipeline.add_augmentation('random_grayscale', transforms.RandomGrayscale(p=0.2))
    pipeline.add_augmentation('gaussian_blur', transforms.GaussianBlur(3, sigma=(0.1, 2.0)))
    return pipeline