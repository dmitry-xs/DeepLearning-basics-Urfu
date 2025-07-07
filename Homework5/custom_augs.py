import torch
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import cv2


class RandomColorShift:
    """Случайный сдвиг цветовых каналов"""

    def __init__(self, max_shift=15):
        self.max_shift = max_shift

    def __call__(self, img):
        img_np = np.array(img)
        h, w, c = img_np.shape

        # Случайные сдвиги для каждого канала
        shifts = [random.randint(-self.max_shift, self.max_shift) for _ in range(3)]

        # Применяем сдвиг к каждому каналу
        for i in range(3):
            if shifts[i] > 0:
                img_np[:, :, i] = np.pad(img_np[:, :, i], ((0, 0), (shifts[i], 0)), mode='edge')[:, :-shifts[i]]
            elif shifts[i] < 0:
                img_np[:, :, i] = np.pad(img_np[:, :, i], ((0, 0), (0, -shifts[i])), mode='edge')[:, -shifts[i]:]

        return Image.fromarray(img_np)


class RandomWaveDistortion:
    """Волнообразное искажение изображения"""

    def __init__(self, amplitude=5, frequency=0.05):
        self.amplitude = amplitude
        self.frequency = frequency

    def __call__(self, img):
        img_np = np.array(img)
        h, w, c = img_np.shape

        # Создаем координатную сетку
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Добавляем волнообразные искажения
        dx = self.amplitude * np.sin(2 * np.pi * self.frequency * y)
        dy = self.amplitude * np.cos(2 * np.pi * self.frequency * x)

        # Применяем искажения
        x_new = np.clip(x + dx, 0, w - 1).astype(np.float32)
        y_new = np.clip(y + dy, 0, h - 1).astype(np.float32)

        # Деформируем изображение
        distorted = cv2.remap(img_np, x_new, y_new, cv2.INTER_LINEAR)
        return Image.fromarray(distorted)


class RandomPixelShuffle:
    """Случайное перемешивание блоков пикселей"""

    def __init__(self, block_size=16, shuffle_prob=0.1):
        self.block_size = block_size
        self.shuffle_prob = shuffle_prob

    def __call__(self, img):
        img_np = np.array(img)
        h, w, c = img_np.shape

        # Разбиваем изображение на блоки
        blocks_h = h // self.block_size
        blocks_w = w // self.block_size

        # Создаем список блоков
        blocks = []
        for i in range(blocks_h):
            for j in range(blocks_w):
                block = img_np[i * self.block_size:(i + 1) * self.block_size,
                        j * self.block_size:(j + 1) * self.block_size]
                blocks.append(block)

        # Перемешиваем блоки с заданной вероятностью
        for i in range(len(blocks)):
            if random.random() < self.shuffle_prob:
                j = random.randint(0, len(blocks) - 1)
                blocks[i], blocks[j] = blocks[j], blocks[i]

        # Собираем изображение обратно
        result = np.zeros_like(img_np)
        idx = 0
        for i in range(blocks_h):
            for j in range(blocks_w):
                result[i * self.block_size:(i + 1) * self.block_size,
                j * self.block_size:(j + 1) * self.block_size] = blocks[idx]
                idx += 1

        return Image.fromarray(result)