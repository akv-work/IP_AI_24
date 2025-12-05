import os
import time
import requests
from io import BytesIO
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models

# ======================
# ПАРАМЕТРЫ
# ======================
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
INPUT_SIZE = 64
NUM_WORKERS = 4
SAVE_DIR = 'results_lab2'

# Ссылка на произвольное изображение из интернета для теста (Пункт 4)
TEST_IMAGE_URL = "data:image/jpeg;base64,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"  # Пример футболки

# Устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Классы Fashion-MNIST
CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


# ======================
# 1. МОДЕЛИ
# ======================


# --- Модель из ЛР 2 (DenseNet121) ---
def create_densenet121(num_classes=10):
    print("Creating DenseNet121...")
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    # Замена классификатора
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


# --- Модель из ЛР 1 (Кастомная архитектура) ---
# ВАЖНО: Сюда лучше вставить класс твоей модели из первой лабы.
# Я написал универсальный пример, который работает с размером 224x224.
class CustomCNN_Lab1(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN_Lab1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Вход 3 канала (т.к. трансформ общий)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Адаптивный пулинг, чтобы не зависеть от размера входа
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ======================
# 2. ФУНКЦИИ ОБУЧЕНИЯ
# ======================


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader.dataset), 100. * correct / total


def train_model(model, train_loader, test_loader, model_name="Model"):
    print(f"\nTraining {model_name} on {device}...")

    # Оптимизатор RMSprop (по заданию)
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}

    start_time = time.time()

    for epoch in range(EPOCHS):
        t_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        history['train_loss'].append(t_loss)
        history['test_loss'].append(v_loss)
        history['test_acc'].append(v_acc)

        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"{model_name} training finished in {total_time:.1f}s")

    return history, total_time


# ======================
# 3. ВИЗУАЛИЗАЦИЯ
# ======================


def download_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


def visualize_comparison(model_densenet, model_custom, transform):
    print("\n--- Visualizing Predictions ---")

    # Скачиваем изображение
    img_orig = download_image(TEST_IMAGE_URL)
    if img_orig is None:
        return

    # Подготовка изображения
    img_tensor = transform(img_orig).unsqueeze(0).to(device)

    model_densenet.eval()
    model_custom.eval()

    with torch.no_grad():
        # DenseNet Prediction
        out1 = model_densenet(img_tensor)
        prob1 = torch.softmax(out1, dim=1)
        score1, pred1 = prob1.max(1)
        label1 = CLASSES[pred1.item()]

        # Custom CNN Prediction
        out2 = model_custom(img_tensor)
        prob2 = torch.softmax(out2, dim=1)
        score2, prob2 = prob2.max(1)
        label2 = CLASSES[prob2.item()]

    # Отображение
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_orig)
    plt.axis('off')
    plt.title(f"DenseNet121 Prediction:\n{label1} ({score1.item():.2%})", color='green')

    plt.subplot(1, 2, 2)
    plt.imshow(img_orig)
    plt.axis('off')
    plt.title(f"Custom CNN (Lab 1) Prediction:\n{label2} ({score2.item():.2%})", color='blue')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'visualization_comparison.png'))
    plt.show()


# ======================
# MAIN
# ======================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Данные
    # Используем одинаковые преобразования для чистоты эксперимента (размер 224 для DenseNet)
    # Grayscale(3) нужен, так как DenseNet ожидает 3 канала (RGB)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 2. Создание моделей
    dense_model = create_densenet121().to(device)
    custom_model = CustomCNN_Lab1().to(device)

    # 3. Обучение
    dense_hist, dense_time = train_model(dense_model, train_loader, test_loader, "DenseNet121")
    custom_hist, custom_time = train_model(custom_model, train_loader, test_loader, "CustomCNN")

    # 4. Сравнение результатов (Графики)
    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(14, 5))

    # Loss comparison
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, dense_hist['test_loss'], label='DenseNet Val Loss', marker='o')
    plt.plot(epochs_range, custom_hist['test_loss'], label='Custom Val Loss', marker='x', linestyle='--')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy comparison
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, dense_hist['test_acc'], label='DenseNet Val Acc', marker='o')
    plt.plot(epochs_range, custom_hist['test_acc'], label='Custom Val Acc', marker='x', linestyle='--')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(SAVE_DIR, 'comparison_plots.png'))
    print(f"Comparison plots saved to {SAVE_DIR}")
    plt.show()

    # Вывод текстовой таблицы сравнения
    print("\n" + "=" * 40)
    print(f"{'Metric':<15} | {'DenseNet121':<10} | {'CustomCNN':<10}")
    print("-" * 40)
    print(f"{'Best Acc (%)':<15} | {max(dense_hist['test_acc']):<10.2f} | {max(custom_hist['test_acc']):<10.2f}")
    print(f"{'Training Time':<15} | {dense_time:<10.1f}s| {custom_time:<10.1f}s")
    print("=" * 40 + "\n")

    # 5. Визуализация работы на произвольном изображении
    visualize_comparison(dense_model, custom_model, transform)


if __name__ == "__main__":
    main()
