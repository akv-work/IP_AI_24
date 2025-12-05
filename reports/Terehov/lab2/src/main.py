import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')

def create_mobilenet_model(num_classes=100, pretrained=True, freeze_features=False):
    """Создание модели MobileNet v2"""
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
    except:
        model = models.mobilenet_v2(pretrained=pretrained)

    # Замена классификатора
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    return model.to(device)

def train_model():
    """Основная функция обучения"""

    # Параметры (можно менять)
    batch_size = 128
    epochs = 15
    learning_rate = 0.001

    # Аугментации
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761])
    ])

    # Загрузка данных
    print("Downloading CIFAR-100 dataset...")
    train_dataset = datasets.CIFAR100(root='./data', train=True,
                                     download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False,
                                    download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    # Создание модели
    model = create_mobilenet_model(num_classes=100, pretrained=True,
                                  freeze_features=False)

    # Оптимизатор и функция потерь
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate,
                             alpha=0.9, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Обучение
    train_losses = []
    val_accuracies = []

    print("Starting training...")
    for epoch in range(epochs):
        # Обучение
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Валидация
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        train_losses.append(train_loss)
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch+1}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Acc: {val_acc:.2f}%')

        scheduler.step()

    # Графики
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Сохранение модели
    torch.save(model.state_dict(), 'mobilenet_cifar100_colab.pth')
    print("Model saved as 'mobilenet_cifar100_colab.pth'")

    # Скачивание файла
    from google.colab import files
    files.download('mobilenet_cifar100_colab.pth')

    return model

# Запуск обучения
model = train_model()
