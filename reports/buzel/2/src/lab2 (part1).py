import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Убедитесь, что код выполняется только в основном потоке
if __name__ == '__main__':
    # 0. Настройка устройства (GPU, если доступен, иначе CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Используется устройство: {device}')

    # 1. Адаптация данных CIFAR-10 для ResNet
    # Предобученные модели, как ResNet, ожидают изображения большего размера (например, 224x224)
    # и нормализованные с определенными средними и стандартными отклонениями (от датасета ImageNet).
    transform_resnet = transforms.Compose([
        transforms.Resize(224), # Изменяем размер изображений 32x32 на 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Стандартная нормализация для ImageNet
    ])

    # Загрузка данных с новыми трансформациями
    trainset = torchvision.datasets.CIFAR10(root='D:/REALOIIS/lab2', train=True,
                                            download=True, transform=transform_resnet)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, # Можно увеличить batch_size, если позволяет память GPU
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='D:/REALOIIS/lab2', train=False,
                                           download=True, transform=transform_resnet)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 2. Загрузка и модификация предобученной модели ResNet34
    # Загружаем ResNet34 с весами, обученными на ImageNet
    model_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    # "Замораживаем" все слои сети, чтобы их веса не обновлялись во время обучения.
    # Мы хотим использовать их как готовый экстрактор признаков.
    for param in model_resnet.parameters():
        param.requires_grad = False

    # Узнаем, сколько признаков подается на вход последнему слою (классификатору)
    num_ftrs = model_resnet.fc.in_features

    # Заменяем последний слой (fc - fully connected) на новый, который мы будем обучать.
    # У него будет 10 выходов - по числу классов в CIFAR-10.
    # Веса этого слоя будут обучаемыми по умолчанию.
    model_resnet.fc = nn.Linear(num_ftrs, 10)

    # Перемещаем модель на выбранное устройство (GPU/CPU)
    model_resnet = model_resnet.to(device)

    # 3. Определение критерия и оптимизатора
    criterion = nn.CrossEntropyLoss()
    # Оптимизировать будем только параметры нового, незамороженного слоя
    optimizer = optim.SGD(model_resnet.fc.parameters(), lr=0.001, momentum=0.9)

    # 4. Процесс обучения
    print("Начало обучения ResNet34...")
    loss_history_resnet = []
    epochs = 3 # Для transfer learning часто нужно меньше эпох

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Перемещаем данные на то же устройство, что и модель
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model_resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499: # Вывод каждые 500 батчей
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                loss_history_resnet.append(running_loss / 500)
                running_loss = 0.0

    print('Обучение ResNet34 завершено')
    
    # Сохраняем обученную модель для использования в 4 пункте
    torch.save(model_resnet.state_dict(), 'D:/REALOIIS/lab2/cifar_resnet34.pth')

    # 5. Построение графика и оценка
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history_resnet)
    plt.title('График изменения ошибки обучения (ResNet34)')
    plt.xlabel('Итерации (x500)')
    plt.ylabel('Ошибка (Loss)')
    plt.grid(True)
    plt.show()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model_resnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Точность предобученной сети на 10000 тестовых изображений: {accuracy:.2f} %')