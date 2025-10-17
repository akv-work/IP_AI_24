import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Подготовка данных ---

# Трансформации для изображений: преобразование в тензор и нормализация
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# Загрузка обучающей и тестовой выборок Fashion-MNIST
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                  download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                 download=True, transform=transform)

# Создание загрузчиков данных
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Классы для визуализации
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')


# --- 2. Определение архитектуры СНС ---

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Сверточная часть
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # 28x28x1 -> 28x28x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28x28x32 -> 14x14x32
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # 14x14x32 -> 14x14x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 14x14x64 -> 7x7x64
        )
        # Классификационная часть (полносвязные слои)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10) # 10 классов
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 3. Обучение модели ---

# Инициализация модели, функции потерь и оптимизатора
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Обучение на устройстве: {device}")

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters()) # Оптимизатор Adadelta согласно варианту

# Цикл обучения
num_epochs = 15
loss_history = []

print("Начало обучения...")
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Обнуление градиентов
        optimizer.zero_grad()

        # Прямой проход + обратный проход + оптимизация
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    loss_history.append(epoch_loss)
    print(f'Эпоха [{epoch + 1}/{num_epochs}], Потери: {epoch_loss:.4f}')

print('Обучение завершено.')


# --- 4. Оценка эффективности и построение графика ошибки ---

# Оценка на тестовой выборке
model.eval() # Переключение модели в режим оценки
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Точность на 10000 тестовых изображений: {accuracy:.2f} %')


# Построение графика изменения ошибки
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
plt.title('График изменения ошибки (Loss) во время обучения')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.grid(True)
plt.show()


# --- 5. Реализация визуализации работы СНС ---

# Функция для отображения изображения
def imshow(img):
    img = img / 2 + 0.5     # денормализация
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Получаем случайные изображения из тестового набора
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# Выводим первые 4 изображения для примера
print("Пример изображений из тестовой выборки:")
imshow(torchvision.utils.make_grid(images.cpu()[:4]))
print('Реальные метки: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# Получаем предсказания для этих изображений
outputs = model(images[:4])
_, predicted = torch.max(outputs, 1)

print('Предсказанные метки: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

# Визуализация одного произвольного изображения и результата
image_index = np.random.randint(0, len(images))
single_image = images[image_index].unsqueeze(0) # добавляем batch dimension
single_label = labels[image_index]

output = model(single_image)
_, predicted_class = torch.max(output, 1)

print("\n--- Визуализация работы СНС на одном изображении ---")
# Денормализация и отображение
img_to_show = single_image.cpu().squeeze() / 2 + 0.5
plt.imshow(img_to_show, cmap="gray")
plt.title(f"Реальный класс: {classes[single_label.item()]}\n"
          f"Предсказанный класс: {classes[predicted_class.item()]}")
plt.axis('off')
plt.show()

if single_label.item() == predicted_class.item():
    print("Результат: Классификация верна.")
else:
    print("Результат: Классификация неверна.")