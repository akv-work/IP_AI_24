import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация для RGB
])

train_dataset = torchvision.datasets.STL10(root='./data', split='train', transform=transform, download=True)
test_dataset = torchvision.datasets.STL10(root='./data', split='test', transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Определение модели
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 12 * 12, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)  # 10 классов в STL-10

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, 64 * 12 * 12)  # Выравнивание для полносвязного слоя
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x


# Инициализация модели, функции потерь и оптимизатора
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Обучение модели
num_epochs = 20
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Оценка на тестовой выборке
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print(f"Эпоха {epoch + 1}/{num_epochs}, Тренировочная потеря: {train_loss:.4f}, "
          f"Тренировочная точность: {train_accuracy:.2f}%, "
          f"Тестовая потеря: {test_loss:.4f}, Тестовая точность: {test_accuracy:.2f}%")

# Построение графиков
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Тренировочная потеря')
plt.plot(test_losses, label='Тестовая потеря')
plt.xlabel('Эпоха')
plt.ylabel('Потеря')
plt.legend()
plt.title('График изменения ошибки')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Тренировочная точность')
plt.plot(test_accuracies, label='Тестовая точность')
plt.xlabel('Эпоха')
plt.ylabel('Точность (%)')
plt.legend()
plt.title('График изменения точности')

plt.savefig('training_plot.png')
plt.close()


# Визуализация предсказания
def visualize_prediction(model, dataset, device):
    model.eval()
    classes = dataset.classes
    idx = np.random.randint(0, len(dataset))
    image, label = dataset[idx]
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    image = image.squeeze().cpu().numpy().transpose(1, 2, 0)
    image = image * 0.5 + 0.5
    plt.imshow(image)
    plt.title(f'Истинный класс: {classes[label]}\nПредсказанный класс: {classes[predicted.item()]}')
    plt.axis('off')
    plt.savefig('prediction.png')
    plt.close()


visualize_prediction(model, test_dataset, device)

torch.save(model.state_dict(), 'stl10_model.pth')