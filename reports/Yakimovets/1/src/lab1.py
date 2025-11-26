import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Нормализация для MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)


# CNN архитектура
class MNIST_CNN(nn.Module):

    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv_block1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv_block2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc_block1 = nn.Linear(64 * 7 * 7, 128)
        self.act3 = nn.ReLU()
        self.fc_output = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv_block1(x)))
        x = self.pool2(self.act2(self.conv_block2(x)))
        x = self.flatten(x)
        x = self.act3(self.fc_block1(x))
        x = self.fc_output(x)
        return x


# Инициализация модели, лосса и оптимизатора
cnn_model = MNIST_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn_model.parameters(), lr=0.01, momentum=0.9)

# Обучение модели
num_epochs = 10
train_losses = []

for epoch in range(num_epochs):
    cnn_model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        logits = cnn_model(inputs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# --- Сохранение графика ошибок ---
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig("training_loss.png", dpi=200)
plt.close()


# Оценка точности на тестовой выборке
cnn_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        logits = cnn_model(inputs)
        predictions = logits.argmax(dim=1, keepdim=True)

        correct += predictions.eq(labels.view_as(predictions)).sum().item()
        total += labels.size(0)

accuracy = 100. * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')


# Визуализация работы модели
data_iter = iter(test_loader)
images, labels = next(data_iter)

sample_img = images[0]
true_label = labels[0]

with torch.no_grad():
    logits = cnn_model(sample_img.unsqueeze(0))
    predicted_label = logits.argmax().item()

plt.figure()
plt.imshow(sample_img.squeeze(), cmap='gray')
plt.title(f'Predicted: {predicted_label}, True: {true_label}')
plt.savefig("prediction.png", dpi=200)
plt.close()
