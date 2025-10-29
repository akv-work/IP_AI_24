import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


batch_size = 64  
learning_rate = 0.01  
num_epochs = 15  


train_losses = []
test_accuracies = []
test_losses = []


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


fashion_mnist_classes = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.dropout = nn.Dropout(0.5)
        
       
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        
        
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        
       
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        
        
        x = x.view(-1, 128 * 3 * 3)
        
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


model = ImprovedCNN()
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print("Начало обучения...")
print(f"Архитектура модели:\n{model}")


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
   
    scheduler.step()
    
    avg_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(avg_loss)
    
   
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    test_accuracies.append(test_accuracy)
    test_losses.append(test_loss / len(test_loader))
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
          f'Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracy:.2f}%, '
          f'LR: {scheduler.get_last_lr()[0]:.6f}')


model.eval()
final_correct = 0
final_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        final_total += labels.size(0)
        final_correct += (predicted == labels).sum().item()

final_accuracy = 100 * final_correct / final_total
print(f'\nФинальная точность на тестовой выборке: {final_accuracy:.2f}%')


plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, marker='s', label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.xticks(range(1, num_epochs + 1))


plt.subplot(1, 3, 2)
plt.plot(range(1, num_epochs + 1), test_accuracies, marker='o', color='green', label='Test Accuracy')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.xticks(range(1, num_epochs + 1))

plt.subplot(1, 3, 3)
current_accuracy = final_accuracy
sota_accuracy = 96.7  
simple_cnn_accuracy = 88.0  

methods = ['Наша модель', 'Простая CNN', 'SOTA']
accuracies = [current_accuracy, simple_cnn_accuracy, sota_accuracy]
colors = ['blue', 'orange', 'green']

bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
plt.title('Сравнение с State-of-the-Art')
plt.ylabel('Accuracy (%)')
plt.ylim(80, 100)

for bar, accuracy in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{accuracy:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

def visualize_predictions(model, test_loader, num_images=10):
    model.eval()
    images, labels = next(iter(test_loader))
    
    with torch.no_grad():
        outputs = model(images[:num_images])
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
    
    plt.figure(figsize=(15, 8))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        
        true_label = fashion_mnist_classes[labels[i].item()]
        pred_label = fashion_mnist_classes[predicted[i].item()]
        confidence = probabilities[i][predicted[i]].item() * 100
        
        color = 'green' if labels[i] == predicted[i] else 'red'
        plt.title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%', 
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    correct = (labels[:num_images] == predicted).sum().item()
    print(f'Правильно классифицировано: {correct}/{num_images} ({100*correct/num_images:.1f}%)')

print("\nВизуализация предсказаний на тестовой выборке:")
visualize_predictions(model, test_loader)

def visualize_features(model, test_loader):
    """Визуализация активаций сверточных слоев"""
    model.eval()
    images, _ = next(iter(test_loader))
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.conv2.register_forward_hook(get_activation('conv2'))
    
    with torch.no_grad():
        _ = model(images[0:1])  
    
   
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0, 0].imshow(images[0].squeeze(), cmap='gray')
    axes[0, 0].set_title('Исходное изображение')
    axes[0, 0].axis('off')
    
    conv1_act = activations['conv1'][0]
    for i in range(min(8, conv1_act.size(0))):
        row = i // 4
        col = i % 4 + 1
        if row == 0 and col < 4:
            axes[row, col].imshow(conv1_act[i].cpu(), cmap='viridis')
            axes[row, col].set_title(f'Conv1 filter {i+1}')
            axes[row, col].axis('off')
    
    conv2_act = activations['conv2'][0]
    for i in range(min(8, conv2_act.size(0))):
        row = 1
        col = i % 4
        if col < 3:
            axes[row, col].imshow(conv2_act[i].cpu(), cmap='viridis')
            axes[row, col].set_title(f'Conv2 filter {i+1}')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

print("\nВизуализация активаций сверточных слоев:")
visualize_features(model, test_loader)