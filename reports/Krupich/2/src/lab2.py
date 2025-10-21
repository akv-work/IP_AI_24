import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet34, ResNet34_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f'Train size: {len(train_dataset)}')
print(f'Test size: {len(test_dataset)}')

model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
model = model.to(device)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def test_model(model, loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Testing', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = test_loss / len(loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

num_epochs = 10
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
train_accs = []
test_losses = []
test_accs = []

print('\n=== Training ResNet34 ===')
start_time = time.time()

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    test_loss, test_acc = test_model(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

training_time = time.time() - start_time
print(f'\nTraining time: {training_time:.2f} seconds')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train Loss', marker='o')
ax1.plot(test_losses, label='Test Loss', marker='s')
ax1.set_title('ResNet34 - Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(train_accs, label='Train Acc', marker='o')
ax2.plot(test_accs, label='Test Acc', marker='s')
ax2.set_title('ResNet34 - Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resnet34_training.png', dpi=300, bbox_inches='tight')
plt.show()

print('\n' + '='*70)
print('RESULTS')
print('='*70)
print(f'Final Test Accuracy: {test_accs[-1]:.2f}%')
print(f'Training Time: {training_time:.2f}s')

torch.save(model.state_dict(), 'resnet34_fashion_mnist.pth')
print('\nModel saved: resnet34_fashion_mnist.pth')

def visualize_predictions(model, loader, num_images=10):
    model.eval()
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)
    
    for idx in range(num_images):
        img = images[idx].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = img[0].numpy()
        
        axes[idx].imshow(img, cmap='gray')
        axes[idx].axis('off')
        
        true_label = classes[labels[idx]]
        pred_label = classes[predicted[idx]]
        color = 'green' if labels[idx] == predicted[idx] else 'red'
        
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=9)
    
    plt.tight_layout()
    return fig

fig = visualize_predictions(model, test_loader, 10)
fig.suptitle('ResNet34 - Predictions', fontsize=14, fontweight='bold', y=1.02)
plt.savefig('resnet34_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

def evaluate_per_class(model, loader, device):
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
    
    return class_correct, class_total

class_correct, class_total = evaluate_per_class(model, test_loader, device)

print('\n' + '='*70)
print('PER-CLASS ACCURACY')
print('='*70)
print(f'{"Class":<20} {"Accuracy":<15}')
print('-'*35)
for i, class_name in enumerate(classes):
    acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f'{class_name:<20} {acc:>13.2f}%')

print('\nFiles saved:')
print('- resnet34_fashion_mnist.pth')
print('- resnet34_training.png')
print('- resnet34_predictions.png')