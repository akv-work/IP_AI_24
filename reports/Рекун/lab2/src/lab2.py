import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    DATA_DIR = './data'
    BATCH_SIZE = 64  
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    NUM_EPOCHS = 3  
    
   
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
    
    
    transform_train = transforms.Compose([
        transforms.Resize(128),  
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(128),  
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    
    print("Загрузка Fashion-MNIST dataset...")
    train_dataset = torchvision.datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Размер обучающей выборки: {len(train_dataset)}")
    print(f"Размер тестовой выборки: {len(test_dataset)}")
    
   
    print("Инициализация упрощенной модели...")
    
    class SimpleResNet(nn.Module):
        def __init__(self, num_classes=10):
            super(SimpleResNet, self).__init__()

            original_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(
                *list(original_model.children())[:-2]  
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
            
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    model = SimpleResNet(num_classes=10)
    model = model.to(device)
    print(f"Модель загружена на {device}")
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=LEARNING_RATE, 
        momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY
    )
    
    train_losses = []
    test_accuracies = []
    
    print("\nНачало обучения...")
    print("=" * 60)
    
    for epoch in range(NUM_EPOCHS):

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
        
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
           
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_train / total_train:.2f}%'
            })
    
        avg_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_loss)
        
     
        model.eval()
        correct_test = 0
        total_test = 0
        
        test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Test]')
        
        with torch.no_grad():
            for images, labels in test_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
               
                test_pbar.set_postfix({
                    'Acc': f'{100 * correct_test / total_test:.2f}%'
                })
    
        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch [{epoch+1:2d}/{NUM_EPOCHS}] | '
              f'Train Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.2f}% | '
              f'Test Acc: {test_accuracy:.2f}%')
    
    print("=" * 60)
    print("Обучение завершено!")
    
    model.eval()
    final_correct = 0
    final_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Final Evaluation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            final_total += labels.size(0)
            final_correct += (predicted == labels).sum().item()
    
    final_accuracy = 100 * final_correct / final_total
    print(f'\nФинальная точность на тестовой выборке: {final_accuracy:.2f}%')
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, 'b-', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPOCHS + 1), test_accuracies, 'g-', marker='o')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    def quick_visualize_predictions(model, test_loader, num_images=8):
        model.eval()
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(images[:num_images])
            _, predicted = torch.max(outputs, 1)
        
        images = images.cpu()
        predicted = predicted.cpu()
        labels = labels.cpu()
        
        plt.figure(figsize=(12, 6))
        for i in range(num_images):
            plt.subplot(2, 4, i + 1)
            
            img = images[i].squeeze().permute(1, 2, 0)
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            img = torch.clamp(img, 0, 1)
            
            plt.imshow(img)
            
            true_label_name = fashion_mnist_classes[labels[i].item()]
            pred_label_name = fashion_mnist_classes[predicted[i].item()]
            
            color = 'green' if labels[i] == predicted[i] else 'red'
            plt.title(f'True: {true_label_name}\nPred: {pred_label_name}', 
                     color=color, fontsize=10)
            plt.axis('off')
        
        plt.suptitle(f'Предсказания модели (Accuracy: {final_accuracy:.1f}%)', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        correct = (labels[:num_images] == predicted).sum().item()
        print(f'Примеры предсказаний: {correct}/{num_images} правильных')
    
    print("\nВизуализация предсказаний:")
    quick_visualize_predictions(model, test_loader)
    
    # Выводы
    print("\n" + "="*60)
    print("ВЫВОДЫ ПО ЛАБОРАТОРНОЙ РАБОТЕ №2")
    print("="*60)
    print(f"Предобученная ResNet18 (упрощенная): {final_accuracy:.2f}%")
    print(f"Кастомная CNN из ЛР1: 92.66%")
    print(f"State-of-the-art: ~96.7%")

if __name__ == '__main__':
    main()