import argparse
import os
import time
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 8x8

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), 
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    # -----------------------
    # Аргументы
    # -----------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--visualize', type=str, default=None)
    parser.add_argument('--resume', action='store_true', help='Продолжить обучение с best.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # -----------------------
    # Data (CIFAR-10 + расширенная аугментация)
    # -----------------------
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)

    classes = trainset.classes

    # -----------------------
    # Модель/оптимизатор/критерий
    # -----------------------
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -----------------------
    # Визуализация картинки
    # -----------------------
    def predict_image(img_path):
        img = Image.open(img_path).convert('RGB').resize((32, 32))
        x = transform_test(img).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
        return img, pred, probs

    # -----------------------
    # Загрузка из чекпоинта (если нужно)
    # -----------------------
    if args.resume:
        checkpoint_path = os.path.join(args.save_dir, 'best.pth')
        if os.path.isfile(checkpoint_path):
            print(f"Загрузка модели из {checkpoint_path} ...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            best_acc = checkpoint.get('acc', 0.0)
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"Модель загружена. Лучший acc={best_acc:.2f}% (эпоха {start_epoch-1})")
        else:
            print("Чекпоинт не найден, начинаем обучение с нуля.")
            best_acc = 0.0
            start_epoch = 1

        if args.visualize:
            img, pred_idx, probs = predict_image(args.visualize)
            plt.figure(figsize=(3, 3))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Pred: {classes[pred_idx]} ({probs[pred_idx] * 100:.1f}%)')
            plt.savefig("prediction.png", dpi=150)
            print("✅ Картинка с предсказанием сохранена в prediction.png")
            return
    else:
        best_acc = 0.0
        start_epoch = 1

    # -----------------------
    # Функция валидации
    # -----------------------
    def evaluate(loader):
        model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return running_loss / total, 100.0 * correct / total

    # -----------------------
    # Обучение
    # -----------------------
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(trainloader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(trainloader.dataset)
        test_loss, test_acc = evaluate(testloader)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f'Epoch {epoch}/{args.epochs}  TrainLoss={train_loss:.4f}  TestLoss={test_loss:.4f}  TestAcc={test_acc:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model_state': model.state_dict(),
                        'acc': best_acc,
                        'epoch': epoch}, os.path.join(args.save_dir, 'best.pth'))

    total_time = time.time() - start_time
    print(f'Training finished in {total_time/60:.2f} minutes. Best test acc: {best_acc:.2f}%')

    # -----------------------
    # Графики
    # -----------------------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, len(history['test_loss']) + 1), history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history['test_acc']) + 1), history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Test Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_history.png'), dpi=150)
    print(f'History plot saved to {os.path.join(args.save_dir, "training_history.png")}')




if __name__ == "__main__":
    main()
