import argparse, os, time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models


def create_squeezenet(num_classes=10, freeze_backbone=False):
    model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    model.num_classes = num_classes
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    return model

def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return running_loss / total, 100.0 * correct / total


def predict_image(path, model, device, transform, input_size, classes):
    img = Image.open(path).convert('RGB')
    img_resized = img.resize((input_size, input_size))
    x = transform(img_resized).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
    return img, pred, probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--save-dir', type=str, default='checkpoints_cifar10')
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--visualize', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint in save-dir')
    parser.add_argument('--resume-path', type=str, default=None,
                        help='Path to specific checkpoint to resume from')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    os.makedirs(args.save_dir, exist_ok=True)

    input_size = 224
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    classes = train_set.classes

    model = create_squeezenet(num_classes=10, freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(5, args.epochs // 2), gamma=0.1)

    start_epoch = 1
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}

    if args.resume or args.resume_path:
        ckpt_path = args.resume_path
        if args.resume and ckpt_path is None:
            files = [f for f in os.listdir(args.save_dir) if f.endswith('.pth')]
            if files:
                files = sorted(files, key=lambda x: int(x.split('epoch')[-1].split('.')[0]))
                ckpt_path = os.path.join(args.save_dir, files[-1])
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"Loading checkpoint {ckpt_path} ...")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            if 'optimizer_state' in ckpt and not args.visualize:
                optimizer.load_state_dict(ckpt['optimizer_state'])
            start_epoch = ckpt['epoch'] + 1

            if 'history' in ckpt:
                history = ckpt['history']
                print(f"Loaded history with {len(history['train_loss'])} epochs")
            else:
                history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
                print("No history found in checkpoint")

            print(f"Resumed from epoch {ckpt['epoch']}")
        else:
            print("Чекпоинт не найден, обучение начнётся с нуля")

    if args.visualize:
        img, pred_idx, probs = predict_image(
            args.visualize, model, device, test_transform, input_size, classes
        )
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Pred: {classes[pred_idx]} ({probs[pred_idx] * 100:.1f}%)')
        plt.savefig("prediction.png", dpi=150)
        print("Картинка с предсказанием сохранена в prediction.png")
        return

    start_time = time.time()

    if start_epoch > args.epochs:
        print(f"Чекпоинт уже содержит {start_epoch - 1} эпох, что больше запрошенных {args.epochs}")
        print("Отображаем историю всех эпох")
    else:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)

            train_loss = running_loss / len(train_loader.dataset)
            test_loss, test_acc = evaluate(model, test_loader, device, criterion)
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            scheduler.step()

            print(
                f'Epoch {epoch}/{args.epochs} | TrainLoss {train_loss:.4f} | TestLoss {test_loss:.4f} | TestAcc {test_acc:.2f}%')

            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'history': history
            }, os.path.join(args.save_dir, f'squeezenet_epoch{epoch}.pth'))

        total_time = time.time() - start_time
        print('Training finished in {:.2f} min'.format(total_time / 60.0))

    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o', markersize=3)
    plt.plot(epochs_range, history['test_loss'], label='Test Loss', linestyle='--', marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['test_acc'], label='Test Accuracy', marker='^', markersize=3, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(args.save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=150)
    print(f'✅ History plot saved to {save_path} (showing {len(history["train_loss"])} epochs)')

    if history['test_acc']:
        final_acc = history['test_acc'][-1]
        final_loss = history['test_loss'][-1]
        print(f'📊 Final results: Test Accuracy = {final_acc:.2f}%, Test Loss = {final_loss:.4f}')




if __name__ == "__main__":
    main()