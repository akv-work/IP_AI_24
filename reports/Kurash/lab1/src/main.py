import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- Параметры ---
batch_size = 128
num_epochs = 20
lr = 1e-3
model_path = "cifar_simple_cnn.pth"
num_classes = 10
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)

# --- Модель (с BatchNorm для стабильности) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Утилиты для денормализации и отображения ---
def denormalize(img_tensor, mean, std):
    # img_tensor: Tensor [C,H,W] or [B,C,H,W], values in normalized range
    if img_tensor.dim() == 4:
        img = img_tensor.clone().cpu().numpy()
        img = img * np.array(std)[:, None, None] + np.array(mean)[:, None, None]
        return img
    else:
        img = img_tensor.clone().cpu().numpy()
        img = img * np.array(std)[:, None, None] + np.array(mean)[:, None, None]
        return img

def save_batch_grid(X, preds, labels, classes, fname="pred_grid.png", nrow=8):
    # X: tensor [B,C,H,W] in normalized form
    grid = vutils.make_grid(X[:nrow*nrow], nrow=nrow, padding=2)
    img = denormalize(grid, mean, std)  # shape [C,H,W]
    img = np.transpose(img, (1,2,0))
    img = np.clip(img, 0, 1)
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.axis('off')
    # draw labels under images
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print("Saved batch grid to", fname)
    plt.close()

# --- evaluation helpers ---
def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            running_loss += loss.item() * X.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return running_loss / total, correct / total

def per_class_accuracy(model, loader, device, num_classes):
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            preds = out.argmax(dim=1)
            for i in range(y.size(0)):
                label = y[i].item()
                class_total[label] += 1
                class_correct[label] += (preds[i] == y[i]).item()
    accs = [ (class_correct[i] / class_total[i]) if class_total[i]>0 else 0.0 for i in range(num_classes) ]
    return accs

# --- main ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = True if device.type == 'cuda' else False
    num_workers = min(4, os.cpu_count() or 1)

    # --- трансформации (можно добавить AutoAugment, Cutout, MixUp отдельно) ---
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # --- датасеты и загрузчики ---
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    classes = train_set.classes

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_mem)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_mem)

    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        # --- Инициализируем счётчики В НАЧАЛЕ ЭПОХИ ---
        running_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)   # суммируем по примерам
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        # --- усредняем по числу примеров (в конце эпохи) ---
        train_loss = running_loss / total
        train_acc  = correct / total

        test_loss, test_acc = evaluate(model, test_loader, device, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch:2d}/{num_epochs}  Train loss: {train_loss:.4f}  Train acc: {train_acc:.4f}  Test loss: {test_loss:.4f}  Test acc: {test_acc:.4f}")

    # save
    torch.save(model.state_dict(), model_path)
    print("Model saved to", model_path)

    # графики
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(range(1,len(train_losses)+1), train_losses, label="train loss")
    plt.plot(range(1,len(test_losses)+1), test_losses, label="test loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss")

    plt.subplot(1,2,2)
    plt.plot(range(1,len(train_accs)+1), train_accs, label="train acc")
    plt.plot(range(1,len(test_accs)+1), test_accs, label="test acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Training curves saved to training_curves.png")
    plt.close()

    # --- per-class accuracy ---
    accs = per_class_accuracy(model, test_loader, device, num_classes)
    for cls, a in zip(classes, accs):
        print(f"{cls:10s}: {a*100:5.2f}%")

    # --- grid of predictions (сохраняем в файл) ---
    # возьмём 64 примера из теста и сохраним картинку
    model.eval()
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            out = model(Xb)
            preds = out.argmax(dim=1).cpu().numpy()
            save_batch_grid(Xb.cpu(), preds, yb.numpy(), classes, fname="pred_grid.png", nrow=8)
            break

    # --- predict single image (пример) ---
    def predict_image(image_path, topk=5):
        model.eval()
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize((32,32))
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        x = t(img_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top_idx = probs.argsort()[::-1][:topk]
        print("Top predictions for", image_path)
        for i in top_idx:
            print(f"  {classes[i]:10s}: {probs[i]*100:.2f}%")
        # показываем исходное изображение (не нормализованное)
        plt.figure(figsize=(3,3))
        plt.imshow(np.array(img))
        plt.axis('off')
        plt.title(f"Pred: {classes[top_idx[0]]} ({probs[top_idx[0]]*100:.1f}%)")
        plt.savefig("single_pred.png", dpi=150)
        print("Saved prediction view to single_pred.png")
        plt.close()

    # Пример: укажи свой путь к image.png
    example_img = "C:\\Users\\User\\OneDrive\\Desktop\\IP_AI_24\\reports\\Kurash\\lab1\\src\\image.png"
    if os.path.exists(example_img):
        predict_image(example_img)
    else:
        print("Example image not found:", example_img)

if __name__ == "__main__":
    main()
