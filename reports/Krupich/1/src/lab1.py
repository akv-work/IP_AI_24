import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x

def imshow(img_tensor):
    npimg = img_tensor.squeeze().numpy()
    plt.imshow(npimg, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    unnormalize = transforms.Normalize((-1.0,), (2.0,))

    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
                                              shuffle=True, num_workers=2)

    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64,
                                             shuffle=False, num_workers=2)

    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

    model = SimpleCNN()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    loss_history = []
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f'Эпоха {epoch + 1}/{num_epochs}, Ошибка (Loss): {epoch_loss:.4f}')

        model.eval()
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
        train_accuracy = 100 * train_correct / train_total
        print(f'Точность на обучающей выборке: {train_accuracy:.2f} %')

    end_time = time.time()
    print('Обучение завершено')
    print(f'Время обучения: {((end_time - start_time) / 60):.2f} минут')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\nТочность на тестовой выборке: {accuracy:.2f} %')
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o', linestyle='-')
    plt.title('График изменения ошибки во время обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка (CrossEntropyLoss)')
    plt.xticks(range(1, num_epochs + 1))
    plt.grid(True)
    plt.show()

    print("\n--- Визуализация результата ---")
    
    dataiter = iter(test_loader)
    images_viz, labels_viz = next(dataiter)

    img_index = 5
    image_to_show = images_viz[img_index]
    true_label = labels_viz[img_index]

    model.eval()
    with torch.no_grad():
        output = model(image_to_show.to(device).unsqueeze(0))
        _, predicted_index = torch.max(output, 1)

    print(f'Реальный класс:      {classes[true_label]}')
    print(f'Предсказанный класс: {classes[predicted_index.item()]}')

    image_for_display = unnormalize(image_to_show.cpu())
    imshow(image_for_display)