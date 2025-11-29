import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# --- ШАГ 1: Определение архитектур ---

# 1.1. Копируем класс из ЛБ1
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 классов на выходе

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 1.2. Создаем "скелет" для модели ResNet34 из ЛБ2
def create_resnet_model():
    model = models.resnet34(weights=None) # Загружаем архитектуру без предобученных весов
    num_ftrs = model.fc.in_features
    # Заменяем последний слой
    model.fc = nn.Linear(num_ftrs, 10)
    return model

# --- ШАГ 2: Подготовка инстументов ---

# 2.1. Определяем список классов CIFAR-10 
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# 2.2. Создаем два разных набора трансформаций.
# Каждая модель получает изображение в том формате в котором она обучалась

# Трансформации для кастомной модели (32x32, нормализация 0.5)
transform_custom = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Трансформации для ResNet модели (224x224, нормализация ImageNet)
transform_resnet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- ШАГ 3: Загрузка моделей ---

# 3.1. Загрузка кастомной модели
custom_model = SimpleCNN()
custom_model.load_state_dict(torch.load('D:/REALOIIS/lab2/cifar_custom_cnn.pth')) 
# Переводим модель в режим оценки
custom_model.eval()

# 3.2. Загрузка ResNet модели
resnet_model = create_resnet_model()
resnet_model.load_state_dict(torch.load('D:/REALOIIS/lab2/cifar_resnet34.pth'))
resnet_model.eval()

# --- ШАГ 4: Функция для предсказания ---

def predict_image(image_path, model, transform, model_name):
    try:
        # Открываем изображение и конвертируем в RGB 
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"ОШИБКА: Файл не найден по пути '{image_path}'. Проверьте имя файла.")
        return

    print(f"\n--- Предсказание моделью: {model_name} ---")
    
    # Показываем исходное изображение
    plt.imshow(image)
    plt.title(f"Исходное изображение для {model_name}")
    plt.axis('off')
    plt.show()

    # Применяем нужные трансформации и добавляем batch размерность 
    image_tensor = transform(image).unsqueeze(0)
    
    # Делаем предсказание (без вычисления градиентов)
    with torch.no_grad():
        outputs = model(image_tensor)
        # Применяем Softmax для получения вероятности
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        # Находим класс с максимальной вероятностью
        _, predicted_idx = torch.max(outputs, 1)
        
    predicted_class = classes[predicted_idx.item()]
    confidence = probabilities[predicted_idx.item()].item() * 100
    
    print(f"Результат: модель думает, что это '{predicted_class}'")
    print(f"Уверенность: {confidence:.2f}%")

# --- ШАГ 5: Запуск предсказаний ---

if __name__ == '__main__':
    path_to_image = 'D:/REALOIIS/lab2/justadog.png' 

    # Вызываем функцию предсказания для каждой модели
    predict_image(path_to_image, custom_model, transform_custom, "Кастомная СНС (ЛБ1)")
    predict_image(path_to_image, resnet_model, transform_resnet, "Fine-tuned ResNet34 (ЛБ2)")