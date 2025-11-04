import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from scipy.io import arff

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense

# Отключаем слишком подробные логи TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Этап 0: Определение путей к файлам ---
MATERNAL_HEALTH_PATH = "D:/ОИИС/lab3/Maternal Health Risk Data Set.csv"
RICE_DATA_PATH = "D:/ОИИС/lab2/Rice_Cammeo_Osmancik.arff"

# --- Этап 1: Функции для загрузки и подготовки данных ---

def load_maternal_health_data(path):
    # (Код без изменений)
    print("\nЗагрузка датасета 'Maternal Health Risk'...")
    try:
        data = pd.read_csv(path)
        X = data.drop('RiskLevel', axis=1)
        y = data['RiskLevel']
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.values.ravel())
        print(f"Признаков: {X.shape[1]}, Классов: {len(np.unique(y_encoded))}, Объектов: {X.shape[0]}")
        return X.values, y_encoded
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {path}")
        return None, None

def load_rice_data(path):
    # (Код без изменений)
    print("\nЗагрузка датасета 'Rice (Cammeo and Osmancik)'...")
    try:
        data_arff, meta = arff.loadarff(path)
        data = pd.DataFrame(data_arff)
        data['Class'] = data['Class'].apply(lambda x: x.decode('utf-8'))
        X = data.drop('Class', axis=1)
        y = data['Class']
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.values.ravel())
        print(f"Признаков: {X.shape[1]}, Классов: {len(np.unique(y_encoded))}, Объектов: {X.shape[0]}")
        return X.values, y_encoded
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {path}")
        return None, None
        
# --- Новая функция для визуализации ---

def plot_training_history(history_no_pre, history_with_pre, dataset_name):
    """
    Рисует сравнительные графики обучения для двух моделей.
    - history_no_pre: объект History от модели без предобучения.
    - history_with_pre: объект History от модели с предобучением.
    - dataset_name: Название датасета для заголовка.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # График 1: Сравнение Accuracy (Точности)
    ax1.plot(history_no_pre.history['accuracy'], label='Без предобучения (Train Acc)', color='r')
    ax1.plot(history_no_pre.history['val_accuracy'], label='Без предобучения (Val Acc)', linestyle='--', color='salmon')
    ax1.plot(history_with_pre.history['accuracy'], label='С предобучением (Train Acc)', color='b')
    ax1.plot(history_with_pre.history['val_accuracy'], label='С предобучением (Val Acc)', linestyle='--', color='skyblue')
    ax1.set_title('Сравнение Accuracy')
    ax1.set_xlabel('Эпохи')
    ax1.set_ylabel('Точность')
    ax1.legend()
    ax1.grid(True)

    # График 2: Сравнение Loss (Ошибки)
    ax2.plot(history_no_pre.history['loss'], label='Без предобучения (Train Loss)', color='r')
    ax2.plot(history_no_pre.history['val_loss'], label='Без предобучения (Val Loss)', linestyle='--', color='salmon')
    ax2.plot(history_with_pre.history['loss'], label='С предобучением (Train Loss)', color='b')
    ax2.plot(history_with_pre.history['val_loss'], label='С предобучением (Val Loss)', linestyle='--', color='skyblue')
    ax2.set_title('Сравнение Loss')
    ax2.set_xlabel('Эпохи')
    ax2.set_ylabel('Ошибка (Loss)')
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(f"Графики обучения для датасета: {dataset_name}", fontsize=16)
    plt.show()

# --- Этап 2: Логика обучения моделей (с небольшими изменениями) ---

def create_dnn_classifier(input_dim, output_dim, layer_sizes):
    # (Код без изменений)
    model = Sequential([
        Dense(layer_sizes[0], activation='relu', input_shape=(input_dim,)),
        Dense(layer_sizes[1], activation='relu'),
        Dense(layer_sizes[2], activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_training_without_pretraining(X_train, y_train, X_test, y_test, input_dim, output_dim, layer_sizes):
    print("\n--- 1. Обучение без предобучения ---")
    model = create_dnn_classifier(input_dim, output_dim, layer_sizes)
    print("[Без] Обучение модели (50 эпох)...")
    # Добавляем validation_split, чтобы получить val_accuracy и val_loss
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.1)
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("[Без] Обучение завершено.")
    return acc, f1, history # <-- Возвращаем историю обучения

def run_training_with_pretraining(X_train, y_train, X_test, y_test, input_dim, output_dim, layer_sizes):
    print("\n--- 2. Обучение с предобучением ---")
    print("[С предоб.] Предобучение автоэнкодерами...")
    encoders = []
    current_data = X_train
    temp_input_dim = input_dim

    for i, size in enumerate(layer_sizes):
        input_layer = Input(shape=(temp_input_dim,))
        encoded_layer = Dense(size, activation='relu')(input_layer)
        decoded_layer = Dense(temp_input_dim, activation='linear')(encoded_layer)
        autoencoder = Model(input_layer, decoded_layer)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(current_data, current_data, epochs=30, batch_size=64, verbose=0)
        encoder = Model(input_layer, encoded_layer)
        encoders.append(encoder)
        current_data = encoder.predict(current_data)
        temp_input_dim = size
    
    print("[С предоб.] Предобучение завершено.")
    
    model = create_dnn_classifier(input_dim, output_dim, layer_sizes)
    for i, encoder in enumerate(encoders):
        model.layers[i].set_weights(encoder.layers[1].get_weights())
        
    print("[С предоб.] Дообучение модели (20 эпох)...")
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("[С предоб.] Дообучение завершено.")
    return acc, f1, history # <-- Возвращаем историю обучения

# --- Этап 3: Полный цикл эксперимента (с небольшими изменениями) ---

def run_full_experiment(dataset_name, X, y):
    print(f"\n{'='*70}\nЭксперимент для датасета: {dataset_name}\n{'='*70}")
    if X is None or y is None: return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    input_dim = X_train_scaled.shape[1]
    output_dim = len(np.unique(y))
    layer_sizes = [64, 32, 16]

    # Получаем историю обучения от обеих моделей
    acc_no_pre, f1_no_pre, history_no_pre = run_training_without_pretraining(X_train_scaled, y_train, X_test_scaled, y_test, input_dim, output_dim, layer_sizes)
    acc_with_pre, f1_with_pre, history_with_pre = run_training_with_pretraining(X_train_scaled, y_train, X_test_scaled, y_test, input_dim, output_dim, layer_sizes)

    print("\n--- Результаты сравнения ---")
    print(f"Без предобучения → Acc: {acc_no_pre:.4f}, F1: {f1_no_pre:.4f}")
    print(f"С предобучением  → Acc: {acc_with_pre:.4f}, F1: {f1_with_pre:.4f}")
    acc_improvement = acc_with_pre - acc_no_pre
    f1_improvement = f1_with_pre - f1_no_pre
    print(f"Улучшение: Acc {acc_improvement:+.4f}, F1 {f1_improvement:+.4f}")
    
    # Вызываем новую функцию для отрисовки графиков
    plot_training_history(history_no_pre, history_with_pre, dataset_name)
    
    return {
        "Без Acc": acc_no_pre, "С Acc": acc_with_pre, "ΔAcc": acc_improvement,
        "Без F1": f1_no_pre, "С F1": f1_with_pre, "ΔF1": f1_improvement
    }

# --- Этап 4: Запуск всех экспериментов и итоговая таблица ---
if __name__ == '__main__':
    all_results = {}
    
    X_maternal, y_maternal = load_maternal_health_data(MATERNAL_HEALTH_PATH)
    if X_maternal is not None:
        all_results['Maternal Health'] = run_full_experiment('Maternal Health', X_maternal, y_maternal)
    
    X_rice, y_rice = load_rice_data(RICE_DATA_PATH)
    if X_rice is not None:
        all_results['Rice'] = run_full_experiment('Rice', X_rice, y_rice)
    
    if all_results:
        final_df = pd.DataFrame.from_dict(all_results, orient='index')
        print(f"\n\n{'='*70}\nИтоговая таблица\n{'='*70}")
        print(final_df.to_string())