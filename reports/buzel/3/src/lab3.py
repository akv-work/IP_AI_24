import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from scipy.io import arff
from sklearn.neural_network import BernoulliRBM   # Новая библиотека

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense

# Отключаем слишком подробные логи TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Этап 0: Определение путей к файлам ---
MATERNAL_HEALTH_PATH = "D:/OIIS/lab3/Maternal Health Risk Data Set.csv"
RICE_DATA_PATH = "D:/OIIS/lab2/Rice_Cammeo_Osmancik.arff"

# --- Этап 1: Функции для загрузки и подготовки данных ---

def load_maternal_health_data(path):
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
        

def plot_training_history_v2(history_no_pre, history_ae, history_rbm, dataset_name):
    """
    Рисует подробные сравнительные графики обучения для ТРЕХ моделей,
    показывая и train, и validation метрики.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # --- График 1: Сравнение Accuracy (Точности) ---
    # Модель "Без предобучения"
    ax1.plot(history_no_pre.history['accuracy'], label='Без предоб. (Train)', color='red')
    ax1.plot(history_no_pre.history['val_accuracy'], label='Без предоб. (Val)', color='red', linestyle='--')
    
    # Модель "С предобучением (Автоэнкодер)"
    ax1.plot(history_ae.history['accuracy'], label='Автоэнкодер (Train)', color='blue')
    ax1.plot(history_ae.history['val_accuracy'], label='Автоэнкодер (Val)', color='blue', linestyle='--')

    # Модель "С предобучением (RBM)"
    ax1.plot(history_rbm.history['accuracy'], label='RBM (Train)', color='green')
    ax1.plot(history_rbm.history['val_accuracy'], label='RBM (Val)', color='green', linestyle='--')
    
    ax1.set_title('Сравнение Accuracy')
    ax1.set_xlabel('Эпохи')
    ax1.set_ylabel('Точность')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(bottom=0) # Начинаем ось Y с нуля для наглядности

    # --- График 2: Сравнение Loss (Ошибки) ---
    # Модель "Без предобучения"
    ax2.plot(history_no_pre.history['loss'], label='Без предоб. (Train)', color='red')
    ax2.plot(history_no_pre.history['val_loss'], label='Без предоб. (Val)', color='red', linestyle='--')

    # Модель "С предобучением (Автоэнкодер)"
    ax2.plot(history_ae.history['loss'], label='Автоэнкодер (Train)', color='blue')
    ax2.plot(history_ae.history['val_loss'], label='Автоэнкодер (Val)', color='blue', linestyle='--')

    # Модель "С предобучением (RBM)"
    ax2.plot(history_rbm.history['loss'], label='RBM (Train)', color='green')
    ax2.plot(history_rbm.history['val_loss'], label='RBM (Val)', color='green', linestyle='--')
    
    ax2.set_title('Сравнение Loss')
    ax2.set_xlabel('Эпохи')
    ax2.set_ylabel('Ошибка (Loss)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(bottom=0)

    fig.suptitle(f"Графики обучения для датасета: {dataset_name}", fontsize=16)
    plt.show()

# --- Этап 2: Логика обучения моделей ---

def create_dnn_classifier(input_dim, output_dim, layer_sizes):
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

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.1)
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("[Без] Обучение завершено.")
    return acc, f1, history # Возвращаем историю обучения

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
    return acc, f1, history # Возвращаем историю обучения

def run_training_with_rbm_pretraining(X_train, y_train, X_test, y_test, input_dim, output_dim, layer_sizes):
    """
    Эксперимент 3: Обучение DNN с предобучением с помощью Машин Больцмана (RBM).
    """
    print("\n--- 3. Обучение с предобучением RBM ---")

    # --- Шаг А: Предобучение слоев с помощью RBM ---
    print("[С RBM] Предобучение слоев...")
    
    # Создаём копию данных и нормализуем их для RBM.
    # При этом для основной сети будем использовать исходные стандартизированные данные.
    from sklearn.preprocessing import MinMaxScaler
    scaler_rbm = MinMaxScaler()
    X_train_rbm_scaled = scaler_rbm.fit_transform(X_train)

    pretrained_weights = []
    current_data = X_train_rbm_scaled
    
    for i, size in enumerate(layer_sizes):
        print(f"[С RBM] Обучение RBM для слоя {i+1} ({size} нейронов)...")
        # Создаем и обучаем RBM
        rbm = BernoulliRBM(n_components=size, n_iter=20, learning_rate=0.01, verbose=0, random_state=42)
        rbm.fit(current_data)
        
        # Извлекаем веса и смещения (bias) из обученной RBM.
        # У RBM они будут называться "components_" (веса) и "intercept_hidden_" (смещения).
        weights = rbm.components_.T 
        biases = rbm.intercept_hidden_
        pretrained_weights.append((weights, biases))
        
        # Преобразуем данные для подачи на вход следующей RBM
        current_data = rbm.transform(current_data)
        
    print("[С RBM] Предобучение завершено.")
    
    # --- Шаг Б: Дообучение ---
    model = create_dnn_classifier(input_dim, output_dim, layer_sizes)
    
    # Загружаем извлеченные веса в нашу итоговую модель
    for i, (weights, biases) in enumerate(pretrained_weights):
        model.layers[i].set_weights([weights, biases])
        
    print("[С RBM] Дообучение модели (20 эпох)...")
    # Дообучаем на исходных стандартизированных данных, а не на [0,1]
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("[С RBM] Дообучение завершено.")
    return acc, f1, history

# --- Этап 3: Полный цикл эксперимента ---

def run_full_experiment(dataset_name, X, y):
    """Запускает ВСЕ ТРИ вида обучения для одного датасета и выводит результаты."""
    print(f"\n{'='*70}\nЭксперимент для датасета: {dataset_name}\n{'='*70}")
    
    if X is None or y is None:
        return None

    # Разделение данных и стандартизация
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Определение архитектуры сети
    input_dim = X_train_scaled.shape[1]
    output_dim = len(np.unique(y))
    layer_sizes = [64, 32, 16]

    # --- Запуск всех трех экспериментов ---
    # Эксперимент 1 (из ЛР3)
    acc_no_pre, f1_no_pre, history_no_pre = run_training_without_pretraining(
        X_train_scaled, y_train, X_test_scaled, y_test, input_dim, output_dim, layer_sizes)
    
    # Эксперимент 2 (из ЛР3)
    acc_with_ae, f1_with_ae, history_with_ae = run_training_with_pretraining(
        X_train_scaled, y_train, X_test_scaled, y_test, input_dim, output_dim, layer_sizes)

    # Эксперимент 3 (для ЛР4)
    acc_with_rbm, f1_with_rbm, history_with_rbm = run_training_with_rbm_pretraining(
        X_train_scaled, y_train, X_test_scaled, y_test, input_dim, output_dim, layer_sizes)

    # --- Вывод результатов сравнения ---
    print("\n--- Результаты сравнения ---")
    print(f"Без предобучения    -> Acc: {acc_no_pre:.4f}, F1: {f1_no_pre:.4f}")
    print(f"С предоб. (Автоэнкодер) -> Acc: {acc_with_ae:.4f}, F1: {f1_with_ae:.4f}")
    print(f"С предоб. (RBM)       -> Acc: {acc_with_rbm:.4f}, F1: {f1_with_rbm:.4f}")
    
    # --- Вызов функции для отрисовки графиков ---
    # Передаем в нее все три истории обучения
    plot_training_history_v2(history_no_pre, history_with_ae, history_with_rbm, dataset_name)
    
    # --- Возвращаем словарь со всеми результатами ---
    return {
        "Без предоб. (Acc)": acc_no_pre, "Автоэнкодер (Acc)": acc_with_ae, "RBM (Acc)": acc_with_rbm,
        "Без предоб. (F1)": f1_no_pre, "Автоэнкодер (F1)": f1_with_ae, "RBM (F1)": f1_with_rbm
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