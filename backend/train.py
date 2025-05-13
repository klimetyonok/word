import glob
import os

def train_model():
    data_files = glob.glob('data/*.wav')
    print(f"Обучение на {len(data_files)} файлов...")
    # Здесь ваш код обучения модели
    # Например, подготовьте датасет и переобучите модель
    print("Обучение завершено. Модель обновлена!")

if __name__ == "__main__":
    train_model()