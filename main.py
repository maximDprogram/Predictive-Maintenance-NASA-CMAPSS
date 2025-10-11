from data_loader import download_and_extract
from preprocessing import processing
from model_train import train_random_forest
from evaluate import evaluate_model

import os
import joblib

def main():
    url = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
    zip_path = "CMAPSS.zip"
    data_dir = "CMAPSS"
    model_dir = "model"
    results_dir = "results"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Скачивание и распаковка
    download_and_extract(url, zip_path)
    
    # 2. Загрузка данных
    x_train_s, x_test_s, y_train, y_test, scaler, x_test, x_train = processing(data_dir)
    
    # 3. Обучение модели
    rf = train_random_forest(x_train_s, y_train)

    model_path = os.path.join(model_dir, "rf_model_fd001.pkl")
    joblib.dump(rf, model_path)
    
    # 4. Оценка
    evaluate_model(rf, x_test_s, y_test, x_test, scaler, x_train, results_dir)

    print(f"Модель сохранена: {model_path}")
    
if __name__ == "__main__":
    main()