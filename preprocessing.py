import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def processing(data_dir: str):

    # Загружаем один из поднаборов (FD001)
    train_file = os.path.join(data_dir, "train_FD001.txt")
    test_file  = os.path.join(data_dir, "test_FD001.txt")
    rul_file   = os.path.join(data_dir, "RUL_FD001.txt")

    # Определяем имена колонок
    cols = ['unit_number', 'time_in_cycles'] + \
        [f'op_setting_{i}' for i in range(1, 4)] + \
        [f'sensor_{i}' for i in range(1, 22)]

    train_df = pd.read_csv(train_file, sep=r'\s+', header=None, names=cols)
    test_df  = pd.read_csv(test_file,  sep=r'\s+', header=None, names=cols)
    rul_df   = pd.read_csv(rul_file, sep=r'\s+', header=None, names=['RUL'])

    print(f"Размер train: {train_df.shape}")
    print(f"Размер test:  {test_df.shape}")
    print(f"Размер rul:   {rul_df.shape}")

    # Добавляем оставшийся срок службы (RUL) для train
    rul_train = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    rul_train.columns = ['unit_number', 'max_cycle']
    train_df = train_df.merge(rul_train, on='unit_number', how='left')
    train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']

    # Целевая метка: 1 — скоро отказ (≤30 циклов)
    train_df['label'] = (train_df['RUL'] <= 30).astype(int)

    # Подготовка test: добавляем реальные RUL из rul_df
    # Находим последний цикл для каждого двигателя в тесте
    rul_test = test_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    rul_test.columns = ['unit_number', 'last_cycle']

    # Добавляем реальные RUL из файла rul_df (порядок совпадает)
    rul_test['RUL'] = rul_df['RUL']

    # Объединяем с test_df и пересчитываем RUL для каждой строки
    test_df = test_df.merge(rul_test, on='unit_number', how='left')
    test_df['RUL'] = test_df['RUL'] + (test_df['last_cycle'] - test_df['time_in_cycles'])

    # Метка для теста
    test_df['label'] = (test_df['RUL'] <= 30).astype(int)

    # Формируем признаки и целевые переменные
    features = [c for c in train_df.columns if c.startswith('op_setting_') or c.startswith('sensor_')]

    x_train = train_df[features]
    y_train = train_df['label']

    x_test  = test_df[features]
    y_test  = test_df['label']

    # Масштабируем признаки
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s  = scaler.transform(x_test)
    
    return x_train_s, x_test_s, y_train, y_test, scaler, x_test, x_train