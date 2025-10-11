# Predictive Maintenance — NASA CMAPSS

**Проект:** прогнозирование отказов реактивных двигателей (Predictive Maintenance) на основе датасета NASA CMAPSS.

---

## Краткое описание

Этот репозиторий содержит готовый проект для задачи предиктивного обслуживания (Predictive Maintenance) на основе официального набора данных **CMAPSS Jet Engine Simulated Data**. Цель — предсказывать вероятность отказа двигателя в ближайшие N циклов и показать рабочий пайплайн: загрузка данных, предобработка, обучение модели, оценка и визуализация результатов.

Проект полностью адаптирован для локального запуска на любой системе с Python 3.9+.

---

## Особенности проекта

* ✅ Работа с официальным архивом NASA CMAPSS (FD001–FD004)
* ✅ Структура кода разделена на логические модули

---

## Датасет

**NASA CMAPSS** — синтетические данные деградации реактивных двигателей (CMAPSS Jet Engine Simulated Data). В проекте используется поднабор `FD001`.

Структура исходных данных (`train_FD001.txt`):

* `unit_number` — идентификатор двигателя
* `time_in_cycles` — номер рабочего цикла
* `op_setting_1..3` — операционные параметры
* `sensor_1..21` — данные с сенсоров

Для бинарной классификации формируется метка `label = 1`, если `RUL <= threshold` (по умолчанию `threshold = 30` циклов), где `RUL = max_cycle_for_unit - current_cycle`.

---

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/maximDprogram/Predictive-Maintenance-NASA-CMAPSS
cd Predictive_Maintenance_NASA_CMAPSS
```

### 2. Запуск

```bash
python -m venv venv
source venv/bin/activate  # или venv\Scripts\activate на Windows
pip install scikit-learn pandas matplotlib seaborn requests
python main.py
```
---

## 📂 Структура проекта
```
Predictive_Maintenance_NASA_CMAPSS/
├─ 📂 model/                  # Сохранённая модель
│   └─ rf_model_fd001.pkl
├─ 📂 results/                # Графики, отчёты
│   ├─ classification_report.txt
│   ├─ confusion_matrix.png
│   ├─ feature_importances.png
├─ data_loader.py
├─ evaluate.py
├─ model_train.py
├─ preprocessing.py
├─ README.md               # Описание проекта, инструкция по запуску
└─ main.py
```

## Источники

* **NASA CMAPSS dataset:** CMAPSS Jet Engine Simulated Data
