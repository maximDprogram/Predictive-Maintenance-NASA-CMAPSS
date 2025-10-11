import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import os

def evaluate_model(rf, x_test_s, y_test, x_test, scaler, x_train, results_dir):
    
    # Оценка точности
    y_pred = rf.predict(x_test_s)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nТочность модели на TEST наборе: {accuracy*100:.2f}%")

    print("\nКлассификационный отчёт:")
    

    report = classification_report(y_test, y_pred, target_names=["OK", "Скорый отказ"])
    report_path = os.path.join(results_dir, "classification_report.txt")

    with open(report_path, "w") as f:
        f.write(report)

    print(f"Классификационный отчёт сохранён: {report_path}")
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    cm_disp_path = os.path.join(results_dir, "confusion_matrix.png")
    disp = ConfusionMatrixDisplay(cm, display_labels=["OK", "Скорый отказ"])
    disp.plot(cmap="Blues")
    plt.title("Матрица ошибок — RandomForest (FD001 Test)")
    plt.savefig(cm_disp_path)
    plt.close()
    print(f"Матрица ошибок сохранена: {cm_disp_path}")


    # Важность признаков
    importances = pd.Series(rf.feature_importances_, index=x_train.columns).sort_values(ascending=False)
    importance_fig_path = os.path.join(results_dir, "feature_importances.png")

    plt.figure(figsize=(10,5))
    sns.barplot(x=importances[:10], y=importances.index[:10])
    plt.title("Топ-10 наиболее значимых сенсоров (FD001)")
    plt.xlabel("Важность признака")
    plt.ylabel("Сенсор")
    plt.tight_layout()
    plt.savefig(importance_fig_path)
    plt.close()
    print(f"График важности признаков сохранён: {importance_fig_path}")

    # Пример прогноза на одном двигателе
    sample = x_test.iloc[[0]]
    prob = rf.predict_proba(scaler.transform(sample))[0, 1]
    print(f"\nВероятность отказа в ближайшие 30 циклов: {prob:.2%}")