import zipfile
import os
import requests
from requests.exceptions import RequestException
import time

def download_and_extract(url: str, zip_path: str, max_retries: int = 5):
    
    # Создаём папку CMAPSS, если её нет
    extract_dir = "CMAPSS"
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)

        # Пытаемся скачать файл с повторными попытками
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Попытка {attempt}: скачиваем архив...")
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(zip_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                print("Файл успешно скачан.")
                break  # выходим из цикла, если скачивание прошло
            except RequestException as e:
                print(f"Ошибка при скачивании: {e}")
                if attempt == max_retries:
                    raise
                print("Повтор через 5 секунд...")
                time.sleep(5)

        # Распаковываем ZIP
        print("Распаковываем архив...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Архив распакован.")
    else:
        print("Папка CMAPSS уже существует — пропускаем скачивание.")

    # Список файлов
    print("Файлы в CMAPSS:")
    print(os.listdir(extract_dir))