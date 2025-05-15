# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем необходимые зависимости
RUN pip install --upgrade pip
RUN pip install transformers torch flask

# Создаем рабочую директорию
WORKDIR /app

# Копируем файлы в контейнер
COPY . /app

# Открываем порт для Flask приложения
EXPOSE 5000

# Запуск Flask API
CMD ["python", "app.py"]
