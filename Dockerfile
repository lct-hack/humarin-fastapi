# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Нужны только базовые runtime-пакеты (шрифты для matplotlib, tzdata)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core tzdata \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# код
COPY api_server.py processing.py /app/

# каталоги данных
RUN mkdir -p /app/hypoxia /app/regular

EXPOSE 8000

# Один воркер — важен для SSE
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
