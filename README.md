# КТГ Мониторинг API

Эмуляция работы монитора для анализа кардиотокограммы (ЧСС плода и активности матки).  
Сервис реализован на **FastAPI** и поддерживает потоковую передачу моментов и периодические аннотации.  

## Содержание

- [Описание](#описание)  
- [Структура проекта](#структура-проекта)  
- [Установка и запуск](#установка-и-запуск)  
- [API эндпоинты](#api-эндпоинты)  
- [Примеры запросов](#примеры-запросов)  

---

## Описание

- Загружает CSV с сигналами **ЧСС плода (fhr)** и **активности матки (uterus)**  
- Эмулирует реальное время:  
  - *моменты (moments_batch)* идут каждые `interval_sec` секунд  
  - *аннотации* приходят строго каждые 30 секунд **модельного времени**  
- Поддержка SSE (Server-Sent Events) для подписки на поток  

---

## Структура проекта

```
.
├── api_server.py       # FastAPI приложение
├── processing.py       # Логика обработки сигналов (алгоритмы)
├── requirements.txt    # Зависимости Python
├── Dockerfile          # Docker-образ
├── docker-compose.yml  # Оркестрация
└── README.md           # Документация
```

---

## Установка и запуск

### 1. Склонировать репозиторий
```bash
git clone https://github.com/your-org/ctg-monitor.git
cd ctg-monitor
```

### 2. Запуск через Docker Compose
```bash
docker compose up --build -d
```

### 3. Проверка работы
API будет доступно по адресу:  
[http://localhost:8000](http://localhost:8000)  
Документация Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)  

---

## API эндпоинты

### `GET /`
Информация о сервисе и доступных эндпоинтах.

### `POST /api/upload`
Загрузка двух CSV (`fhr_file`, `uterus_file`) и запуск эмуляции.  
Возвращает `monitor_id` и ссылку для подключения к стриму.

### `GET /api/stream/{monitor_id}`
Подключение к потоку **мгновенных моментов и аннотаций** (формат SSE).  

### `GET /api/monitors`
Список активных и завершённых мониторов.  

### `POST /api/instant`
Мгновенная обработка данных без стрима.  
Возвращает полный список `moments` и одну полную `annotation`.  

---

## Примеры запросов

### Загрузка данных и запуск стрима
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "fhr_file=@fhr.csv" \
  -F "uterus_file=@uterus.csv"
```

Ответ:
```json
{
  "monitor_id": "a1b2c3d4-...",
  "points": 1500,
  "interval_sec": 1.0,
  "speed": 1.0,
  "annotation_period_model_sec": 30.0,
  "stream_url": "/api/stream/a1b2c3d4-..."
}
```

### Подключение к стриму
```bash
curl -N http://localhost:8000/api/stream/a1b2c3d4-...
```

### Мгновенная обработка
```bash
curl -X POST "http://localhost:8000/api/instant" \
  -F "fhr_file=@fhr.csv" \
  -F "uterus_file=@uterus.csv"
```

---

## Формат CSV

CSV должен содержать один из вариантов:  

1. Один столбец:
```csv
value
123
124
122
```

2. Два столбца:
```csv
time_sec,value
0.0,123
0.5,124
1.0,122
```
