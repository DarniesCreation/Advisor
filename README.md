# AI Model Advisor v3.1

Умный подбор AI-моделей по описанию задачи.

## Установка

```bash
pip install -r requirements.txt
```

## Использование

### CLI (интерактивный режим)
```bash
python run.py cli
```

### API (FastAPI)
```bash
uvicorn advisor.api.main:app --reload
```

### Тесты
```bash
python run.py test
```

## Структура

```
advisor/
├── data/models.py       # 35 AI-моделей с метаданными
├── nlp/parser.py        # keyword + semantic парсинг запросов
├── core/
│   ├── recommender.py   # движок рекомендаций
│   └── network.py       # нейросеть (PyTorch)
├── api/main.py          # FastAPI эндпоинты
└── cli/main.py          # Rich CLI интерфейс
```

## API эндпоинты

- `POST /recommend` — рекомендации по описанию задачи
- `GET /models` — список всех моделей
- `GET /models/{id}` — информация о модели
- `POST /compare` — сравнение нескольких моделей
- `GET /alternatives/{id}` — альтернативы для модели
# Advisor
